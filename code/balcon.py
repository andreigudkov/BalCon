from assignment import Assignment
from environment import Environment, Settings, Mapping
from algorithm import Algorithm
from utils import AttemptResult

import numpy as np
import heapq

from collections.abc import Iterable
from enum import Enum
from typing import List, Optional


class Situation(Enum):
    IMPOSSIBLE = 0
    AMPLE = 1
    BALANCED = 2
    LOPSIDED = 3


class Stash:
    """
    A priority queue that kept VMs sorted by size.
    The "form of the stash" i.e. cumulative vector of VMs requirements is
    kept precomputed.
    """

    def __init__(self, asg: Assignment, vmids=None) -> None:
        self.asg = asg
        self.vmids = list()
        self.form = np.zeros(2)
        self.add(vmids)

    def add(self, vmids: Iterable) -> None:
        for vmid in vmids:
            self.add_vmid(vmid)

    def add_vmid(self, vmid) -> None:
        heapq.heappush(self.vmids, (self.asg.size_nv[vmid], vmid))
        self.form += self.asg.required_nv_nr[vmid]

    def pop(self) -> int:
        """ Remove largest VM from the stash and return its index """
        _, vmid = heapq.heappop(self.vmids)
        self.form -= self.asg.required_nv_nr[vmid]
        return vmid

    def get_form(self) -> np.array:
        return self.form

    def is_empty(self) -> bool:
        return len(self.vmids) == 0


class Prohibitor:
    """
    If one host is chosen more than two times in a row, another host is chosen.
    All possible hosts will be chosen in a long run.
    """

    def __init__(self, n_hosts):
        self.last_hid = -1
        self.last_hid_counter = 0
        self.tabu_score = np.full(n_hosts, 0, dtype=np.float)

    def forbid_long_repeats(self, hids, hid):
        if hid == self.last_hid:
            self.last_hid_counter += 1
            if self.last_hid_counter > 2:
                hid = hids[np.argmax(self.tabu_score[hids])]
                self.tabu_score[hid] -= 1
        else:
            self.last_hid = hid
            self.last_hid_counter = 0
        return hid


class Coin:
    def __init__(self) -> None:
        self.state = 0

    def flip(self) -> int:
        self.state = (self.state + 1) % 2
        return self.state


def tan(v: np.array) -> float:
    return v[0] / v[1]


class ForceFit:
    def __init__(self, asg: Assignment, max_force_steps=4000):
        self.asg = asg
        self.coin = Coin()
        self.force_steps_counter = 0
        self.max_force_steps = max_force_steps
        self.alpha = 0.95

    def place_vmids(self, vmids: List[int], hids: List[int]) -> bool:
        stash = Stash(asg=self.asg, vmids=vmids)
        prohibitor = Prohibitor(n_hosts=len(self.asg.env.hosts))
        self.force_steps_counter = 0
        while not stash.is_empty():
            vmid = stash.pop()
            situation = self.classify(hids, stash, vmid)
            if situation == Situation.IMPOSSIBLE:
                return False
            elif situation == Situation.AMPLE:
                self.best_fit(vmid, hids)
            else:
                self.force_steps_counter += 1
                if self.force_steps_counter >= self.max_force_steps:
                    return False
                hids_filtered = self.filter_hosts(vmid, hids)
                if situation == Situation.BALANCED:
                    hid = self.choose_hid_balanced(vmid, hids_filtered)
                    hid = prohibitor.forbid_long_repeats(hids_filtered, hid)
                    vmids = self.choose_vmids_balanced(hid)
                else:
                    hid = self.choose_hid_lopsided(vmid, hids_filtered)
                    hid = prohibitor.forbid_long_repeats(hids_filtered, hid)
                    vmids = self.choose_vmids_lopsided(hid, vmid)
                residue = self.push_vmid(vmid, hid, vmids)
                stash.add(residue)
        return stash.is_empty()

    def classify(self, hids: List[int], stash: Stash, vmid: int) -> Situation:
        if np.any(self.asg.is_feasible_nh(vmid)[hids]):
            return Situation.AMPLE

        hids_filtered = self.filter_hosts(vmid, hids)
        if len(hids_filtered) == 0:
            return Situation.IMPOSSIBLE

        f_nr = stash.get_form() + self.asg.required_nv_nr[vmid]
        rem_nh_nr = self.asg.remained_nh_nr[hids]
        capacity = np.sum(np.amin(rem_nh_nr / f_nr, axis=1))
        potential_capacity = np.amin(np.sum(rem_nh_nr, axis=0) / f_nr)

        if potential_capacity < 1.0:
            return Situation.IMPOSSIBLE
        elif capacity >= 1.0 and capacity >= self.alpha * potential_capacity:
            return Situation.BALANCED
        else:
            return Situation.LOPSIDED

    def select_tightest_hid(self, hids: np.array) -> int:
        load_nh = np.sum(
                self.asg.occupied_nh_nr[hids] / self.asg.capacity_nh_nr[hids],
                axis=1
        )
        return hids[np.argmax(load_nh)]

    def best_fit(self, vmid: int, hids: np.array) -> None:
        feasible_nh = self.asg.is_feasible_nh(vmid)[hids]
        feasible_hids = np.asarray(hids)[feasible_nh]
        tightest_hid = self.select_tightest_hid(feasible_hids)
        self.asg.include(vmid, tightest_hid)

    def count_smaller_vms_nh(self, vmid: int, hids: np.array) -> np.array:
        size = self.asg.size_nv[vmid]
        smaller_vmids = self.asg.vmids[(self.asg.size_nv <= size) &
                                       (self.asg.mapping != -1)]
        count = np.bincount(self.asg.mapping[smaller_vmids])
        result = np.zeros(max(hids) + 1)
        result[:count.size] = count
        return result[hids]

    def choose_hid_lopsided(self, vmid: int, hids: np.array) -> int:
        rid = self.coin.flip()
        hid = self.choose_hid_lopsided_rid(vmid, hids, rid)
        if hid is not None:
            return hid

        rid = (rid + 1) % 2
        hid = self.choose_hid_lopsided_rid(vmid, hids, rid)
        if hid is not None:
            return hid

        # all hosts have the same tan as VM
        free_nh_nr = self.asg.remained_nh_nr[hids] / self.asg.capacity_nh_nr[hids]
        free_nh = np.sum(free_nh_nr, axis=1)
        idx = np.argmax(free_nh)
        return hids[idx]

    def filter_hosts(self, vmid: int, hids: np.array) -> np.array:
        """ Filter hosts that can't host the VM in any case """
        mask = np.all(
                self.asg.capacity_nh_nr[hids] >= self.asg.required_nv_nr[vmid],
                axis=1
        )
        return np.array(hids)[mask]

    def choose_hid_lopsided_rid(self, vmid: int, hids: np.array, rid: int) -> Optional[int]:
        arid = (rid + 1) % 2
        tan_vm = self.asg.required_nv_nr[vmid, rid] / self.asg.required_nv_nr[vmid, arid]
        tan_nh = self.asg.occupied_nh_nr[hids, rid] / self.asg.occupied_nh_nr[hids, arid]
        free_nh = self.asg.remained_nh_nr[hids, rid] / self.asg.capacity_nh_nr[hids, rid]
        free_nh[~(tan_nh < tan_vm)] = -1
        idx = np.argmax(free_nh)
        if free_nh[idx] >= 0:
            return hids[idx]
        return None

    def choose_vmids_lopsided(self, hid: int, in_vmid: int) -> np.array:
        vmids = self.asg.get_vmids_on_hid(hid)
        tan_in_vm = tan(self.asg.required_nv_nr[in_vmid])
        tan_host = tan(self.asg.occupied_nh_nr[hid])
        tan_nv = self.asg.required_nv_nr[:, 0] / self.asg.required_nv_nr[:, 1]

        first_key = \
            tan_nv[vmids] >= tan_in_vm \
                if tan_in_vm > tan_host else \
                tan_nv[vmids] <= tan_in_vm
        second_key = (self.asg.init_mapping == self.asg.mapping)[vmids]
        third_key = self.asg.required_nv_nr[vmids, 1]
        return vmids[np.lexsort((third_key, second_key, first_key))]

    def choose_hid_balanced(self, vmid: int, hids: np.array) -> int:
        metric_nh = self.count_smaller_vms_nh(vmid, hids)
        idx = np.argmax(metric_nh)
        return hids[idx]

    def choose_vmids_balanced(self, hid: int) -> np.array:
        vmids = self.asg.get_vmids_on_hid(hid)
        first_key = (self.asg.init_mapping == self.asg.mapping)[vmids]
        second_key = self.asg.required_nv_nr[vmids, 1]
        return vmids[np.lexsort((second_key, first_key))]

    def push_vmid(self, in_vmid: int, hid: int, vmids: np.array) -> np.array:
        ejected_vmids = list()
        for vmid in vmids:
            self.asg.exclude(vmid)
            ejected_vmids.append(vmid)
            if self.asg.is_feasible(in_vmid, hid): break

        self.asg.include(in_vmid, hid)

        residue = list()
        for vmid in ejected_vmids[::-1]:
            if self.asg.is_feasible(vmid, hid):
                self.asg.include(vmid, hid)
            else:
                residue.append(vmid)
        return residue


class BalCon(Algorithm):
    """Reference implementation of SerConFF algorithm """

    def __init__(self, env: Environment, settings: Settings) -> None:
        super().__init__(env, settings)
        self.asg = Assignment(env, settings)
        self.placer = ForceFit(self.asg, max_force_steps=settings.max_force_steps)
        self.debug = False
        self.coin = Coin()
        self.initial_memory_nh = self.asg.occupied_nh_nr[:, 1].copy()

    def clear_host(self, hid: int) -> np.array:
        vmids = self.asg.get_vmids_on_hid(hid)
        for vmid in vmids:
            self.asg.exclude(vmid)
        return vmids

    def choose_hid_to_try(self, hids: np.array) -> int:
        first_key = self.initial_memory_nh[hids]
        second_key = self.asg.occupied_nh_nr[hids, 1]
        return hids[np.lexsort((second_key, first_key))[0]]

    def log_result(self, result: AttemptResult, hid: int) -> None:
        self.log(f'try host {hid:<5}\t'
                 f'force steps: {self.placer.force_steps_counter:<7}\t'
                 f'{result}')

    def solve_(self) -> Mapping:
        allowed_hids = list(self.asg.hids)
        hosts_to_try = list(self.asg.hids)

        while hosts_to_try and not self.tl.exceeded():
            self.asg.backup()
            score = self.asg.compute_score()

            hid = self.choose_hid_to_try(hosts_to_try)
            hosts_to_try.remove(hid)
            allowed_hids.remove(hid)
            vmids = self.clear_host(hid)
            if self.placer.place_vmids(vmids, allowed_hids):
                result = AttemptResult.SUCCESS
                if self.asg.compute_score() > score:
                    result = AttemptResult.WORSE
            else:
                result = AttemptResult.FAIL

            if result != AttemptResult.SUCCESS:
                allowed_hids.append(hid)
                self.asg.restore()
            self.log_result(result, hid)

        return self.asg.get_solution()
