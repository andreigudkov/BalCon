import numpy as np
from typing import List, Set

from environment import Environment, Settings, Mapping
from assignment import Assignment, Resources
from algorithm import Algorithm
from utils import AttemptResult


class SerConOriginal(Algorithm):
    """
    Implementation of algorithm SerCon from
    Murtazaev-Oh'11 'Sercon: Server Consolidation Algorithm
    using Live Migration of Virtual Machines for Green Computing'
    """

    def calculate_score_nh(self) -> np.array:
        load_nh_nr = self.asg.occupied_nh_nr / self.asg.capacity_nh_nr
        lambda_ = np.sum(load_nh_nr[:, Resources.CPU]) / np.sum(load_nh_nr)
        cpu_summand = load_nh_nr[:, Resources.CPU] * lambda_
        mem_summand = load_nh_nr[:, Resources.MEM] * (1 - lambda_)
        return cpu_summand + mem_summand

    def calculate_score_nv(self, vmids: np.array) -> np.array:
        r_nv_nr = self.asg.required_nv_nr[vmids]
        lambda_ = np.sum(r_nv_nr[:, Resources.CPU]) / np.sum(r_nv_nr)
        cpu_summand = r_nv_nr[:, Resources.CPU] * lambda_
        mem_summand = r_nv_nr[:, Resources.MEM] * (1 - lambda_)
        return cpu_summand + mem_summand

    def get_host_to_be_released(self, unsuccessful_migration_attempts: int) -> int:
        n_active_hosts = np.sum(self.asg.occupied_nh_nr[:, Resources.MEM] != 0)
        return n_active_hosts - 1 - unsuccessful_migration_attempts

    def release_host(self, position: int) -> bool:
        sorted_hids = np.argsort(self.calculate_score_nh())[::-1]
        hid_to_be_released = sorted_hids[position]
        if not self.will_objective_improve(hid_to_be_released):
            return False

        vmids = self.asg.get_vmids_on_hid(hid_to_be_released)
        vmids = vmids[np.argsort(self.calculate_score_nv(vmids))][::-1]

        for vmid in vmids:
            self.asg.exclude(vmid)
            for hid in sorted_hids[:position]:
                if self.asg.is_feasible(vmid, hid):
                    self.asg.include(vmid, hid)
                    break
            else:
                return False
        return True

    def solve_(self) -> Mapping:
        unsuccessful_migration_attempts = 0
        self.asg.backup()

        while not self.tl.exceeded():
            hid = self.get_host_to_be_released(unsuccessful_migration_attempts)
            if hid == -1:
                break

            if not self.release_host(hid):
                status = AttemptResult.FAIL
                unsuccessful_migration_attempts += 1
                self.asg.restore()
            else:
                status = AttemptResult.SUCCESS
                unsuccessful_migration_attempts = 0
                self.asg.backup()
            self.log(f'try host {hid}\t{status}')
        return self.asg.get_solution()


class SerCon(Algorithm):
    """
    Adaptation of algorithm SerCon from
    Murtazaev-Oh'11 'Sercon: Server Consolidation Algorithm
    using Live Migration of Virtual Machines for Green Computing'
    """

    def solve_(self) -> Mapping:
        asg = self.asg

        active_hids = asg.get_active_hids()
        migration_cost_nh = asg.occupied_nh_nr[active_hids, Resources.MEM]
        hids_to_free = active_hids[np.argsort(migration_cost_nh)]
        hids_to_load = set(active_hids)

        for hid in hids_to_free:
            asg.backup()
            hids_to_load.remove(hid)
            if not self.try_to_empty_host(hid, hids_to_load):
                hids_to_load.add(hid)
                asg.restore()
                status = AttemptResult.FAIL
            else:
                status = AttemptResult.SUCCESS

            self.log(f'try host {hid}\t{status}')
            if self.tl.exceeded():
                break
        return asg.get_solution()

    def select_tightest_hid(self, hids: np.array) -> np.array:
        load_nh = np.sum(self.asg.occupied_nh_nr[hids] / self.asg.capacity_nh_nr[hids],
                         axis=1)
        return hids[np.argmax(load_nh)]

    def best_fit(self, vmid: int, hids: List[int]) -> bool:
        feasible_nh = self.asg.is_feasible_nh(vmid)[hids]
        feasible_hids = np.asarray(hids)[feasible_nh]
        if len(feasible_hids) > 0:
            tightest_hid = self.select_tightest_hid(feasible_hids)
            self.asg.include(vmid, tightest_hid)
            return True
        return False

    def try_to_empty_host(self, hid: int, hids_to_load: Set[int]) -> bool:
        if not self.will_objective_improve(hid) or \
                not self.can_host_be_emptied(hid, list(hids_to_load)):
            return False

        asg = self.asg
        vmids = asg.get_vmids_on_hid(hid)
        sizes = asg.size_nv[vmids]
        vmids = vmids[np.argsort(sizes)][::-1]

        for vmid in vmids:
            asg.exclude(vmid)
            if not self.best_fit(vmid, list(hids_to_load)):
                return False
        return True

