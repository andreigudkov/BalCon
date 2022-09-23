import numpy as np
import json
from typing import Optional, List
from dataclasses import dataclass, asdict

from utils import TimeOut, NumpyConverter
from environment import Environment, Settings, Mapping, VM, Resources
from assignment import Assignment
import logging


@dataclass
class Solution:
    mapping: Optional[Mapping] = None
    objective: Optional[float] = None
    savings_score: Optional[float] = None    # is the number of active hosts in the solution
    migration_score: Optional[float] = None  # is the amount of migrated memory in the solution
    status: Optional[str] = None
    elapsed: float = 0

    def save(self, path) -> None:
        with open(path, 'w') as fd:
            json.dump(asdict(self), fd, indent=4, default=NumpyConverter)


class Score:
    def __init__(self, env: Environment, settings: Settings) -> None:
        self.env = env
        self.settings = settings

        
    def objective(self, solution: Solution):
        wa = self.settings.wa
        wm = self.settings.wm
        solution.objective = 0
        solution.objective += wa * self.savings_score(solution)
        solution.objective += wm * self.migration_score(solution)
        return solution.objective

    def repr_objective(self, solution: Solution) -> str:
        O = self.objective(solution)
        A = self.savings_score(solution)
        M = self.migration_score(solution)
        wa = self.settings.wa
        wm = self.settings.wm
        return f'{O:.4f} ({wa:.2f}*{A} + {wm:.2f}*{M:.2f})'

    def savings_score(self, solution: Solution) -> float:
        """ Number of active hosts """
        if solution.mapping is not None:
            active_hosts = list(set([
                    hid for hid in solution.mapping.mapping if hid != -1
            ]))
            solution.savings_score = len(active_hosts)
        return solution.savings_score

    def repr_savings_score(self, solution) -> str:
        initial_solution = Solution(mapping=self.env.mapping)
        A_init = self.savings_score(initial_solution)
        A = self.savings_score(solution)
        return f'{A} (+{A_init-A} empty hosts ' \
               f'{A_init}->{A}/{len(self.env.hosts)})'

    def migration_score(self, solution: Solution) -> float:
        """ Amount of migrated memory in terabytes """
        if solution.mapping is not None:
            mem_moved_mb = 0
            for vmid, vm in enumerate(self.env.vms):
                src_hid = self.env.mapping[vmid]
                dst_hid = solution.mapping[vmid]
                if src_hid != dst_hid:
                    mem_moved_mb += vm.mem
            solution.migration_score = mem_moved_mb / 1024 / 1024
        return solution.migration_score

    def repr_migration_score(self, solution) -> str:
        score = f'{self.migration_score(solution):.2f} TiB'
        additional_info = ''

        if solution.mapping is not None:
            initial_solution = Solution(mapping=self.env.mapping)
            M_max = get_terabytes(self.env.vms)

            inner_migrations, outer_migrations = get_migrations(
                    initial_solution,
                    solution
            )
            M_outer = get_terabytes_from_vmids(self.env, outer_migrations)
            M_inner = get_terabytes_from_vmids(self.env, inner_migrations)

            n_vms = len(self.env.vms)
            n_outer = len(outer_migrations)
            n_inner = len(inner_migrations)

            migrated_vms = f'{n_outer}+{n_inner}/{n_vms} VMs'
            migrated_mem =  f'{M_outer:.2f}+{M_inner:.2f}/{M_max:.2f} TiB'

            additional_info = f'({migrated_vms}, {migrated_mem})'

        return f'{score} {additional_info}'


def get_terabytes_from_vmids(env: Environment, vmids: List[int]) -> float:
    return sum([env.vms[vmid].mem for vmid in vmids]) / 1024 / 1024


def get_terabytes(vms: List[VM]) -> float:
    return sum([vm.mem for vm in vms]) / 1024 / 1024


def get_migrations(initial_solution: Solution, solution: Solution) -> (List[VM], List[VM]):
    """
    This function returns VMs for induced migrations in inner_migrations list 
    and list of rest migrations in outer_migrations list.
    """
    outer_migrations = []
    inner_migrations = []
    active_hids = set(solution.mapping.mapping)
    for vmid in range(len(solution.mapping.mapping)):
        src_hid = initial_solution.mapping[vmid]
        dst_hid = solution.mapping[vmid]
        if src_hid != dst_hid:
            if src_hid in active_hids:
                inner_migrations.append(vmid)
            else:
                outer_migrations.append(vmid)
    return inner_migrations, outer_migrations


class Algorithm:
    """ Base class with utilities """

    def __init__(self, env: Environment, settings: Settings) -> None:
        self.env = env
        self.settings = settings
        self.asg = Assignment(env, settings)

        self.n_vms = len(env.vms)
        self.n_hosts = len(env.hosts)

        self.tl = TimeOut(0)

    def solve_(self) -> Mapping:
        ...

    def solve(self) -> Solution:
        self.tl = TimeOut(self.settings.tl)
        return Solution(mapping=self.solve_())

    def log(self, message:str) -> None:
        logging.info(f'[{self.tl.get_time()}] {message}')

    def will_objective_improve(self, hid: int) -> bool:
        """
        Check if objective will improve after release of the host
        assuming that no additional migrations will be performed
        """
        asg = self.asg
        mask = (asg.mapping == asg.init_mapping) & (asg.mapping == hid)
        migration_cost = np.sum(asg.required_nv_nr[mask, Resources.MEM])
        migration_cost = migration_cost / 1024 / 1024
        wa = self.settings.wa
        wm = self.settings.wm
        return migration_cost * wm < 1 * wa

    def can_host_be_emptied(self, hid: int, hids: List[int]) -> bool:
        """ Relaxed check if the VMs from <hid> host can be moved on <hids> """
        required = self.asg.occupied_nh_nr[hid]
        remained = np.sum(self.asg.remained_nh_nr[hids], axis=0)
        return np.all(remained >= required)


