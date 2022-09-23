from enum import Enum
from typing import List
import numpy as np
from ortools.linear_solver import pywraplp

from environment import Environment, Settings, Mapping
from algorithm import Solution, Algorithm


class Status(Enum):
    OPTIMAL = pywraplp.Solver.OPTIMAL
    FEASIBLE = pywraplp.Solver.FEASIBLE
    INFEASIBLE = pywraplp.Solver.INFEASIBLE


class Model(Algorithm):
    def __init__(self, env: Environment, settings: Settings, backend='CBC'):
        super().__init__(env, settings)
        self.backend = backend
        self.model = None
        self.A = None
        self.M = None

    def initialize_model(self) -> None:
        self.model = pywraplp.Solver.CreateSolver(self.backend)
        self.model.SetNumThreads(1)
        self.model.set_time_limit(int(self.settings.tl * 1000))

    def solve(self) -> Solution:
        self.initialize_model()
        self.add_variables()
        self.add_initial_solution()
        self.add_constraints()
        return self.optimize()

    def add_variables(self) -> None:
        ...

    def add_initial_solution(self) -> None:
        ...

    def add_constraints(self) -> None:
        ...

    def optimize(self) -> Solution:
        ...


class AllocationModel(Model):
    """
    Straightforward MILP model
    introduced in Bartok-Mann'15:
    'A branch-and-bound approach to virtual machine placement'
    """
    def __init__(self, env: Environment, settings: Settings, backend='CBC'):
        super().__init__(env, settings, backend)
        self.alloc_nv_nh = None
        self.active_nh = None
        self.migr_nv = None

    def add_variables(self) -> None:
        """
        Decision variables:
            alloc[i][j] - whenever vm[i] is allocated on host[j]
            active[j]   - whenever host[j] is active (i.e. non-empty)
            migr[i]     - whenever vm[i] is migrated
                        (i.e. moved out from initial location)
        Score variables:
            A - savings score: amount of active hosts
            M - migration score: amount of migrated memory in terabytes
        """
        self.alloc_nv_nh = [[self.model.BoolVar(f'alloc_{vmid}_{hid}')
                             for vmid in range(self.n_hosts)]
                            for hid in range(self.n_vms)]
        self.active_nh = [self.model.BoolVar(f'active_{hid}')
                          for hid in range(self.n_hosts)]
        self.migr_nv = [self.model.BoolVar(f'migr_{vmid}')
                        for vmid in range(self.n_vms)]

        infinity = self.model.infinity()
        self.A = self.model.NumVar(0.0, infinity, name='A')
        self.M = self.model.NumVar(0.0, infinity, name='M')

        self.model.Minimize(
                self.settings.wa * self.A + self.settings.wm * self.M
        )

    def add_score_constraints(self) -> None:
        migrated_memory_terabytes = sum([
                (self.env.vms[vmid].mem / 1024 / 1024) * self.migr_nv[vmid]
                for vmid in range(self.n_vms)
        ])

        self.model.Add(
                sum(self.active_nh) == self.A
        )
        self.model.Add(
                migrated_memory_terabytes == self.M
        )

    def add_allocation_constraints(self) -> None:
        """
        (1) Each VM must be allocated on exactly one host
        (2) VM is not migrated if it is allocated on it's initial place
        (3) VM can be allocated only on active host
        """
        for vmid in range(self.n_vms):
            self.model.Add(
                    sum(self.alloc_nv_nh[vmid]) == 1
            )

            source = self.env.mapping[vmid]
            self.model.Add(
                    self.migr_nv[vmid] == 1 - self.alloc_nv_nh[vmid][source]
            )

            for hid in range(self.n_hosts):
                self.model.Add(
                        self.alloc_nv_nh[vmid][hid] <= self.active_nh[hid]
                )

    def add_capacity_constraints(self) -> None:
        for hid in range(self.n_hosts):
            capacity_cpu = self.env.hosts[hid].cpu
            occupied_cpu = sum([
                    self.env.vms[vmid].cpu * self.alloc_nv_nh[vmid][hid]
                    for vmid in range(self.n_vms)
            ])
            capacity_mem = self.env.hosts[hid].mem
            occupied_mem = sum([
                    self.env.vms[vmid].mem * self.alloc_nv_nh[vmid][hid]
                    for vmid in range(self.n_vms)
            ])

            self.model.Add(
                    occupied_cpu <= capacity_cpu
            )
            self.model.Add(
                    occupied_mem <= capacity_mem
            )

    def add_constraints(self) -> None:
        self.add_score_constraints()
        self.add_allocation_constraints()
        self.add_capacity_constraints()

    def get_mapping(self) -> Mapping:
        mapping = [-1 for _ in range(self.n_vms)]
        for vmid in range(self.n_vms):
            for hid in range(self.n_hosts):
                if self.alloc_nv_nh[vmid][hid].solution_value() == 1:
                    mapping[vmid] = hid
        return Mapping(mapping)

    def optimize(self) -> Solution:
        status = self.model.Solve()
        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            return Solution(
                    mapping=self.get_mapping(),
                    status=Status(status).name
            )
        else:
            return Solution(
                    mapping=None,
                    status=Status.INFEASIBLE.name
            )

    def add_initial_solution(self) -> None:
        mapping = self.env.mapping.mapping
        variables = []
        values = []

        for vmid in range(self.n_vms):
            for hid in range(self.n_hosts):
                variables.append(self.alloc_nv_nh[vmid][hid])
                values.append(1. if mapping[vmid] == hid else 0.)

        for hid in range(self.n_hosts):
            variables.append(self.active_nh[hid])
            values.append(1.)

        for vmid in range(self.n_vms):
            variables.append(self.migr_nv[vmid])
            values.append(0.)

        self.model.SetHint(variables, values)


class FlowModel(Model):
    """
    Flavor Flow Model
    The model based on flavors (VM types) of VMs (vm.cpu, vm.cpu) instead of
    directly VMs and can be applied in case of small amount of different types
    """
    class Mode(Enum):
        INTEGER = 0
        HALF_RELAXED = 1

    def __init__(self, env: Environment, settings: Settings, backend='CBC',
                 mode=Mode.INTEGER):
        super().__init__(env, settings, backend)
        self.mode = mode

        self.flavors = list(set(self.env.vms))
        self.n_flavors = len(self.flavors)
        self.vm_to_fid = {vm: fid for fid, vm in enumerate(self.flavors)}
        self.count_nf_nh = self.count_flavors()

        self.in_ = None
        self.out = None
        self.active = None

    def count_flavors(self) -> List[List[int]]:
        """
        Count amount of each flavor on each host in the initial solution
        """
        count_nf_nh = [[0 for _ in range(self.n_hosts)] for _ in self.flavors]

        for vmid, vm in enumerate(self.env.vms):
            hid = self.env.mapping[vmid]
            fid = self.vm_to_fid[vm]
            count_nf_nh[fid][hid] += 1
        return count_nf_nh

    def add_variables(self) -> None:
        """
        Integer variables:
            in_[i][j]   - amount of VMs of flavor[i] moved into host[j]
            out[i][j]   - amount of VMs of flavor[i] moved out of host[j]
            active[j]   - whenever host[j] is active

        HALF-RELAXED MODE:
            in_[i][j], out[i][j] variables are relaxed in order decrease
            the model complexity, the active[j] kept BINARY

        Score variables:
            A - savings score: amount of active hosts
            M - migration score: amount of migrated memory in terabytes
        """
        var_type_class = {
                FlowModel.Mode.INTEGER: self.model.IntVar,
                FlowModel.Mode.HALF_RELAXED: self.model.NumVar
        }[self.mode]

        self.in_ = [[
                var_type_class(0, self.n_vms, name=f'in_{fid}_{hid}')
                for hid in range(self.n_hosts)]
                for fid in range(self.n_flavors)
        ]
        self.out = [[
                var_type_class(0, self.n_vms, name=f'out_{fid}_{hid}')
                for hid in range(self.n_hosts)]
                for fid in range(self.n_flavors)
        ]

        self.active = [self.model.BoolVar(f'active_{hid}')
                       for hid in range(self.n_hosts)]

        infinity = self.model.infinity()
        self.A = self.model.NumVar(0.0, infinity, name='A')
        self.M = self.model.NumVar(0.0, infinity, name='M')

        self.model.Minimize(self.settings.wa * self.A + self.settings.wm * self.M)

    def add_score_constraints(self) -> None:
        migrated_memory_terabytes = sum([
                (self.flavors[fid].mem / 1024 / 1024) * self.out[fid][hid]
                for fid in range(self.n_flavors)
                for hid in range(self.n_hosts)
        ])
        self.model.Add(
                sum(self.active) == self.A
        )
        self.model.Add(
                migrated_memory_terabytes == self.M
        )

    def add_count_constraints(self) -> None:
        """
        (1) You can't move out more VMs of flavor[i] from the host than
            it was originally on the host
        (2) If host is not active all VMs must be moved out
        """
        for fid in range(self.n_flavors):
            for hid in range(self.n_hosts):
                self.model.Add(
                        self.out[fid][hid] <= self.count_nf_nh[fid][hid]
                )
                self.model.Add(
                        self.count_nf_nh[fid][hid] * (1 - self.active[hid]) <= self.out[fid][hid]
                )

    def add_capacity_constraints(self) -> None:
        for hid in range(self.n_hosts):
            capacity_cpu = self.env.hosts[hid].cpu
            occupied_cpu = sum([
                    self.flavors[fid].cpu * (
                            self.count_nf_nh[fid][hid]
                            + self.in_[fid][hid]
                            - self.out[fid][hid]
                    )
                    for fid in range(self.n_flavors)
            ])

            self.model.Add(
                    occupied_cpu <= capacity_cpu * self.active[hid]
            )

            capacity_mem = self.env.hosts[hid].mem
            occupied_mem = sum([
                    self.flavors[fid].mem * (
                            self.count_nf_nh[fid][hid]
                            + self.in_[fid][hid]
                            - self.out[fid][hid]
                    ) for fid in range(self.n_flavors)
            ])
            self.model.Add(
                    occupied_mem <= capacity_mem * self.active[hid]
            )

    def add_flow_constraints(self) -> None:
        """
        Flow Preservation Constraint
        For each flavor the amount of VMs of that flavor move out of the hosts
        must coincide with the amount of VMs moved into the hosts
        """
        for fid in range(self.n_flavors):
            self.model.Add(
                    sum([
                            self.out[fid][hid]
                            for hid in range(self.n_hosts)
                    ]) == sum([
                            self.in_[fid][hid]
                            for hid in range(self.n_hosts)
                    ])
            )

    def add_constraints(self) -> None:
        self.add_score_constraints()
        self.add_count_constraints()
        self.add_capacity_constraints()
        self.add_flow_constraints()

    def optimize(self) -> Solution:
        status = self.model.Solve()
        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            if self.mode == FlowModel.Mode.INTEGER:
                return Solution(
                        mapping=self.get_mapping(),
                        status=Status(status).name
                )
            else:
                return Solution(
                        status=Status(status).name,
                        savings_score=self.A.solution_value(),
                        migration_score=self.M.solution_value(),
                        objective=self.model.Objective().Value(),
                )
        else:
            return Solution(
                    mapping=None,
                    status=Status.INFEASIBLE.name
            )

    def add_initial_solution(self) -> None:
        variables = []
        values = []

        for fid in range(self.n_flavors):
            for hid in range(self.n_hosts):
                variables.append(self.in_[fid][hid])
                values.append(0.)

                variables.append(self.out[fid][hid])
                values.append(0.)

        for hid in range(self.n_hosts):
            variables.append(self.active[hid])
            values.append(1.)

        self.model.SetHint(variables, values)

    def get_mapping(self) -> Mapping:
        """
        Reconstruction of some VM-to-host mapping from correct flavor flow.
        Note that the mapping corresponding to the correct flavor flow is not
        unique.
        """
        assert self.mode == FlowModel.Mode.INTEGER

        out_nf_nh = np.array([[
                int(self.out[fid][hid].solution_value())
                for hid in range(self.n_hosts)]
                for fid in range(self.n_flavors)
        ])
        in_nf_nh = np.array([[
                int(self.in_[fid][hid].solution_value())
                for hid in range(self.n_hosts)]
                for fid in range(self.n_flavors)
        ])
        init_mapping = np.array([hid for hid in self.env.mapping])
        mapping = [-1 for _ in range(self.n_vms)]

        vmids_out_nf = [[] for _ in range(self.n_flavors)]
        for hid in range(self.n_hosts):
            vmids = np.flatnonzero(init_mapping == hid)

            for vmid in vmids:
                fid = self.vm_to_fid[self.env.vms[vmid]]
                if out_nf_nh[fid][hid] > 0:
                    out_nf_nh[fid][hid] -= 1
                    vmids_out_nf[fid].append(vmid)
                else:
                    mapping[vmid] = hid

        for hid in range(self.n_hosts):
            for fid in range(self.n_flavors):
                while in_nf_nh[fid][hid] > 0:
                    vmid = vmids_out_nf[fid].pop()
                    mapping[vmid] = hid
                    in_nf_nh[fid][hid] -= 1
        return Mapping(mapping)


class FlowModelRelaxed(FlowModel):
    def __init__(self, env: Environment, settings: Settings, backend='CBC',
                 mode=FlowModel.Mode.HALF_RELAXED):
        super().__init__(env, settings, backend, mode)
