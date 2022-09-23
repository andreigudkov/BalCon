import numpy as np
from enum import Enum

from environment import Mapping, Environment, Settings
from assignment import Assignment, Resources
from algorithm import Algorithm


class FirstFit(Algorithm):
    class SortingRule(Enum):
        SIZE = 0
        RATIO = 1

    def __init__(self, env: Environment, settings: Settings,
                 sorting=SortingRule.SIZE):
        super().__init__(env, settings)
        self.sorting = sorting
        self.asg = Assignment(env, settings)
        self.sorted_vmids = self.sort_vms()

    def sort_vms(self) -> np.array:
        if self.sorting == FirstFit.SortingRule.SIZE:
            return np.argsort(self.asg.size_nv)[::-1]
        elif self.sorting == FirstFit.SortingRule.RATIO:
            return np.argsort(
                    self.asg.required_nv_nr[:, Resources.CPU] /
                    self.asg.required_nv_nr[:, Resources.MEM]
            )
        else:
            raise ValueError("Unknown Sorting Rule")

    def solve_(self) -> Mapping:
        asg = self.asg
        asg.clear()

        for vmid in self.sorted_vmids:
            for hid in np.flatnonzero(asg.is_feasible_nh(vmid)):
                asg.include(vmid, hid)
                break
            if not asg.is_assigned(vmid) or self.tl.exceeded():
                return self.env.mapping
        return asg.get_solution()
