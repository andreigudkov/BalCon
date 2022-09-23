import numpy as np
from environment import Settings, Environment, Mapping, Resources


class Assignment:
    """
    Class for working with feasible solutions, but some VM can be unassigned.
    Basic operations: INCLUDE VM on a host, EXCLUDE VM
    Useful values kept precomputed as numpy arrays.
    """
    def __init__(self, env: Environment, settings: Settings) -> None:
        self.env = env
        self.settings = settings

        self.n_vms = len(env.vms)
        self.n_hosts = len(env.hosts)

        self.vmids = np.arange(self.n_vms)
        self.hids = np.arange(self.n_hosts)

        self.init_mapping = np.array(self.env.mapping.mapping)
        self.mapping = np.array(self.env.mapping.mapping)

        self.required_nv_nr = np.array([[v.cpu, v.mem] for v in env.vms])
        self.capacity_nh_nr = np.array([[h.cpu, h.mem] for h in env.hosts])

        self.occupied_nh_nr = np.zeros((self.n_hosts, 2))
        np.add.at(self.occupied_nh_nr, self.mapping, self.required_nv_nr)
        self.remained_nh_nr = self.capacity_nh_nr - self.occupied_nh_nr

        self.backup_mapping = None
        self.backup_remained_nh_nr = None
        self.backup_occupied_nh_nr = None

        total_nr = np.sum(self.required_nv_nr, axis=0)
        self.size_nv = np.sum(self.required_nv_nr / total_nr, axis=1)

    def is_assigned(self, vmid: int) -> bool:
        return self.mapping[vmid] != -1

    def exclude(self, vmid: int) -> None:
        assert self.is_assigned(vmid)
        hid = self.mapping[vmid]
        self.mapping[vmid] = -1
        self.occupied_nh_nr[hid] -= self.required_nv_nr[vmid]
        self.remained_nh_nr[hid] += self.required_nv_nr[vmid]

    def clear(self) -> None:
        for vmid in range(self.n_vms):
            if self.is_assigned(vmid):
                self.exclude(vmid)

    def include(self, vmid: int, hid: int) -> None:
        assert (self.mapping[vmid] == -1)
        assert (np.all(self.remained_nh_nr[hid] >= self.required_nv_nr[vmid]))
        self.mapping[vmid] = hid
        self.occupied_nh_nr[hid] += self.required_nv_nr[vmid]
        self.remained_nh_nr[hid] -= self.required_nv_nr[vmid]

    def backup(self) -> None:
        self.backup_mapping = self.mapping.copy()
        self.backup_occupied_nh_nr = self.occupied_nh_nr.copy()
        self.backup_remained_nh_nr = self.remained_nh_nr.copy()

    def restore(self) -> None:
        self.mapping = self.backup_mapping.copy()
        self.occupied_nh_nr = self.backup_occupied_nh_nr.copy()
        self.remained_nh_nr = self.backup_remained_nh_nr.copy()

    def compute_score(self) -> float:
        A = len(np.unique(self.mapping[self.mapping != -1]))
        moved_vm_mask = self.init_mapping != self.mapping
        M = np.sum(self.required_nv_nr[moved_vm_mask, Resources.MEM])
        return A * self.settings.wa + (M / 1024 / 1024) * self.settings.wm

    def get_vmids_on_hid(self, hid) -> np.array:
        return np.flatnonzero(self.mapping == hid)

    def is_feasible(self, vmid: int, hid:int) -> bool:
        return np.all(self.remained_nh_nr[hid] >= self.required_nv_nr[vmid])

    def is_feasible_nh(self, vmid: int) -> np.array:
        return np.all(self.remained_nh_nr >= self.required_nv_nr[vmid], axis=1)

    def get_solution(self) -> Mapping:
        return Mapping(list(self.mapping))

    def get_active_hids(self) -> np.array:
        return np.flatnonzero(self.occupied_nh_nr[:, Resources.MEM] > 0)
