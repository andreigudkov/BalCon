import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Set
import json
from utils import NumpyConverter
from enum import IntEnum


@dataclass
class Host:
    cpu: int
    mem: int


@dataclass(frozen=True)
class VM:
    cpu: int
    mem: int


@dataclass
class Mapping:
    mapping: List[int]

    def __getitem__(self, key: int) -> int:
        return self.mapping[key]

    @staticmethod
    def emtpy(n_vms:int):
        return Mapping([-1 for _ in range(n_vms)])


class Resources(IntEnum):
    CPU = 0
    MEM = 1


@dataclass
class Environment:
    hosts: List[Host]
    vms: List[VM]
    mapping: Mapping

    def save(self, path: str) -> None:
        with open(path, 'w') as fd:
            json.dump({
                    'hosts': [asdict(host) for host in self.hosts],
                    'vms': [asdict(vm) for vm in self.vms],
                    'mapping': self.mapping.mapping,
            }, fd, indent=4, default=NumpyConverter)

    @staticmethod
    def load(path: str):
        with open(path, 'r') as fd:
            data = json.load(fd)

        hosts = [Host(**host) for host in data['hosts']]
        vms = [VM(**vm) for vm in data['vms']]
        mapping = Mapping(data['mapping'])
        return Environment(hosts, vms, mapping)

    def validate_mapping(self, mapping: Mapping) -> bool:
        n_hosts = len(self.hosts)
        rem_nh_nr = np.zeros((n_hosts, 2))

        for hid in range(n_hosts):
            rem_nh_nr[hid, Resources.CPU] = self.hosts[hid].cpu
            rem_nh_nr[hid, Resources.MEM] = self.hosts[hid].mem

        for vmid, hid in enumerate(mapping.mapping):
            if hid == -1:
                return False
            rem_nh_nr[hid, Resources.CPU] -= self.vms[vmid].cpu
            rem_nh_nr[hid, Resources.MEM] -= self.vms[vmid].mem

        return np.all(rem_nh_nr >= 0)


@dataclass
class Settings:
    wa: float = 10.
    wm: float = 1.
    tl: float = 60.
    max_force_steps: int = 4000
