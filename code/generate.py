import numpy as np
import pathlib
from typing import List, Tuple
from numpy.random import Generator
from itertools import product
from tqdm import trange

from dataclasses import replace
from environment import Environment, Host, VM, Mapping, Settings, Resources
from firstfit import FirstFit


HOST_SIZE = (88, 702 * 1024)
FLAVORS_CPU = (1, 2, 4, 6, 8, 12, 16, 24, 32, 64)
FLAVORS_RATIO = (1, 2, 4)
N_VMS_RANGE = (500, 5000)


def predefined_flavors() -> List[VM]:
    """ Predefined set of flavors used in synthetic problem instances """
    return [VM(cpu, cpu * ratio * 1024)
            for cpu, ratio in product(FLAVORS_CPU, FLAVORS_RATIO)]


def remove_inactive_hosts(env: Environment) -> Environment:
    """ Removes hosts with no VMs from the problem instance """
    active_hids = list(set(env.mapping.mapping))
    old2new = {old_hid: new_hid for new_hid, old_hid in enumerate(active_hids)}
    return Environment(
            hosts=[replace(env.hosts[old_hid]) for old_hid in active_hids],
            vms=env.vms,
            mapping=Mapping([old2new[hid] for hid in env.mapping.mapping])
    )


def determine_host_size(vms: List[VM]) -> Tuple[int, int]:
    """
    Starting from predefined host size adjust host size so that none of
    resources are dominated
    """
    required_nr = np.sum([[vm.cpu, vm.mem] for vm in vms], axis=0)
    n_nr = required_nr / np.array(HOST_SIZE)
    resource = Resources.CPU if n_nr[Resources.CPU] < n_nr[Resources.MEM] else Resources.MEM
    return np.round(required_nr / n_nr[resource])


def generate_vms(n_vms: int, rng: Generator) -> List[VM]:
    """
    Generate random distribution on the set of predefined flavors
    such that small flavors are more likely
    than sample <n_vms> VMs from the distribution
    """
    flavors = predefined_flavors()
    cpu_nf = np.array([flavor.cpu for flavor in flavors])
    weights = -np.log(rng.random(len(flavors))) / cpu_nf
    weights /= np.sum(weights)
    return rng.choice(flavors, n_vms, p=weights)


def generate_instance(n_vms: int, rng: Generator) -> Environment:
    """ Generate problem instances with heavy resource imbalance """
    vms = generate_vms(n_vms, rng)
    cpu, mem = determine_host_size(vms)
    hosts = [Host(cpu, mem) for _ in range(n_vms)]

    env = Environment(hosts=hosts, vms=vms, mapping=Mapping.emtpy(n_vms))
    scheduler = FirstFit(env, Settings(), sorting=FirstFit.SortingRule.RATIO)
    env.mapping = scheduler.solve().mapping
    env = remove_inactive_hosts(env)

    assert env.validate_mapping(env.mapping)
    return env


def generate_synthetic_dataset(n_instances: int, path: str) -> None:
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(0)
    for i in trange(n_instances):
        n_vms = rng.integers(N_VMS_RANGE[0], N_VMS_RANGE[1] + 1)
        env = generate_instance(n_vms=n_vms, rng=rng)
        problem_path = path / f'n{i}v{n_vms}.json'
        env.save(problem_path)


if __name__ == '__main__':
    generate_synthetic_dataset(200, 'data/synthetic')
