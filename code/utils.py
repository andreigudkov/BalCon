import numpy as np
from time import perf_counter
from enum import Enum


class TimeOut:
    def __init__(self, limit, shift=0):
        self._shift = shift
        self._start = perf_counter()
        self._limit = limit

    def exceeded(self):
        if self._limit is None:
            return False
        if perf_counter() - self._start < self._limit:
            return False
        return True

    def remaining(self):
        return self._start + self._limit - perf_counter()

    def get_time(self, shift=0):
        return '{:4.2f}'.format(perf_counter() - self._start + shift)

    def print_time(self):
        if self._shift == 0:
            return f'[{self.get_time()}]'
        else:
            return f'[{self.get_time(self._shift)} | {self.get_time()}]'

    def elapsed(self):
        return perf_counter() - self._start


def NumpyConverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()


class AttemptResult(Enum):
    SUCCESS = 0
    FAIL = 1
    WORSE = 2

