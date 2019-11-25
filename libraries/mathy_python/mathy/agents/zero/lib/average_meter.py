"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
import errno
import os
import sys
import time
import math
from multiprocessing import Value

__all__ = ["AverageMeter"]


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    @property
    def val(self):
        return self._val.value

    @property
    def sum(self):
        return self._sum.value

    @property
    def count(self):
        return self._count.value

    @property
    def avg(self):
        return self._avg.value

    def reset(self):
        self._val = Value("d", 0)
        self._sum = Value("d", 0)
        self._count = Value("d", 0)
        self._avg = Value("d", 0)

    def update(self, val, n=1):
        self._val.value = val
        self._sum.value += val * n
        self._count.value += n
        self._avg.value = self.sum / self.count
