# -*- coding: utf-8 -*-
from __future__ import division, print_function
import multiprocessing as mp
import time


def ts(): return time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())


class Parallel(object):
    def __init__(self, function, cpucount=None):
        self.function = function
        self.cpucount = cpucount or mp.cpu_count()
        self.pool = mp.Pool(self.cpucount)
        self.batch_num = 0

    def __call__(self, iterable):
        chunksize = max(len(iterable) // self.cpucount, 1)
        self.batch_num += 1
        print(ts(), "Computing batch {} of '{}' (batchsize: {} | chunksize: {})".format(self.batch_num, self.function.__name__, len(iterable), chunksize))
        result = self.pool.map_async(self.function, iterable, chunksize)
        return result.get()

    def __del__(self):
        self.pool.terminate()


class Sequential(object):
    def __init__(self, function, cpucount=None):
        self.function = function
        self.batch_num = 0

    def __call__(self, iterable):
        chunksize = len(iterable)
        self.batch_num += 1
        print(ts(), "Computing batch {} of '{}' (batchsize: {} | chunksize: {})".format(self.batch_num, self.function.__name__, len(iterable), chunksize))
        result = map(self.function, iterable)
        return list(result)
