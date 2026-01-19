# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
from functools import partial
import argparse, os, glob
import multiprocessing as mp
from load_info import get_paths, load_info, load_problem, load_sampler
from sampleDump import SampleDumpNPZ as SampleDump
# from sampleDump import SampleDumpHDF5 as SampleDump
import time


DEBUG = False


BATCH_NUM = 0
BATCH_NUM_ACC = 0
def linear_expectation(_batchSize, _sampleDump):
    if DEBUG: print('linear_expectation')
    global BATCH_NUM
    BATCH_NUM += 1
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
    print("{} Computing batch: {} ({} accumulated | {} stored)".format(ts, BATCH_NUM, BATCH_NUM_ACC, _sampleDump.fileCount))

    if DEBUG: print('\tsampling')
    ys = sampler(_batchSize)
    ys_finite = np.all(np.isfinite(ys), axis=1)
    if not np.all(ys_finite):
        raise RuntimeError("Invalid value encountered in sampler output: {} samples are not finite.\n              Generator: {}\n              Offset: {}".format(np.sum(~ys_finite), sampler.generator, sampler.offset))

    if DEBUG:
        print('\tmapping')
        results = []
        for y in ys:
            results.append(problem.evaluate_u(y))
    else:
        pool = mp.Pool()
        results = pool.map_async(problem.evaluate_u, ys)
        results = results.get()
        pool.close()
        pool.join()
    us = np.array(results)

    print("{} Computing moments".format(ts))
    if DEBUG: print('\tdumping')
    _sampleDump << (ys, us)
    return np.hstack([np.mean(us, axis=0), np.mean(us**2, axis=0)])


def hierarchical_expectation(_fnc, _lvls, _save, _init=None, _init_level=0):
    if DEBUG: print('hierarchical_expectation')
    if _init_level < 0:
        raise ValueError("expected argument `_init_level` to be non-negative, got %d"%(_init_level))
    if _init_level > 0 and _init is None:
        raise ValueError("argument `_init` has to be provided for positive `_init_level`")
    if _lvls < 0:
        raise ValueError("expected argument `_lvls` to be non-negative, got %d"%(_lvls))
    elif _lvls <= _init_level:
        if _init is None:
            global BATCH_NUM_ACC
            ret = np.array([_fnc()])
            BATCH_NUM_ACC += 1
            return ret
        else:
            return _init

    l = hierarchical_expectation(_fnc, _lvls-1, _save, _init, _init_level)
    r = hierarchical_expectation(_fnc, _lvls-1,  None,  None,           0)
    assert l.ndim == r.ndim == 2
    assert l.shape == r.shape
    assert len(l) == len(r) == 2**(_lvls-1)

    ifactors = lambda t: t/np.arange(1,t+1) + 1
    n = l[-1] + (r-l[-1])/ifactors(len(r))[:,None]
    assert len(n) == len(r)

    ret = np.vstack([l, n])
    assert len(ret) == 2**_lvls
    if _save is not None:
        ts = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
        np.savez_compressed(_save, ret) #TODO: store m1 / m2 separately
        print("{} Saving expectation: '{}'".format(ts, _save))
    return ret


if __name__=='__main__':
    descr = """Sample solutions for the given problem and compute the first and second moment.
The samples will be written to `PATH/samples/` and the moments will be written to `PATH/moments.npz`.
"""
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('PATH', help='path to the directory containing the `info.json` for the specific problem')
    parser.add_argument('--resume', dest='RESUME', action='store_true', help='resume computation of already computed results')
    args = parser.parse_args()
    path = args.PATH
    if not os.path.isdir(path):
        raise IOError("'{}' is not a valid directory".format(path))

    info = load_info(path)
    problem = load_problem(info)
    FEM = problem.setup_FEM(info)
    sampler = load_sampler(info)

    exp_path = os.path.join(path, 'moments.npz')
    spl_path = os.path.join(path, 'samples')
    sd = SampleDump(spl_path, info['sampling']['dump limit'])

    batch_size = info['sampling']['batch size']

    init = None
    init_level = 0
    if args.RESUME:
        init = np.load(exp_path)['arr_0']
        init_level = int(np.log2(len(init)))
        BATCH_NUM_ACC = len(init)

        num_files = len(glob.glob(sd.filePath.format(fileCount='*')))
        num_samples = 0
        for f_idx in range(num_files):
            splz = np.load(sd.filePath.format(fileCount=f_idx))
            assert len(splz['ys']) == len(splz['us']) == batch_size
            num_samples += len(splz['ys'])
        sampler.offset = max(num_samples, batch_size*len(init))
        #TODO: Das geht nicht! Du kannst bei QMC keine Lücken lassen.
        sd.fileCount = num_files
        sd.dumpLimit -= num_samples

    compute_batch = partial(linear_expectation, batch_size, sd)
    us = hierarchical_expectation(compute_batch, info['sampling']['levels'], exp_path, init, init_level)
    while sd.dumpLimit > 0: linear_expectation(batch_size, sd)
