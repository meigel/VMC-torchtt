# -*- coding: utf-8 -*-
from __future__ import division, print_function

import os, sys
import torch
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from collections import Sequence
from load_info import get_paths, load_info, load_problem, load_sampler

import uq_adf_torchtt as xe
import numpy as np
import scipy.sparse as sps

#TODO: assert a priori that that enough samples are provided

# use Łukaszyk–Karmowski metric
# LKmetric = True
# LKmetric = False
#TODO: die Idee war, die LK-Metrik oder eine alternative Metrik, wie E[- |U(y)-u(y)| ln(p(y))]
#      das geht aber nicht mit dem adf...

def save_TT(fname, TT, **kwargs):
    if isinstance(TT, Sequence):
        components = TT
    elif hasattr(TT, 'cores'):
        components = [c.detach().cpu().numpy() for c in TT.cores]
    else:
        raise ValueError()

    keys = map("cmp_{}".format, range(len(components)))
    items = dict(zip(keys, components))
    items['format'] = 'TT'
    items['num_cmp'] = len(components)
    # links = [(i,i+1) for i in range(len(components)-1)]
    # items['links']
    items.update(kwargs)
    np.savez_compressed(fname, **items)


def load_TT(fname):
    import os
    import sys
    import torch
    try:
        import torchtt
    except ImportError:
        _here = os.path.dirname(__file__)
        _torchtt_path = os.path.join(_here, '..', 'torchTT')
        if _torchtt_path not in sys.path:
            sys.path.insert(0, _torchtt_path)
        import torchtt

    TTz = np.load(fname)
    items = dict(TTz.items())
    assert items.get('format') == 'TT'
    items.pop('format', None)

    num_cmp = int(items.pop('num_cmp'))
    components = [items.pop('cmp_%d' % i) for i in range(num_cmp)]
    cores = [torch.tensor(c) for c in components]
    TT = torchtt.TT(cores)
    return TT, items


if __name__=='__main__':
    descr = '''Perform UQ tensor reconstruction with specified samples and parameters.
Results will be written to `RECO_PATH/reconstruction.npz`.
'''
    problem_path, reco_path = get_paths(descr,
            ('SAMPLE_PATH', 'path to the directory containing the `info.json` with which the samples where generated'),
            ('RECO_PATH', 'path to the directory containing the `info.json` for reconstruction'))

    problem_info = load_info(problem_path)

    if problem_info['sampling']['distribution'] == 'uniform':
        basis = xe.PolynomBasis.Legendre
        basis_name = "Legendre"
    elif problem_info['sampling']['distribution'] == 'normal':
        basis = xe.PolynomBasis.Hermite
        basis_name = "Hermite"
    else:
        print("ERROR: unknown distribution '{}'".format(problem_info['distribution']))
        exit(1)

    spl_path = os.path.join(problem_path, 'samples', '{}.npz')

    reco_info = load_info(reco_path)
    device_str = os.environ.get('VMC_DEVICE') or reco_info.get('device')
    device = torch.device(device_str) if device_str else None

    n_samples = reco_info['sample size']
    target_eps = reco_info['tolerance']
    max_itr = reco_info['maximum number of iterations']
    order = reco_info['basis dimension']
    ortho = reco_info['orthogonalization']
    us_key = 'nus' if ortho else 'us'

    us = []
    ys = []
    n_loaded = 0
    f_idx = 0
    print('\r[ 0%] Loading Data', end='')
    while n_loaded < n_samples:
        print('\r[{0:2d}%] Loading Data ({1} files read, {2} samples collected)'.format(int(100*n_loaded/n_samples), f_idx+1, n_loaded), end='')
        sys.stdout.flush()

        if not os.path.exists(spl_path.format(f_idx)):
            break
        splz = np.load(spl_path.format(f_idx))

        ys.append(splz['ys'])
        us.append(splz[us_key])
        assert len(us[-1]) == len(ys[-1])

        n_loaded += len(us[-1])
        f_idx += 1
    print('\r[done] Loading Data ({} files read, {} samples collected)'.format(f_idx, n_loaded))
    if n_loaded < n_samples:
        print('Not enough data. {} samples missig.'.format(n_samples - n_loaded))
    us = np.concatenate(us)[:n_samples]
    ys = np.concatenate(ys)[:n_samples]

    x_dim = us.shape[1]                                 # nodes in physical space
    poly_dim = [order]*ys.shape[1]                      # polynomial degree to fit one dimensional stochastic representation
    dimension = [x_dim] + poly_dim                      # define tensor dimensions

    print("Basis of the Tensor:", basis_name)
    print("Dimension of the Tensor:", dimension)

    print('\r[ 0%] Creating Measurement Set', end='')
    measurements = xe.UQMeasurementSet()                # create measurement set
    for i,y,u in zip(range(len(ys)), ys, us):           # for every sample
        print('\r[{0:2d}%] Creating Measurement Set'.format(int(100*i/len(ys))), end='')
        sys.stdout.flush()

        measurements.add(y, u)                          # add parameter and sample result to measurement list
    print('\r[done] Creating Measurement Set')

    #                                                   # run adf algorithm
    result = xe.uq_ra_adf(measurements, basis, dimension, target_eps, max_itr, device=device)

    components = [c.detach().cpu().numpy() for c in result.cores]

    if ortho:
        chol_path = os.path.join(problem_path, 'cholesky.npz')
        chol_perm_path = os.path.join(problem_path, 'cholesky_permutation.npz')

        L = sps.load_npz(chol_path)
        P = sps.load_npz(chol_perm_path)

        # from scipy.sparse.linalg import spsolve
        # c0 = np.array(spsolve(L, c0.reshape(c0s[1:]))).reshape(c0s)
        from scipy.sparse.linalg import spsolve_triangular
        c0 = components[0]
        c0s = c0.shape
        c0.shape = c0s[1:]
        c0 = spsolve_triangular(P.dot(L).tocsr(), P.dot(c0))
        c0 = np.array(c0).reshape(c0s)
        components[0] = c0

    tt_name = os.path.join(reco_path, 'reconstruction.npz')
    save_TT(tt_name, components, orthogonalized=ortho)
