# -*- coding: utf-8 -*-
from __future__ import division, print_function

import os
import sys
import time
import numpy as np
import torch

_here = os.path.dirname(__file__)
_parent = os.path.join(_here, '..')
if _parent not in sys.path:
    sys.path.insert(0, _parent)

try:
    import torchtt
except ImportError:
    _torchtt_path = os.path.join(_here, '..', '..', 'torchTT')
    if _torchtt_path not in sys.path:
        sys.path.insert(0, _torchtt_path)
    import torchtt

import uq_adf_torchtt as uq

ts = lambda: time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())

def TTto_ndarray_components(tt):
    if hasattr(tt, 'cores'):
        return [c.detach().cpu().numpy() for c in tt.cores]
    components = []
    for ci in range(tt.degree()):
        c = tt.get_component(ci)
        components.append(c.to_ndarray())
    return components

def TTfrom_ndarray_components(components):
    cores = [torch.tensor(c) for c in components]
    return torchtt.TT(cores)


def save_tt_npz(fname, tt, **kwargs):
    components = TTto_ndarray_components(tt)
    keys = map("cmp_{}".format, range(len(components)))
    items = dict(zip(keys, components))
    items['format'] = 'TT'
    items['num_cmp'] = len(components)
    items.update(kwargs)
    np.savez_compressed(fname, **items)

def tt_moment_1(tt, basis=None):
    tt = TTto_ndarray_components(tt)
    phys = tt[0]
    phys.shape = phys.shape[1:]
    stoch = tt[1:]

    mom = np.ones((1,))
    for c in reversed(stoch):
        mom = np.tensordot(c[:,0], mom, (1,0))

    return phys.dot(mom)

def tt_moment_2(tt, basis):
    tt = TTto_ndarray_components(tt)
    phys = tt[0]
    phys.shape = phys.shape[1:]
    stoch = tt[1:]

    stochastic_dim = max(c.shape[1] for c in stoch)
    if basis == uq.PolynomBasis.Legendre:
        fs = 1/(2*np.arange(stochastic_dim)+1)
    elif basis == uq.PolynomBasis.Hermite:
        from scipy.special import factorial
        fs = factorial(np.arange(stochastic_dim), exact=True)
    else:
        print("ERROR: unknown basis '{}'".format(basis))
        exit(1)

    mom = np.ones((1,1))
    for c1,c2 in reversed(zip(stoch, stoch)):
        mom = np.tensordot(c1, mom, (2,0))
        mom = mom*fs[None,:mom.shape[1],None]
        mom = np.tensordot(mom, c2, ((1,2), (1,2)))

    return np.einsum('ij,ik,jk->i', phys, phys, mom)

def TTapply(tt, x, basis):
    tt = TTto_ndarray_components(tt)
    phys = tt[0]
    phys.shape = phys.shape[1:]
    stoch = tt[1:]
    assert len(x) == len(stoch)

    stochastic_dim = max(c.shape[1] for c in stoch)
    if basis == uq.PolynomBasis.Legendre:
        from numpy.polynomial.legendre import legval
        poly = lambda x: legval(x, np.eye(stochastic_dim)).T
    elif basis == uq.PolynomBasis.Hermite:
        from numpy.polynomial.hermite_e import hermeval
        poly = lambda x: hermeval(x, np.eye(stochastic_dim)).T
    else:
        print("ERROR: unknown basis '{}'".format(basis))
        exit(1)

    pos = poly(x)
    assert pos.shape == (len(x), stoch[0].shape[1])

    cum = np.ones((1,))
    for y,c in reversed(zip(pos,stoch)):
        cum = np.tensordot(c, cum, (2,0))
        cum = np.tensordot(cum, y, (1,0))
        assert cum.ndim == 1
    cum = np.tensordot(phys, cum, (1,0))
    return cum
