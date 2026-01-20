# -*- coding: utf-8 -*-
from __future__ import division, print_function

import os
import sys
import numpy as np
import torch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import torchtt
except ImportError:
    _torchtt_path = os.path.join(_repo_root, 'torchTT')
    if _torchtt_path not in sys.path:
        sys.path.insert(0, _torchtt_path)
    import torchtt

from vmc_reconstruction import uq_adf_torchtt as uq


def _poly_matrix(x, degree, basis):
    if basis == uq.PolynomBasis.Legendre:
        from numpy.polynomial.legendre import legval
        return legval(x, np.eye(degree)).T
    if basis == uq.PolynomBasis.Hermite:
        from numpy.polynomial.hermite_e import hermeval
        return hermeval(x, np.eye(degree)).T
    raise ValueError("Unknown basis '{}'".format(basis))


def tt_apply_np(cores, x, basis):
    phys = cores[0]
    phys = phys.reshape(phys.shape[1], phys.shape[2])
    stoch = cores[1:]

    pos = _poly_matrix(np.asarray(x), stoch[0].shape[1], basis)
    cum = np.ones((1,))
    for y, c in reversed(list(zip(pos, stoch))):
        cum = np.tensordot(c, cum, (2, 0))
        cum = np.tensordot(cum, y, (1, 0))
    return np.tensordot(phys, cum, (1, 0))


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    n_samples = 200
    n_test = 50
    x_dim = 6
    order = 4
    n_vars = 2
    dims = [x_dim] + [order] * n_vars
    basis = uq.PolynomBasis.Legendre

    device = torch.device(os.environ.get('VMC_DEVICE', 'cpu'))

    true_tt = torchtt.random(dims, 2, dtype=torch.float64, device=device)
    true_cores = [c.detach().cpu().numpy() for c in true_tt.cores]

    ys = np.random.uniform(-1.0, 1.0, size=(n_samples, n_vars))
    us = np.array([tt_apply_np(true_cores, y, basis) for y in ys])

    meas = uq.UQMeasurementSet()
    for y, u in zip(ys, us):
        meas.add(y, u)

    reco = uq.uq_ra_adf(meas, basis, dims, targeteps=1e-6, maxitr=200, device=device, orthonormal=False)
    reco_cores = [c.detach().cpu().numpy() for c in reco.cores]

    test_ys = np.random.uniform(-1.0, 1.0, size=(n_test, n_vars))
    ref = np.array([tt_apply_np(true_cores, y, basis) for y in test_ys])
    pred = np.array([tt_apply_np(reco_cores, y, basis) for y in test_ys])

    rel_err = np.linalg.norm(pred - ref) / np.linalg.norm(ref)
    print("UQ-ADF smoke test relative error:", rel_err)


if __name__ == '__main__':
    main()
