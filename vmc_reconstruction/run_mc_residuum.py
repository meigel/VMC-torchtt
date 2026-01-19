from __future__ import division, print_function

import os
import sys
import time
import numpy as np
import torch

try:
    import torchtt
except ImportError:
    _here = os.path.dirname(__file__)
    _torchtt_path = os.path.join(_here, '..', 'torchTT')
    if _torchtt_path not in sys.path:
        sys.path.insert(0, _torchtt_path)
    import torchtt

import uq_adf_torchtt as uq
from samplers import Uniform, CartesianProductSampler

mesh_size = 25
n_samples = 100
Dims = [6]

print_old = print

ts = lambda: time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
print = lambda *args, **kwargs: print_old(ts(), *args, **kwargs)


def list2str(ls):
    return '-'.join(map(str, ls))


transpose = lambda ls: list(zip(*ls))


def save_tt_npz(fname, tt, **kwargs):
    cores = [c.detach().cpu().numpy() for c in tt.cores]
    keys = map("cmp_{}".format, range(len(cores)))
    items = dict(zip(keys, cores))
    items['format'] = 'TT'
    items['num_cmp'] = len(cores)
    items.update(kwargs)
    np.savez_compressed(fname, **items)


def load_tt_npz(fname):
    z = np.load(fname)
    items = dict(z.items())
    assert items.get('format') == 'TT'
    items.pop('format', None)
    num_cmp = int(items.pop('num_cmp'))
    components = [items.pop('cmp_%d' % i) for i in range(num_cmp)]
    cores = [torch.tensor(c) for c in components]
    return torchtt.TT(cores), items


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


def tt_apply_tt(tt, x, basis):
    cores = [c.detach().cpu().numpy() for c in tt.cores]
    return tt_apply_np(cores, x, basis)


def tt_frob_norm(tt):
    return torch.sqrt(torchtt.dot(tt, tt)).item()


def prolong_tt(tt, dims):
    cores = []
    for idx, core in enumerate(tt.cores):
        r1, n, r2 = core.shape
        new_n = dims[idx]
        if new_n < n:
            raise ValueError('Cannot shrink TT core size.')
        if new_n == n:
            cores.append(core)
        else:
            new_core = torch.zeros((r1, new_n, r2), dtype=core.dtype, device=core.device)
            new_core[:, :n, :] = core
            cores.append(new_core)

    for idx in range(len(tt.cores), len(dims)):
        new_core = torch.zeros((1, dims[idx], 1), dtype=tt.cores[0].dtype, device=tt.cores[0].device)
        new_core[0, 0, 0] = 1.0
        cores.append(new_core)

    return torchtt.TT(cores)


def fix_mode(tt, mode, index):
    idx = [slice(None)] * len(tt.N)
    idx[mode] = index
    return tt[tuple(idx)]


def inner(t1, t2):
    return torchtt.dot(t1, t2).item()


def save_sampled_values(dims):
    print("Calling save_sampled_values")
    print("Loading problem ...")
    from problem.darcy import Problem

    print('Creating samples ...')
    sampler = CartesianProductSampler([Uniform((-1, 1))] * len(dims))
    sample = sampler.sample
    weight = lambda nodes: np.full(nodes.shape[1], 1 / nodes.shape[1])

    print('Sampling ...')
    ys = sample(n_samples)

    info = {
        "problem": {"name": "darcy"},
        "fe": {
            "mesh": mesh_size,
            "degree": 1
        },
        "expansion": {
            "mean": 1.0,
            "scale": 12.5,
            "size": len(dims),
            "decay rate": 2.0
        },
        "sampling": {
            "distribution": "uniform"
        }
    }
    problem = Problem(info)

    print('Computing values ...')
    results = []
    for y in ys:
        results.append(problem.solution(y))
    us = np.array(results)

    assert ys.shape == (n_samples, len(dims))
    assert us.shape[0] == n_samples
    x_dim = us.shape[1]

    print('Saving values ...')
    nodes = ys.T
    wgts = weight(nodes)
    vals = us
    np.savez_compressed('run_mc_residuum_test/samples_{}.npz'.format(list2str(dims)), nodes=nodes, weights=wgts, values=vals)


save_sampled_values(Dims)


def save_reconstruction(dims):
    print("Calling save_reconstruction")
    print('Loading values ...')
    z = np.load('run_mc_residuum_test/samples_{}.npz'.format(list2str(dims)))
    nodes = z['nodes']
    vals = z['values']
    x_dim = vals.shape[1]

    ys = nodes.T
    meas = uq.UQMeasurementSet()
    for y, u in zip(ys, vals):
        meas.add(y, u)

    print('Reconstruct ...')
    reco = uq.uq_ra_adf(meas, uq.PolynomBasis.Legendre, [x_dim] + list(dims), targeteps=1e-8, maxitr=300)

    print('Saving values ...')
    save_tt_npz('run_mc_residuum_test/reconstruction_{}.npz'.format(list2str(dims)), reco)


save_reconstruction(Dims)


def save_reconstructed_values(dims):
    print("Calling save_reconstructed_values")
    print('Loading values ...')
    reco, _ = load_tt_npz('run_mc_residuum_test/reconstruction_{}.npz'.format(list2str(dims)))

    print('Creating samples ...')
    sampler = CartesianProductSampler([Uniform((-1, 1))] * len(dims))
    sample = sampler.sample
    weight = lambda nodes: np.full(nodes.shape[1], 1 / nodes.shape[1])

    print('Sampling ...')
    ys = sample(n_samples)

    print('Computing values ...')
    us = np.array([tt_apply_tt(reco, y, uq.PolynomBasis.Legendre) for y in ys])
    assert us.dtype == np.float64, us.dtype

    assert ys.shape == (n_samples, len(dims))
    assert us.shape == (n_samples, reco.N[0])

    print('Saving values ...')
    nodes = ys.T
    wgts = weight(nodes)
    vals = us
    np.savez_compressed('run_mc_residuum_test/samples_reconstructed_{}.npz'.format(list2str(dims)), nodes=nodes, weights=wgts, values=vals)


save_reconstructed_values(Dims)


def save_residuum_values(sol_dims, res_dims):
    print("Calling save_residuum_values")
    assert len(sol_dims) == len(res_dims) or len(sol_dims) == len(res_dims) - 1
    print("Loading problem ...")
    from problem.darcy import Problem

    print('Loading values ...')
    z = np.load('run_mc_residuum_test/samples_reconstructed_{}.npz'.format(list2str(sol_dims)))
    nodes = z['nodes']
    wgts = z['weights']
    vals = z['values']
    x_dim = vals.shape[1]

    info = {
        "problem": {"name": "darcy"},
        "fe": {
            "mesh": mesh_size,
            "degree": 1
        },
        "expansion": {
            "mean": 1.0,
            "scale": 12.5,
            "size": len(res_dims),
            "decay rate": 2.0
        },
        "sampling": {
            "distribution": "uniform"
        }
    }
    problem = Problem(info)

    if len(res_dims) == len(sol_dims) + 1:
        nodes = np.concatenate([nodes, 2 * np.random.rand(1, nodes.shape[1]) - 1])
    ys = nodes.T
    assert ys.shape == (n_samples, len(res_dims))
    us = vals

    print('Computing values ...')
    results = []
    for y, u in zip(ys, us):
        results.append(problem.residuum((y, u)))
    res = np.array(results)

    assert res.ndim == 2
    assert res.shape[0] == n_samples

    print('Saving values ...')
    vals = res
    np.savez_compressed('run_mc_residuum_test/samples_residuum_{}.npz'.format(list2str(res_dims)), nodes=nodes, weights=wgts, values=vals)


inc_dims = [d + 1 for d in Dims] + [2]
save_residuum_values(Dims, Dims)
save_residuum_values(Dims, inc_dims)


def save_reconstruction_residuum(dims):
    print('Calling save_reconstruction_residuum')
    print('Loading values ...')
    z = np.load('run_mc_residuum_test/samples_residuum_{}.npz'.format(list2str(dims)))
    nodes = z['nodes']
    vals = z['values']
    x_dim = vals.shape[1]

    ys = nodes.T
    meas = uq.UQMeasurementSet()
    for y, u in zip(ys, vals):
        meas.add(y, u)

    print('Reconstruct ...')
    reco = uq.uq_ra_adf(meas, uq.PolynomBasis.Legendre, [x_dim] + list(dims), targeteps=1e-8, maxitr=300)

    print('Saving values ...')
    save_tt_npz('run_mc_residuum_test/reconstruction_residuum_{}.npz'.format(list2str(dims)), reco)


save_reconstruction_residuum(Dims)
save_reconstruction_residuum(inc_dims)


print('Loading values ...')
R_lam, _ = load_tt_npz('run_mc_residuum_test/reconstruction_residuum_{}.npz'.format(list2str(Dims)))
R_tet, _ = load_tt_npz('run_mc_residuum_test/reconstruction_residuum_{}.npz'.format(list2str(inc_dims)))

assert R_lam is not None and R_tet is not None

print(tt_frob_norm(R_lam))
R_lam = prolong_tt(R_lam, [R_lam.N[0]] + inc_dims)
print(tt_frob_norm(R_lam))
print(tt_frob_norm(R_tet))
R = R_tet - R_lam
print(tt_frob_norm(R))

rs = []
for m in range(len(inc_dims)):
    R_m = fix_mode(R, m + 1, inc_dims[m] - 1)
    r = inner(R_m, R_m)
    rs.append(r)
print("Error estimators:", rs)
