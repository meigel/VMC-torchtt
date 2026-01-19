# coding: utf-8
from __future__ import division, print_function
import os,sys,glob
from pprint import pprint
from tabulate import tabulate
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from load_info import get_paths, load_info, load_problem
from print_ranks import bold, indent, IndentBlock, frame, prints_ranks

if __name__=='__main__':
    descr = """ ... """
    param_path, = get_paths(descr, ('PARAM_PATH', 'path to the directory containing the `info.json`'))
    param_info = load_info(param_path)
    path = param_info['reconstruction info']
    mc_path = param_info['mc info']
    ref_path = param_info['reference info']

    print("Loading reconstruction from: '{}'".format(path))
    print("Loading Monte Carlo samples from: '{}'".format(mc_path))
    print("Loading reference solution from: '{}'".format(ref_path))

    info = load_info(path)

    mc_info = load_info(mc_path)
    problem = load_problem(mc_info)
    FEM = problem.setup_FEM(mc_info)

    ref_info = load_info(ref_path)
    assert ref_info['sampling']['strategy'] == 'sobol'
    del mc_info['sampling']['strategy']
    del ref_info['sampling']['strategy']
    del mc_info['sampling']['dump limit']
    del ref_info['sampling']['dump limit']
    if mc_info != ref_info:
        print("ERROR: parameters do not match")
        print("MC_PATH:")
        pprint(mc_info)
        print("REF_PATH:")
        pprint(ref_info)
        exit(1)

    fn = os.path.join(path, 'reconstruction.npz')
    print("Loading '%s'"%fn)
    Uz = np.load(fn)

    num_components = Uz['num_cmp'] if 'num_cmp' in Uz else len(Uz.keys())
    components = [Uz['cmp_%d'%i] for i in range(num_components)]
    shapes = [c.shape for c in components]
    dims = [int(s[1]) for s in shapes]
    ranks = [s[0] for s in shapes[1:]]

    phys = components[0].view()
    phys.shape = phys.shape[1:]
    stoch = components[1:]


    nnzs =[round(np.sum(cmpt!=0)/cmpt.size, 2) for cmpt in components]
    sparses = [(idx, nnz) for idx,nnz in enumerate(nnzs) if nnz < 1]
    ppair = lambda xy: "(%s: %s)"%xy

    print(bold("Tensor info:"))

    with IndentBlock():
        print("Dimensions: %s"%dims)
        print(bold("Sparsity: ")
            + ' '.join(ppair(pair) for pair in sparses)
            + bold(" (all other components are dense)"))
        print(bold("Ranks: ") + "(maxRank: %d)"%max(ranks))
        print(frame(prints_ranks(ranks)), 1)

    rks = [1] + ranks + [1]

    def moment_1(tt):
        cum = np.ones((1,))
        for c in reversed(tt):
            cum = np.tensordot(c[:,0], cum, (1,0))
        return cum

    # polynomial normalization factors for xerus (see alea-testing/src/alea/linalg/tensor/extended_tt.py l.656)
    max_degree_p = max(dims[1:]) # predecessor of the maximal degree

    if ref_info['sampling']['distribution'] == 'normal':
        print(indent(bold("Basis: Hermite")))

        from numpy.polynomial.hermite_e import hermeval
        poly = lambda x: hermeval(x, np.eye(max_degree_p)).T

        # Xerus uses a renormalized version of boost::hermite to obtain the polynomials He_n(x)
        # The scalar product wrt the standard normal distribution is orthogonal with factors
        from scipy.special import factorial
        fs = factorial(np.arange(max_degree_p), exact=True)
        # The factor 2*pi is not needed since its inverse is part of the probability density.

    elif ref_info['sampling']['distribution'] == 'uniform':
        print(indent(bold("Basis: Legendre")))

        from numpy.polynomial.legendre import legval
        poly = lambda x: legval(x, np.eye(max_degree_p)).T

        # Xerus uses boost::legendre_p to obtain the polynomials P_n(x)
        # The scalar product wrt the uniform distribution on [-1,1] is orthogonal with factors
        fs = 1/(2*np.arange(max_degree_p)+1)
        # The factor 2 is not needed since its inverse is part of the probability density.

    else:
        print("ERROR: unknown distribution '{}'".format(info['sampling']['distribution']))
        exit(1)

    def moment_2(tt):
        cum = np.ones((1,1))
        for c1,c2 in reversed(zip(tt, tt)):
            cum = np.tensordot(c1, cum, (2,0))
            cum = cum*fs[None,:cum.shape[1],None]
            cum = np.tensordot(cum, c2, ((1,2), (1,2)))
        return cum

    # Since the core is at position 0: If np.all(fs==1)
    # def moment_2(tt):
    #     return np.eye(tt[0].shape[0])

    def apply(pos, phys, stoch):
        cum = np.ones((1,))
        assert len(pos) == len(stoch)
        for y,c in reversed(zip(pos,stoch)):
            assert c.ndim == 3
            assert cum.ndim == 1
            cum = np.tensordot(c, cum, (2,0))
            assert cum.ndim == 2
            assert y.ndim == 1
            cum = np.tensordot(cum, y, (1,0))
        assert phys.ndim == 2
        assert cum.ndim == 1
        cum = np.tensordot(phys, cum, (1,0))
        return cum

    # # Use samples to approximate the second moment
    # def expvar(phys, stoch):
    #     assert not problem.FEM['normal_rvs']
    #     m1 = 0
    #     m2 = 0
    #     for y in 2*np.random.rand(10000, len(stoch)) - 1:
    #         pos = poly(y)
    #         x = apply(pos, phys, stoch)
    #         m1 = m1 + x
    #         m2 = m2 + x**2
    #     return m1/10000, m2/10000

    M1 = moment_1(stoch)
    e = phys.dot(M1)
    M2 = moment_2(stoch)
    var = np.einsum('ij,ik,jk->i', phys, phys, M2) - e**2

    # e, var = expvar(phys, stoch)

    def weighted_norm(M):
        return lambda x: np.sqrt(x.dot(M.dot(x)))

    vectorize = lambda f: lambda xs: np.array(map(f, xs))

    def get_stiffness_matrix(path):
        stiff_path = os.path.join(path, 'stiffness.npz')
        S = sps.load_npz(stiff_path)
        return S

    def load_values(path):
        fn = os.path.join(path, 'moments.npz')
        es = np.load(fn)['arr_0']
        m1, m2 = np.split(es, 2, axis=1)
        e = m1
        v = m2 - m1**2
        n = mc_info['sampling']['batch size']*np.arange(1,len(es)+1)
        return n, e, v

    n_mc, es_mc, vs_mc = load_values(mc_path)
    _, es_ref, vs_ref = load_values(ref_path)
    es_ref = es_ref[-1]
    vs_ref = vs_ref[-1]

    # norm = lambda xs: np.linalg.norm(xs, axis=1)
    norm = vectorize(weighted_norm(get_stiffness_matrix(mc_path)))
    e_diff = norm(e - es_mc) / norm(es_mc)
    v_diff = norm(var - vs_mc) / norm(vs_mc)

    # norm = np.linalg.norm
    norm = weighted_norm(get_stiffness_matrix(mc_path))
    idx = int(info['sample size']/mc_info['sampling']['batch size']) - 1
    e_diff_mc = norm(es_mc[idx]-es_ref) / norm(es_ref)
    v_diff_mc = norm(vs_mc[idx]-vs_ref) / norm(vs_ref)
    e_diff_ref = norm(e - es_ref) / norm(es_ref)
    v_diff_ref = norm(var - vs_ref) / norm(vs_ref)

    print(bold("Reconstruction info:"))
    ndofs = sum(rks[i] * dims[i] * rks[i+1] for i in range(len(dims))) - sum(rk**2 for rk in ranks)
    print(indent(
          tabulate([[ndofs, info['sample size'], ndofs/info['sample size']]],
                    ["#DoFs", "#Samples", "Ratio"],
                    tablefmt="fancy_grid",
                    floatfmt=".2f")
          ))

    print(bold("Relative Error of first and second Moment:"))
    print(indent(
          tabulate([["M1", e_diff_ref, e_diff_mc],
                    ["M2", v_diff_ref, v_diff_mc]],
                    ["Reconstruction", "Monte Carlo"],
                    tablefmt="fancy_grid",
                    floatfmt=".2e")
          ))


    def load_samples(num_samples, start=0):
        assert num_samples > 0
        input_dir = os.path.join(mc_path, 'samples')
        print("Loading samples from '%s'"%input_dir)
        input_base = os.path.join(input_dir, '{}.npz')

        num_files = len(glob.glob(input_base.format('*')))
        skip_files = int(start/mc_info['sampling']['batch size'])
        skip_samples = start - mc_info['sampling']['batch size']*skip_files

        for in_idx in range(skip_files, num_files):
            ez = np.load(input_base.format(in_idx))
            ys = ez['ys']
            us = ez['us']
            assert len(ys) == len(us)
            if in_idx == skip_files:
                ys = ys[skip_samples:]
                us = us[skip_samples:]
            for y,u in zip(ys, us):
                yield y, u
                num_samples -= 1
                if num_samples == 0: return

        raise IOError("Not enough files to obtain required amount of samples.")
        # print("WARNING: Not enough files to obtain required amount of samples.")


    error = np.float64(0)
    ns = 2000 # take only `ns` many samples
    nse = 0
    try:
        for y, u in load_samples(ns, start=info['sample size']):
            pos = poly(y)
            assert pos.shape == (len(y), max_degree_p)
            est = apply(pos, phys, stoch)
            error += norm(est - u) / norm(u)
            nse += 1
        error /= np.float64(nse)
    except IOError:
        try:
            for y, u in load_samples(ns):
                pos = poly(y)
                assert pos.shape == (len(y), max_degree_p)
                est = apply(pos, phys, stoch)
                error += norm(est - u) / norm(u)
                nse += 1
            error /= np.float64(nse)
            print("WARNING: Not enough files to obtain required amount of samples. Error was computet on training data.")
        except IOError:
            print("ERROR: Not enough files to obtain required amount of samples.")
            error = np.nan
    print(bold("Average relative pointwise Error on {0:d} Samples:".format(nse)))
    print(indent(
          tabulate([["Error", error]],
                    tablefmt="fancy_grid",
                    floatfmt=".2e")
          ))
