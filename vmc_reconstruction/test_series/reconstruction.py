# -*- coding: utf-8 -*-
from __future__ import division, print_function

import os, sys, glob, argparse, json, random

_here = os.path.dirname(__file__)
_parent = os.path.join(_here, '..')
if _parent not in sys.path:
    sys.path.insert(0, _parent)
from load_info import get_paths, load_info, load_problem, load_sampler

import uq_adf_torchtt as xe
import numpy as np
from utils import *


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('PATH', help='path to the directory containing the `info.json` for reconstruction')
    args = parser.parse_args()
    path = args.PATH
    if not os.path.isdir(path):
        raise IOError("'{}' is not a valid directory".format(path))

    reco_info = load_info(path)

    stochastic_dim = reco_info["basis dimension"]
    target_eps     = reco_info['tolerance']
    max_itr        = reco_info['maximum iterations']
    ortho          = reco_info['orthogonalization']
    maxSamples     = reco_info["maximum sample size"]
    sample_path    = reco_info["sample path"]
    reference_path = reco_info["reference path"]

    def get_fem_dofs(info):
        problem = load_problem(info)
        fem_space = problem.setup_space(info)['V']
        return len(fem_space.dofmap().dofs())

    def load_samples(path, max_samples):
        f_name = os.path.join(path, 'samples', '{}.npz')
        print(ts(), "Loading samples: '%s'"%f_name.format('*'))
        info = load_info(path)
        spatial_dim = get_fem_dofs(info)
        M = info["expansion"]["size"]
        if info['sampling']['distribution'] == 'uniform':
            basis = xe.PolynomBasis.Legendre
        elif info['sampling']['distribution'] == 'normal':
            basis = xe.PolynomBasis.Hermite
        else:
            print("ERROR: unknown distribution '{}'".format(info['sampling']['distribution']))
            exit(1)
        batch_size = info["sampling"]["batch size"]

        num_files = len(glob.glob(f_name.format('*')))
        print(ts(), "There are", num_files, "datafiles of", batch_size, "samples each.")
        num_files   = min(num_files, int(np.ceil(max_samples/batch_size)))
        num_samples = min(num_files*batch_size, max_samples)

        params = np.empty((num_samples, M))
        test_solutions = np.empty((num_samples, spatial_dim))
        if ortho: training_solutions = np.empty((num_samples, spatial_dim))
        else: training_solutions = test_solutions
        a = 0
        for i in range(num_files):
            o = min(a+batch_size, max_samples)
            z = np.load(f_name.format(i))
            params[a:o] = z['ys']
            test_solutions[a:o] = z['us']
            if ortho: training_solutions[a:o] = z['nus']
            a = o
        return spatial_dim, M, basis, batch_size, params, test_solutions, training_solutions

    spatial_dim, M, basis, batch_size, params, test_solutions, training_solutions = load_samples(sample_path, maxSamples)
    dimensions = [spatial_dim] + [stochastic_dim]*M

    print(ts(), "Spacial Dimension:", spatial_dim, "| Expansion size:", M,
            "| Stochastic Dimension:", stochastic_dim-1, "| Stochastic Basis Type:", basis)

    def load_reference(path):
        f_name = os.path.join(path, 'moments.npz')
        print(ts(), "Load reference: '%s'"%f_name)
        moments = np.load(f_name)['arr_0']
        return np.split(moments[-1], 2)

    qmcM1, qmcM2 = load_reference(reference_path)
    qmcM1Norm = np.linalg.norm(qmcM1)
    qmcM2Norm = np.linalg.norm(qmcM2)

    def sample_set(set_size, num_samples, exclude=set()):
        ret = set()
        while len(ret) < num_samples:
            candidate = np.random.randint(0, set_size-1)
            if candidate not in (exclude | ret):
                ret.add(candidate)
        return ret

    if ortho:
        import warnings
        warnings.filterwarnings("ignore", message="numpy.dtype size changed")
        warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
        import scipy.sparse as sps
        chol_path = os.path.join(sample_path, 'cholesky.npz')
        chol_perm_path = os.path.join(sample_path, 'cholesky_permutation.npz')

        L = sps.load_npz(chol_path)
        P = sps.load_npz(chol_perm_path)

        from scipy.sparse.linalg import spsolve_triangular
        def transform_tensor(reconstruction):
            components = TTto_ndarray_components(reconstruction)

            print(ts(), "Transform reconstruction.")
            c0 = components[0]
            c0s = c0.shape
            c0.shape = c0s[1:]
            c0 = spsolve_triangular(P.dot(L).tocsr(), P.dot(c0))
            c0 = np.array(c0).reshape(c0s)
            components[0] = c0

            return TTfrom_ndarray_components(components)
    else:
        transform_tensor = lambda x: x

    def reconstruct(sampleSet):
        print(ts(), "Running reconstruction using", len(sampleSet), "samples.")

        # sampleSet = sample_set(len(training_solutions), N)

        print(ts(), "Create measurment set.")
        measurments = xe.UQMeasurementSet()
        for i in sampleSet:
            y = params[i]
            measurments.add(y, training_solutions[i])

        print(ts(), "Run reconstruction.")
        reconstruction = xe.uq_ra_adf(measurments, basis, dimensions, target_eps, max_itr)

        return transform_tensor(reconstruction)
        # return transform_tensor(reconstruction), sampleSet

    completeSet = set(range(len(training_solutions)))
    result_path = os.path.join(path, 'results', '{}')
    starts = [400, 200, 100, 300]
    for start in starts:
        for N in range(start, 10001, 400):
            sampleSet = set(random.sample(completeSet, N))
            reconstruction = reconstruct(sampleSet)
            # reconstruction, sampleSet = reconstruct(N)

            print(ts(), "Save results.");
            result_path_N = result_path.format(N)
            try: os.makedirs(result_path_N)
            except OSError as e:
                if e.args[1] != 'File exists':
                    raise e
            f_name = os.path.join(result_path_N, 'reconstruction.npz')
            save_tt_npz(f_name, reconstruction)

            print(ts(), "Calculate moments from samples.")
            sampM1 = test_solutions[list(sampleSet)].mean(axis=0)
            sampM2 = (test_solutions[list(sampleSet)]**2).mean(axis=0)

            print(ts(), "Calculate moments from reconstruction.")
            m1 = tt_moment_1(reconstruction, basis)
            m2 = tt_moment_2(reconstruction, basis)

            testError = 0.0
            if N+1000 <= len(test_solutions):
                testSet = set(random.sample(completeSet-sampleSet, 1000))
                # testSet = sample_set(len(test_solutions), 1000, sampleSet)
                for s in testSet:
                    samp = TTapply(reconstruction, params[s], basis)
                    testError += np.linalg.norm(samp - test_solutions[s])/np.linalg.norm(test_solutions[s])
                testError /= len(testSet)

            results = {
                'N': N,
                'mc error': {
                    'moment 1': np.linalg.norm(sampM1-qmcM1)/qmcM1Norm,
                    'moment 2': np.linalg.norm(sampM2-qmcM2)/qmcM2Norm
                },
                'reco error': {
                    'moment 1': np.linalg.norm(m1-qmcM1)/qmcM1Norm,
                    'moment 2': np.linalg.norm(m2-qmcM2)/qmcM2Norm
                },
                'testError': testError
            }

            print(ts(), "MC Error:   {0:1.2e} | {1:1.2e}".format(results['mc error']['moment 1'], results['mc error']['moment 2']))
            print(ts(), "Reco Error: {0:1.2e} | {1:1.2e}".format(results['reco error']['moment 1'], results['reco error']['moment 2']))
            print(ts(), "Test Error: {0:1.2e}".format(testError))

            f_name = os.path.join(result_path_N, 'reconstruction.json')
            with open(f_name, 'w') as f:
                json.dump(results, f)
