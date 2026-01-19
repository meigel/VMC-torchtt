from __future__ import division, print_function
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import sys, os
from pprint import pprint
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from load_info import get_paths, load_info, load_problem

def get_stiffness_matrix(path):
    stiff_path = os.path.join(path, 'stiffness.npz')
    S = sps.load_npz(stiff_path)
    return S

if __name__=='__main__':
    descr = """ ... """
    path, ref_path  = get_paths(descr,
            ('PATH', 'path to the directory containing the `info.json` for given samples'),
            ('REF_PATH', 'path to the directory containing the `info.json` for the corresponding reference samples'))
    info = load_info(path)

    ref_info = load_info(ref_path)
    if not ref_info['sampling']['strategy'] == 'sobol':
        print("WARNING: sampling strategy for reference is '%s'"%(ref_info['sampling']['strategy']))
    del info['sampling']['strategy']
    del ref_info['sampling']['strategy']
    del info['sampling']['dump limit']
    del ref_info['sampling']['dump limit']
    if info != ref_info:
        print("ERROR: parameters do not match")
        print("PATH:")
        pprint(info)
        print("REF_PATH:")
        pprint(ref_info)
        exit(1)

    # optparser.add_option('--ref', '--reference',
    #                      action='store_true', default=False, dest='ref',
    #                      help='use QMC reference solution')

    def weighted_norm(M):
        return lambda x: np.sqrt(x.dot(M.dot(x)))

    vectorize = lambda f: lambda xs: np.array(map(f, xs))

    # norm = lambda xs: np.linalg.norm(xs, axis=1)
    norm = vectorize(weighted_norm(get_stiffness_matrix(path)))

    def load_values(path):
        fn = os.path.join(path, 'moments.npz')
        es = np.load(fn)['arr_0']
        m1, m2 = np.split(es, 2, axis=1)
        e = m1
        v = m2 - m1**2
        n = info['sampling']['batch size']*np.arange(1,len(es)+1)
        return n, e, v

    n,e,v = load_values(path)
    _,e_ref,v_ref = load_values(ref_path)

    f, axs = plt.subplots(1,2)

    axs[0].plot(n, norm(e - e_ref[-1]))
    axs[0].set_xscale('linear')
    axs[0].set_yscale('log')
    axs[0].set_title("Expectation")

    axs[1].plot(n, norm(v - v_ref[-1]))
    axs[1].set_xscale('linear')
    axs[1].set_yscale('log')
    axs[1].set_title("Variance")

    plt.tight_layout()
    plt.show()
