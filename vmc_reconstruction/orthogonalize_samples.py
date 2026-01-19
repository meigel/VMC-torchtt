from __future__ import division, print_function
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
import scipy.sparse as sps
import os, sys, glob
from load_info import get_paths, load_info

if __name__=='__main__':
    descr = """Orthogonalize the sampled solutions for the given problem.
The orthogonalized samples will be stored in the same .npz files as the original samples.
"""
    path, = get_paths(descr, ('PATH', 'path to the directory containing the `info.json` for the specific problem'))
    info = load_info(path)

    print("[ 0%] Loading Factored Stiffness Matrix", end='')
    chol_path = os.path.join(path, 'cholesky.npz')
    L = sps.load_npz(chol_path)
    print("\r[done] Loading Factored Stiffness Matrix")

    spl_path = os.path.join(path, 'samples', '{}.npz')
    nr_inputs = len(glob.glob(spl_path.format('*')))

    print('[ 0%] Orthogonalizing Samples (file {}/{})'.format(0, nr_inputs), end='')
    for in_idx in range(nr_inputs):
        print('\r[{0:2d}%] Orthogonalizing Samples (file {1}/{2})'.format(int(100*(in_idx+1)/nr_inputs), in_idx+1, nr_inputs), end='')
        sys.stdout.flush()

        splz = np.load(spl_path.format(in_idx))
        if 'nus' in splz: continue
        ys = splz['ys']
        us = splz['us']
        nus = L.dot(us.T).T
        np.savez_compressed(spl_path.format(in_idx), us=us, ys=ys, nus=nus)
    print('\r[done] Orthogonalizing Samples')
