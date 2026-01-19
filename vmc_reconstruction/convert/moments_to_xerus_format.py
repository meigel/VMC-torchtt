from __future__ import division, print_function
import numpy as np
import argparse, os, sys, glob
from load_info import load_info

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('PATH', help='path to the directory containing the `info.json` for given samples')
    parser.add_argument('STORE_PATH', help='path to the directory to which the samples will be saved')
    parser.add_argument('--ortho', dest='ORTHO', action='store_true', help='use orthogonal samples')
    args = parser.parse_args()

    in_file = os.path.join(args.PATH, 'moments.npz')

    out_dir = args.STORE_PATH
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file_base_1 = os.path.join(out_dir, '{}m1.dat')
    out_file_base_2 = os.path.join(out_dir, '{}m2.dat')

    print("Reading '{}'".format(in_file))
    us = np.load(in_file)['arr_0']
    es, vs = np.split(us, 2, axis=1)

    batch_size = load_info(args.PATH)['sampling']['batch size']

    out_file_1 = lambda i: out_file_base_1.format(batch_size*i)
    out_file_2 = lambda i: out_file_base_2.format(batch_size*i)
    to_str = lambda x: ' '.join(x.astype(str)) + '\n'

    if args.ORTHO:
        import scipy.sparse as sps

        print("[ 0%] Loading Factored Stiffness Matrix", end='')
        L = sps.load_npz(os.path.join(args.PATH, 'cholesky.npz'))
        print("\r[done] Loading Factored Stiffness Matrix")

        print('[ 0%] Orthogonalizing Estimates', end='')
        es = L.dot(es.T).T
        vs = L.dot(vs.T).T
        print('\r[done] Orthogonalizing Estimates')

    for idx, (e,v) in enumerate(zip(es,vs), start=1):

        print("\rWriting '{}' ({}/{})".format(out_file_1(idx), 2*idx-1, 2*len(es)), end='')
        sys.stdout.flush()
        with open(out_file_1(idx), 'w') as f:
            f.write(to_str(e))

        print("\rWriting '{}' ({}/{})".format(out_file_2(idx), 2*idx, 2*len(es)), end='')
        sys.stdout.flush()
        with open(out_file_2(idx), 'w') as f:
            f.write(to_str(v))

    print()
