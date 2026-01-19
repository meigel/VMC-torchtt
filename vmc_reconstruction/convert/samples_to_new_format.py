from __future__ import division, print_function
import numpy as np
import os, sys, glob
from load_info import get_paths

if __name__=='__main__':
    path, = get_paths("", ('PATH', 'path to the directory containing the `info.json` for the specific problem'))

    in_base = os.path.join(path, 'samples_old_format', '{}_{}.npz')
    out_base = os.path.join(path, 'samples', '{}.npz')
    nr_inputs = len(glob.glob(in_base.format('us', '*')))
    assert nr_inputs == len(glob.glob(in_base.format('ys', '*')))
    ortho = len(glob.glob(in_base.format('nus', '*'))) > 0

    os.mkdir(os.path.join(path, 'samples'))

    print('[ 0%] Converting Samples (file {}/{})'.format(0, nr_inputs), end='')
    for in_idx in range(nr_inputs):
        print('\r[{0:2d}%] Converting Samples (file {1}/{2})'.format(int(100*(in_idx+1)/nr_inputs), in_idx+1, nr_inputs), end='')
        sys.stdout.flush()

        ys = np.load(in_base.format('ys', in_idx))['arr_0']
        us = np.load(in_base.format('us', in_idx))['arr_0']
        if ortho:
            nus = np.load(in_base.format('nus', in_idx))['arr_0']
            splz = np.savez_compressed(out_base.format(in_idx), us=us, ys=ys, nus=nus)
        splz = np.savez_compressed(out_base.format(in_idx), us=us, ys=ys)
    print('\r[done] Converting Samples')
