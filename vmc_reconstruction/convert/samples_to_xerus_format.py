from __future__ import division, print_function
import numpy as np
import argparse, os, sys, glob

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('PATH', help='path to the directory containing the `info.json` for given samples')
    parser.add_argument('STORE_PATH', help='path to the directory to which the samples will be saved')
    parser.add_argument('--ortho', dest='ORTHO', action='store_true', help='use orthogonal samples')
    args = parser.parse_args()

    path = args.PATH
    if not os.path.isdir(path):
        raise IOError("'{}' is not a valid directory".format(path))

    us_key = 'n'*args.ORTHO + 'us'

    in_base = os.path.join(path, 'samples/{}.npz')
    nr_inputs = len(glob.glob(in_base.format('*')))
    print("Reading Samples from '{}' ({} files)".format(in_base.format('*'), nr_inputs))

    output_dir = args.STORE_PATH
    out_name  = os.path.basename(output_dir)
    out_base = os.path.join(output_dir, out_name+'-')
    out_offset = len(glob.glob(out_base+'*.dat'))

    print("Writing Samples to '{}*.dat'".format(out_base))
    if out_offset:
        print("WARNING: {} output files already exist and will not be overwritten".format(out_offset))

    to_str = lambda x: ' '.join(x.astype(str)) + '\n'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    out_idx = out_offset
    for in_idx in range(nr_inputs):
        z = np.load(in_base.format(in_idx))
        us = z[us_key]
        ys = z['ys']
        lus = len(us)
        assert lus == len(ys)

        for i,u,y in zip(range(1,lus+1), us, ys):
            out_idx += 1
            print('\rReading {}/{} -- Writing {}/{} ({} overall)'.format(in_idx+1, nr_inputs, i, lus, out_idx), end='')
            with open("{}{}.dat".format(out_base, out_idx), 'w') as f:
                f.write(to_str(y))
                f.write(to_str(u))
    print()
