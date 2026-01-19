# coding: utf-8
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import os, argparse, json, glob
from load_info import load_info


parser = argparse.ArgumentParser()
parser.add_argument('PATH', help='path to the directory containing the `info.json` for reconstruction')
args = parser.parse_args()
reco_path = args.PATH
reco_info = load_info(reco_path)
sample_path = reco_info["sample path"]
sampling_info = load_info(sample_path)
reference_path = reco_info["reference path"]

files = glob.glob(reco_path+'/results/*/reconstruction.json')
files = sorted(files, key=lambda f_name: int(f_name.split('/')[2]))


class SampleLoader(object):
    def __init__(self, path):
        self.buffer = []
        self.buffer_pos = 0
        self.f_name = os.path.join(path, '{}.npz')
        self.f_idx = 0
        self.max_f_idx = len(glob.glob(self.f_name.format('*')))-1

    def get(self, batch_size):
        ret = []
        while batch_size > 0:
            if self.buffer_pos >= len(self.buffer):
                if self.f_idx > self.max_f_idx:
                    raise RuntimeError("Not enough files under '{}'".format(self.f_name.format('*')))
                bz = np.load(self.f_name.format(self.f_idx))
                self.buffer = bz['us']
                self.buffer_pos = 0
                self.f_idx += 1
            batch = self.buffer[self.buffer_pos:self.buffer_pos+batch_size]
            self.buffer_pos += batch_size
            ret.append(batch)
            batch_size -= len(batch)
        return np.concatenate(ret)


Ns = []
for f_name in files:
    with open(f_name, 'r') as f:
        data = json.load(f)
    Ns.append(data['N'])

def compute_moments(sample_loader):
    samples = np.concatenate([[0], Ns])
    moments = [(0,0)]
    for i in range(1, len(samples)):
        us = sample_loader.get(samples[i]-samples[i-1])
        om1, om2 = moments[-1]
        m1 = np.mean(us, axis=0)
        m2 = np.mean(us**2, axis=0)
        m1 = (om1-m1)*(samples[i-1]/samples[i]) + m1
        m2 = (om2-m2)*(samples[i-1]/samples[i]) + m2
        moments.append((m1,m2))
    return zip(*moments[1:])

mc_samples = SampleLoader(os.path.join(sample_path, 'samples'))
mc_m1, mc_m2 = compute_moments(mc_samples)

qmc_samples = SampleLoader(os.path.join(reference_path, 'samples'))
qmc_m1, qmc_m2 = compute_moments(qmc_samples)

#TODO: recompute both with a hierarchical sum...

def load_reference(path):
    f_name = os.path.join(path, 'moments.npz')
    # print("Load reference: '%s'"%f_name)
    moments = np.load(f_name)['arr_0']
    return np.split(moments[-1], 2)
ref_m1, ref_m2 = load_reference(reference_path)
ref_m1_norm = np.linalg.norm(ref_m1)
ref_m2_norm = np.linalg.norm(ref_m2)
mc_m1 = np.linalg.norm(mc_m1 - ref_m1, axis=1) / ref_m1_norm
mc_m2 = np.linalg.norm(mc_m2 - ref_m2, axis=1) / ref_m2_norm
qmc_m1 = np.linalg.norm(qmc_m1 - ref_m1, axis=1) / ref_m1_norm
qmc_m2 = np.linalg.norm(qmc_m2 - ref_m2, axis=1) / ref_m2_norm


lines = []
for e, f_name in enumerate(files):
    with open(f_name, 'r') as f:
        data = json.load(f)
    row = [ data['N'],
            # data['mc error']['moment 1'], data['mc error']['moment 2'],
            mc_m1[e], mc_m2[e],
            qmc_m1[e], qmc_m2[e],
            data['reco error']['moment 1'], data['reco error']['moment 2'],
            data['testError']
          ]
    lines.append(' '.join(map(str, row)))

problem = sampling_info['problem']['name']
f_name = reco_path+'/{}.data'.format(problem)
with open(f_name, 'w') as f:
    for line in lines:
        f.write(line+'\n')
