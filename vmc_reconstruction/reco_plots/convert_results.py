# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os, argparse, json, glob
from load_info import load_info


parser = argparse.ArgumentParser()
parser.add_argument('RECO_PATH', help='path to the directory containing the `info.json` for reconstruction')
parser.add_argument('REF_PATH', help='path to the directory containing the `info.json` for the reference sampling (qmc)')
parser.add_argument('OUT_PATH', help='path to the output file')
args = parser.parse_args()
reco_path = args.RECO_PATH
ref_path = args.REF_PATH
out_path = args.OUT_PATH

# reco_path = args.PATH
# reco_info = load_info(reco_path)
# sample_path = reco_info["sample path"]
# reference_path = reco_info["reference path"]
# sampling_info = load_info(sample_path)
# problem = sampling_info['problem']['name']

# files = glob.glob(reco_path+'/results/*/reconstruction.json')
files = glob.glob(os.path.join(reco_path,'results', '*', 'reconstruction.json'))
files = sorted(files, key=lambda f_name: int(f_name.split('/')[-2]))

def load_reference(path):
    f_name = os.path.join(path, 'moments.npz')
    print("Load reference: '%s'"%f_name)
    moments = np.load(f_name)['arr_0']
    return np.split(moments, 2, axis=1)
# qmcM1, qmcM2 = load_reference(reference_path)
qmcM1, qmcM2 = load_reference(ref_path)
qmcNs = 2000*np.arange(1, len(qmcM1)+1)
qmcM1 = np.linalg.norm(qmcM1 - qmcM1[-1], axis=1)/np.linalg.norm(qmcM1[-1])
qmcM2 = np.linalg.norm(qmcM2 - qmcM2[-1], axis=1)/np.linalg.norm(qmcM2[-1])
qmcNs = np.concatenate([[0], qmcNs])
qmcM1 = np.concatenate([[2*qmcM1[0] - qmcM1[1]], qmcM1])
qmcM2 = np.concatenate([[2*qmcM2[0] - qmcM2[1]], qmcM2])
from scipy import interpolate
qmcM1 = interpolate.interp1d(qmcNs, qmcM1)
qmcM2 = interpolate.interp1d(qmcNs, qmcM2)

lines = []
for f_name in files:
    with open(f_name, 'r') as f:
        data = json.load(f)
    row = [ data['N'],
            data['mc error']['moment 1'], data['mc error']['moment 2'],
            qmcM1(data['N']), qmcM2(data['N']),
            data['reco error']['moment 1'], data['reco error']['moment 2'],
            data['testError']
          ]
    lines.append(' '.join(map(str, row)))

# with open(reco_path+'/{}.data'.format(problem), 'w') as f:
with open(out_path, 'w') as f:
    for line in lines:
        f.write(line+'\n')
