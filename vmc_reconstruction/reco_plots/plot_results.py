# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import argparse, json, glob
from load_info import load_info


parser = argparse.ArgumentParser()
parser.add_argument('PATH', help='path to the directory containing the `info.json` for reconstruction')
args = parser.parse_args()
reco_path = args.PATH
reco_info = load_info(reco_path)
sample_path = reco_info["sample path"]
sampling_info = load_info(sample_path)

files = glob.glob(reco_path+'/results/*/reconstruction.json')
files = sorted(files, key=lambda f_name: int(f_name.split('/')[2]))

Ns = []
m1s = []
m2s = []
r1s = []
r2s = []
errs = []
for f_name in files:
    with open(f_name, 'r') as f:
        data = json.load(f)
    Ns.append(data['N'])
    m1s.append(data['mc error']['moment 1'])
    m2s.append(data['mc error']['moment 2'])
    r1s.append(data['reco error']['moment 1'])
    r2s.append(data['reco error']['moment 2'])
    errs.append(data['testError'])

f,ax = plt.subplots(1,2)

ax[0].plot(Ns, m1s, label='MC M1')
ax[0].plot(Ns, m2s, label='MC M2')
ax[0].plot(Ns, r1s, label='RECO M1')
ax[0].plot(Ns, r2s, label='RECO M2')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].legend()

ax[1].plot(Ns, errs, label='Pointwise error')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].legend()

plt.suptitle("Reconstruction of '{}'".format(sampling_info['problem']['name']), fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
