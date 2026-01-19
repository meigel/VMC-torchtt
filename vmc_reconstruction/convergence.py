# coding: utf-8
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import os, argparse, glob

parser = argparse.ArgumentParser()
parser.add_argument('DATA_PATH', help='path to the data file')
parser.add_argument('FIG_PATH', help='path to the plot')
args = parser.parse_args()
data_path = args.DATA_PATH
fig_path = args.FIG_PATH


print(data_path, "-->", fig_path)

with open(data_path, 'r') as f:
    data = f.readlines()

rows = []
for line in data:
    row = line.split()
    rows.append(row)

Ns, m1s, m2s, q1s, q2s, r1s, r2s, errs = zip(*rows)

Ns = np.array(Ns, dtype=int)
m1s = np.array(m1s, dtype=float)
m2s = np.array(m2s, dtype=float)
q1s = np.array(q1s, dtype=float)
q2s = np.array(q2s, dtype=float)
r1s = np.array(r1s, dtype=float)
r2s = np.array(r2s, dtype=float)
errs = np.array(errs, dtype=float)

# f,ax = plt.subplots(1,2)

# print(plt.rcParams['figure.figsize'])
f,ax = plt.subplots(1,1, figsize=(6, 3.5))
ax = [ax]

ax[0].set_prop_cycle(cycler(color=plt.cm.get_cmap('tab20').colors))
ax[0].plot(Ns, m1s, label='MC M1')
ax[0].plot(Ns, m2s, label='MC M2')
ax[0].plot(Ns, q1s, label='QMC M1')
ax[0].plot(Ns, q2s, label='QMC M2')
ax[0].plot(Ns, r1s, label='ERM M1')
ax[0].plot(Ns, r2s, label='ERM M2')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlabel('# samples')
ax[0].set_ylabel('rel. error')
ax[0].legend()

# ax[1].plot(Ns, errs, label='Pointwise error')
# ax[1].set_xscale('log')
# ax[1].set_yscale('log')
# ax[1].set_xlabel('# samples')
# ax[1].set_ylabel('error')
# ax[1].legend()

# plt.suptitle("Reconstruction of '{}'".format(problem), fontsize=16)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.tight_layout()
# plt.show()
# plt.savefig("figures/{}.png".format(problem), dpi=300)
# plt.savefig(f_mask.format(fig_path, name, "png"), dpi=300)
plt.savefig(fig_path, dpi=300)
