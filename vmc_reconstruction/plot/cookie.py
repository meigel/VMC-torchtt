# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

f_name = 'results/cookie4_single/N{N}/ord{ord}_ortho_50k/reconstruction.npz'


plt.figure()
# plt.suptitle("Coefficient Norms", fontsize=16)

colors = ['xkcd:emerald', 'xkcd:cobalt']
markers = 'sDo^><v'

plt.subplot(121)
plt.title("Polynomial Coefficients")
marker_idx = 0
for color_idx,d in enumerate([25,40]):
    for marker_idx,N in enumerate([50,100,200,400]):
        try:
            tz = np.load(f_name.format(N=N,ord=d))
        except:
            continue
        t0 = tz['cmp_0'][0]       # first index is dummy
        t1 = tz['cmp_1'][..., 0]  # last index is dummy
        ns = []
        for i in range(t1.shape[1]):
            ns.append(np.linalg.norm(t0.dot(t1[:,i])))
        ns = np.array(ns)
        ps = ns/sum(ns)
        # print(N,d,ps)
        plt.plot(ps, color=colors[color_idx], marker=markers[marker_idx], linestyle='none', label="N%d_ord%d"%(N,d))
plt.xlabel("basis function")
plt.yscale("log")
plt.ylabel("norm")
plt.legend()
plt.tight_layout()

plt.subplot(122)
plt.title("Singular Values")
marker_idx = 0
for color_idx,d in enumerate([25,40]):
    for marker_idx,N in enumerate([50,100,200,400]):
        try:
            tz = np.load(f_name.format(N=N,ord=d))
        except:
            continue
        t0 = tz['cmp_0'][0]
        # ss = np.linalg.svd(t0)[1]
        ss = np.linalg.norm(t0, axis=0)
        # print(N,d,ss)
        plt.plot(ss, color=colors[color_idx], marker=markers[marker_idx], linestyle='none', label="N%d_ord%d"%(N,d))
plt.xlabel("basis function")
plt.yscale("log")
plt.ylabel("value")
plt.legend()
plt.tight_layout()

plt.show()
