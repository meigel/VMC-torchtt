# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

z = np.load("cookie2/moments.npz")
m1, m2 = np.split(z['arr_0'], 2, axis=1)

ns = 2000*np.arange(1, len(m1)+1)
m1err = np.linalg.norm(m1 - m1[-1], axis=1)/np.linalg.norm(m1[-1])
m2err = np.linalg.norm(m2 - m2[-1], axis=1)/np.linalg.norm(m2[-1])

m1err_interp = lambda test_ns: np.interp(test_ns, ns, m1err)
m2err_interp = lambda test_ns: np.interp(test_ns, ns, m2err)

test_ns = np.arange(0, 16000, 100)
o = 8  # 16000 / 2000

f, ax = plt.subplots(1,2)

ax[0].plot(ns[:o], m1err[:o], 'ko-')
ax[0].plot(test_ns, m1err_interp(test_ns), 'r')
ax[0].set_yscale('log')
ax[0].set_title("Die Rundungen kommen vom log-log.")

ax[1].plot(ns[:o], m2err[:o], 'ko-')
ax[1].plot(test_ns, m2err_interp(test_ns), 'r')
ax[1].set_yscale('log')
ax[1].set_title("Die Rundungen kommen vom log-log.")

plt.show()
