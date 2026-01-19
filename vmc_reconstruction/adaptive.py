# -*- coding: utf-8 -*-
from __future__ import division, print_function
from dolfin import plot, interactive, Function
import time, numpy as np
from problem.darcy import Problem
from parallel import Parallel, Sequential
def ts(): return time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
log_level = lambda s: ['warning', 'progress', 'info', 'debug'].index(s)
LOG_LEVEL = 0
def set_log_level(s):
    global LOG_LEVEL
    LOG_LEVEL = log_level(s)

set_log_level('progress')

# def marking(_, __): # uniform marking
#     return np.arange(len(indicators))
def marking(vec_est, theta):
    order = np.argsort(vec_est)
    vec_est = vec_est[order]
    sum_est = sum(vec_est)
    cum_est = 0
    for idx, est in enumerate(reversed(vec_est), 1):
        cum_est += est
        if cum_est >= theta*sum_est: break
    return order[-idx:]

def samples(batch_size, dimension):
    if LOG_LEVEL >= log_level('info'): print(ts(), "Sampling parameters")
    if info['expansion']['lognormal']:
        return np.random.randn(batch_size, dimension)
    else:
        return 2*np.random.rand(batch_size, dimension)-1

def moments(yus, nums=1):
    if isinstance(nums, int): nums = [nums]
    if LOG_LEVEL >= log_level('info'): print(ts(), "Computing moments")
    us = np.array([u for y,u in yus])
    return np.hstack([np.mean(us**num, axis=0) for num in nums])


#TODO: dump und refine als callback in den pool geben und nicht mehr explizit auf den pool warten...


# Adaptive iterations
max_iterations = 10

# Refinement fraction
theta = 0.3
max_dofs = 1e4

info = {
    "problem": {"name": "darcy"},
    "fe": {
        "degree": 1,
        # "mesh": 4
        # "mesh": "mesh/tube.xml"
        "mesh": "mesh/reentrant.xml"
    },
    "expansion": {
        "mean": 1.0,
        "scale": 12.5,
        "size": 20,
        "decay rate": 4.0,
        "lognormal": True
    },
    "sampling": {
        "batch size": 40
    }
}

p = Problem(info)

solutions = Parallel(p.solution)
physical_estimators = Parallel(p.residual_estimator)

if LOG_LEVEL >= log_level('info'): print(ts(), "Dofs:", p.dofs())
ys = samples(info['sampling']['batch size'], info['expansion']['size'])
yus = solutions(ys)



from numpy.polynomial.legendre import leggauss, legval

class Projection(object):
    def __init__(self, m, d):
        M = info['expansion']['size']
        coeffs = [0]*M
        coeffs[d] = 1
        self.polynomial = lambda ym: legval(ym, coeffs)
        self.gauss_points = leggauss(4) #TODO: 4
        self.m = m

    def insert(self, ym, y_m):
        M = info['expansion']['size']
        m = self.m
        param = np.empty(M)
        param[:m] = y_m[:m]
        param[m] = ym
        param[m+1:] = y_m[m:]
        return param

    def __call__(self, y_m):
        M = info['expansion']['size']
        if self.m >= M: return y_m, 0

        wms, yms = self.gauss_points
        P = self.polynomial
        projection = 0
        for wm, ym in zip(wms, yms):
            y = self.insert(ym, y_m)
            yus = p.solution(y)
            u = yus[0][1]
            projection += wm*u*P(ym)
        return y_m, projection


stochastic_dimensions = [1,0]

def stochastic_estimator(md):
    m,d = md
    M = info['expansion']['size']
    if LOG_LEVEL >= log_level('info'): print(ts(), "Computing stochastic estimator for component", m)
    y_ms = samples(info['sampling']['batch size'], M-1)
    Pr = Projection(m,d)
    yprs = [Pr(y_m) for y_m in y_ms]
    return moments(yprs, 2)

stochastic_estimators = Parallel(stochastic_estimator)

def increase_dimensions(modes):
    for m in modes:
        stochastic_dimensions[m] += 1
    if stochastic_dimensions[-1] != 0:
        stochastic_dimensions.append(0)



for i in range(1, max_iterations+1):
    print(ts(), "Refinement", i)

    if p.dofs() >= max_dofs:
        break

    # Refine mesh
    physical_indicators = moments(physical_estimators(yus))
    print(ts(), "Total physical error estimate:", physical_indicators.sum())
    marked_cells = marking(physical_indicators, theta)
    print(ts(), "Refining %d cells"%(len(marked_cells)))
    p.refine_mesh(marked_cells)
    print(ts(), "Dofs:", p.dofs())

    # Increase stochastic basis dimension
    stochastic_indicators = stochastic_estimators(list(enumerate(stochastic_dimensions)))
    stochastic_indicators = np.concatenate(stochastic_indicators)
    print(ts(), "Total stochastic error estimate:", sum(stochastic_indicators))
    marked_modes = marking(stochastic_indicators, theta)
    print(ts(), "Increase dimension of %d modes"%(len(marked_modes)))
    increase_dimensions(marked_modes)
    print(ts(), "Dimensions:", stochastic_dimensions)

    # Solve problem
    ys = samples(info['sampling']['batch size'], info['expansion']['size'])
    yus = solutions(ys)


# Plot 5 samples of the solution
for _,u_vec in yus[:5]:
    u = Function(p.space)
    u.vector()[:] = u_vec
    plot(u)

# Plot expectation of the solution
u_vec = moments(yus)
u = Function(p.space)
u.vector()[:] = u_vec
plot(u)
plot(p.mesh)

# Hold plots
interactive()

