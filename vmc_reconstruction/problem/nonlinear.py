from __future__ import division
import numpy as np
from dolfin import *
from field.testfield import TestField
set_log_level(WARNING)

FEM = dict()


def setup_space(info):
    global FEM

    # setup mesh and function space
    degree = info['fe']['degree']
    N = info['fe']['mesh size']
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, 'CG', degree)

    FEM.update({"V": V})
    return FEM


def setup(info):
    global FEM

    setup_space(info)
    V = FEM['V']

    # get problem parameters
    fac = Constant(info['problem']['fac'])             # factor in nonlinearity (TODO: might have to be smaller than 0.1)
    kappa = Constant(info['problem']['kappa'])

    FEM.update({'fac': fac, 'kappa': kappa})

    # define boundary condition
    bc = DirichletBC(V, Constant(0.0), 'on_boundary')

    # define variational problem
    u = Function(V)
    v = TestFunction(V)
    f = Constant(1)
    # f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=10)

    FEM.update({"bc": bc, "V": V, "u": u, "v": v, "f": f, "initial_guess": u.vector()})

    # setup random field
    M = info['expansion']['size']
    mean = info['expansion']['mean']
    scale = info['expansion']['scale']
    sigma = info['expansion']['decay rate']
    expfield = info['sampling']['distribution'] == 'normal'
    field = TestField(M, mean=mean, expfield=expfield, scale=scale, sigma=sigma)

    FEM.update({'field': field, 'M': M})

    # compute initial guess
    FEM["initial_guess"] = evaluate_u(np.zeros(M))

    return FEM


def setup_FEM(info):
    return setup(info)


def evaluate_u(y):
    fac = FEM['fac']
    kappa = FEM['kappa']

    def q(fy, u):
        "nonlinear coefficient"
        return (kappa + fac*fy + u)**2
        # return (1 + u)**2

    # construct coefficient
    fy = FEM['field'].realisation(y, FEM['V'])

    u = Function(FEM["V"])
    u.vector()[:] = FEM["initial_guess"]
    # define nonlinear functional and solve
    F = (q(fy, u) * inner(grad(u), grad(FEM["v"])) - FEM["f"] * FEM["v"]) * dx
    solve(F == 0, u, FEM["bc"])
    ret = u.vector().get_local()
    return ret
