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
    kappa = Constant(info['problem']['kappa'])

    FEM.update({'kappa': kappa})

    # define boundary condition
    bc = DirichletBC(V, Constant(0.0), 'on_boundary')

    # define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1)
    # f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=10)
    try: h = CellDiameter(V.mesh())
    except NameError: h = CellSize(V.mesh())

    FEM.update({"bc": bc, "V": V, "u": u, "v": v, "f": f, "h": h})

    # setup random field
    M = info['expansion']['size']
    mean = info['expansion']['mean']
    scale = info['expansion']['scale']
    sigma = info['expansion']['decay rate']
    expfield = info['sampling']['distribution'] == 'normal'
    field = TestField(M, mean=mean, expfield=expfield, scale=scale, sigma=sigma)

    FEM.update({'field': field, 'M': M})

    return FEM


def setup_FEM(info):
    return setup(info)


def evaluate_u(y):
    h = FEM['h']
    kappa = FEM['kappa']

    def vel(fy):
        vel = as_vector([0.99-fy,(0.99-abs(fy))])
        return vel

    # define discretisation
    u, v, f = FEM["u"], FEM["v"], FEM["f"]
    fy = FEM['field'].realisation(y, FEM["V"])
    a = (inner(vel(fy), grad(u)) * v + kappa * inner(grad(u), grad(v)) +
                    (h / (2.0 * sqrt(inner(vel(fy), vel(fy))))) * inner(vel(fy), grad(v)) * (inner(vel(fy), grad(u)) - kappa * div(grad(u)))) * dx
    L = ((f * v) + (h / (2.0 * sqrt(inner(vel(fy), vel(fy))))) * inner(vel(fy), grad(v)) * f) * dx

    # solve
    uy = Function(FEM["V"])
    solve(a == L, uy, FEM["bc"])
    ret = uy.vector().get_local()
    return ret
