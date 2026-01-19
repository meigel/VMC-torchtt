from __future__ import division
import numpy as np
from dolfin import *
from field.cookiefield import CookieField
set_log_level(WARNING)


DEBUG = False
FEM = dict()


def setup_space(info):
    global FEM

    # setup mesh and function space
    degree = info['fe']['degree']
    N = info['fe']['mesh size']
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, 'CG', degree)

    FEM.update({"V": V, "N": N})
    return FEM


def setup(info):
    global FEM

    setup_space(info)
    V = FEM['V']

    # get problem parameters
    D = info['problem']['subdomains per axis']
    dc = info['problem']['dough coefficient']
    sc = info['problem']['sprinkles coefficient']

    FEM.update({'dc': dc, 'sc': sc, 'D': D})

    # define boundary condition
    bc = DirichletBC(V, Constant(0.0), 'on_boundary')

    # define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1)
    # f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=10)

    FEM.update({"bc": bc, "u": u, "v": v, "f": f})

    # setup random field
    M = D**2
    assert info['expansion']['size'] == M
    field = CookieField(D, prepare_cookie_subdomain_data, kappa=dc)

    FEM.update({'field': field, 'M': M})

    return FEM


def setup_FEM(info):
    return setup(info)


def evaluate_u(y):
    if DEBUG: print('evaluate_u')
    # discretisation on reference space
    u, v, f = FEM["u"], FEM["v"], FEM["f"]
    mu = FEM['field'].realisation(y, FEM["V"])
    a = mu * inner(grad(u), grad(v)) * dx
    L = f * v * dx

    if DEBUG: print('\tsolve')
    uy = Function(FEM["V"])
    solve(a == L, uy, FEM["bc"])
    ret = uy.vector().get_local()
    return ret


def cart(*args):
    "Cartesian product"
    return np.array(np.meshgrid(*args)).T.reshape(-1, len(args))


def prepare_cookie_subdomain_data(fyc, cookie_D):
    if DEBUG: print('prepare_cookie_subdomain_data')
    # prepare cookie subdomain data
    sc = FEM['sc']
    fy1 = (fyc+1)/2    # rescale `fyc` to be in [0,1]
    cx = 1 / cookie_D  # subdomain size
    rxM = cx / 2       # maximal cookie radius

    sc = FEM['N'] / cookie_D   # number of cells per subdomain
    rxm =  (5 / 2) * cx / sc  # minimal cookie radius (every cookie should contain at least 5 cells)

    idcs = np.arange(cookie_D)
    centers = rxM + cx*cart(idcs, idcs).T  # centers of the cookies
    coeffs  = np.full(cookie_D**2, sc)     # coefficients inside the cookies
    radii   = rxm + (rxM-rxm)*fy1          # radii of the cookies

    # data format: (center[0], center[1], coefficient, radius)
    subdomain_data = np.stack([centers[0], centers[1], coeffs, radii]).T
    return subdomain_data
