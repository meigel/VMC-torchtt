from __future__ import division, print_function
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import os
import numpy as np
import scipy.sparse as sps
import scikits.sparse.cholmod as cholmod
from dolfin import *
from load_info import get_paths, load_info, load_problem
set_log_level(WARNING)
parameters.linear_algebra_backend = "Eigen"

def get_mass_matrix(fs):
    u = TrialFunction(fs)
    v = TestFunction(fs)
    mass = inner(u, v) * dx
    mass = assemble(mass)

    M = sps.csr_matrix(as_backend_type(mass).sparray())
    return M

def get_stiffness_matrix(fs):
    u = TrialFunction(fs)
    v = TestFunction(fs)
    stiffness = inner(grad(u), grad(v)) * dx
    stiffness = assemble(stiffness)

    S = sps.csr_matrix(as_backend_type(stiffness).sparray())
    return S

def cholesky(S):
    eps = np.finfo(S.dtype).eps

    m = sps.diags(1/np.sqrt(S.diagonal()))
    mI = sps.diags(np.sqrt(S.diagonal()))
    T = m.dot(S).dot(m)
    T.eliminate_zeros()

    factor = cholmod.cholesky(T.tocsc(), eps)
    PL,DD = factor.L_D()
    DD_diag = DD.diagonal()
    DD_diag[DD_diag<0] = 0
    D = sps.diags(np.sqrt(DD_diag))
    L = mI.dot(factor.apply_Pt(PL.tocsc()).dot(D))
    P = factor.apply_P(sps.eye(S.shape[0], format='csc'))

    assert np.linalg.norm((L.dot(L.T)-S).data) < 1e-12
    assert len(sps.triu(P.dot(L), 1).data) == 0

    return P, L


if __name__=='__main__':
    descr = """Compute the stiffness matrix `S`, its Cholesky factor `L` and a permutation `P` for the given problem.
The three matrices satisfy the identity `P S P^T = L L^T`.
They will be stored in `PATH/stiffness.npz`, `PATH/cholesky.npz` and `PATH/cholesky_permutation.npz` respectively.
"""
    path, = get_paths(descr, ('PATH', 'path to the directory containing the `info.json` for the specific problem'))
    info = load_info(path)
    problem = load_problem(info)
    V = problem.setup_space(info)['V']

    stiff_path = os.path.join(path, 'stiffness.npz')
    S = get_stiffness_matrix(V)
    print("Saving stiffness matrix: '{}'".format(stiff_path))
    sps.save_npz(stiff_path, S)

    chol_path = os.path.join(path, 'cholesky.npz')
    chol_perm_path = os.path.join(path, 'cholesky_permutation.npz')
    P,L = cholesky(S)
    print("Saving cholesky factor: '{}'".format(chol_path))
    sps.save_npz(chol_path, L)
    print("Saving cholesky permutation matrix: '{}'".format(chol_perm_path))
    sps.save_npz(chol_perm_path, P)
