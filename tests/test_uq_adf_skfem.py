import os
import numpy as np
import pytest
import torch

from vmc_reconstruction import uq_adf_torchtt as uq


ORTHONORMAL = False
SLOW_ENV = "RUN_SLOW_DARCY"


def _legendre_vals(x, degree):
    vals = np.array([np.polynomial.legendre.legval(x, [0] * k + [1]) for k in range(degree)], dtype=float)
    if ORTHONORMAL and degree > 0:
        scale = np.sqrt((2.0 * np.arange(degree) + 1.0) / 2.0)
        vals = vals * scale
    return vals


def _eval_tt(cores, y):
    val = np.asarray(cores[0][0, :, :], dtype=float)
    for dim, yi in enumerate(y, start=1):
        core = cores[dim]
        basis_vals = _legendre_vals(yi, core.shape[1])
        tmp = np.tensordot(core, basis_vals, axes=([1], [0]))
        val = val @ tmp
    return np.squeeze(val)


def _assemble_l2_h1(basis):
    from skfem import BilinearForm
    from skfem.helpers import dot, grad

    @BilinearForm
    def mass(u, v, w):
        return u * v

    @BilinearForm
    def stiff(u, v, w):
        return dot(grad(u), grad(v))

    m = mass.assemble(basis)
    k = stiff.assemble(basis)
    return m, k


def _relative_l2_h1(err, ref, m, k):
    err = np.asarray(err, dtype=float)
    ref = np.asarray(ref, dtype=float)
    l2_num = float(err @ (m @ err))
    l2_den = float(ref @ (m @ ref))
    h1_num = float(err @ ((m + k) @ err))
    h1_den = float(ref @ ((m + k) @ ref))
    return np.sqrt(l2_num / l2_den), np.sqrt(h1_num / h1_den)


def _run_darcy_case(
    n,
    n_fine,
    ns,
    poly_dim,
    maxitr,
    init_rank,
    rank_max,
    rank_increase,
    rank_every,
    als_cg_maxit,
    eval_count=3,
):
    skfem = pytest.importorskip("skfem")
    pytest.importorskip("scipy")
    from skfem import MeshQuad, ElementQuad1, InteriorBasis, BilinearForm, LinearForm, condense, solve
    from skfem.helpers import dot, grad

    np.random.seed(0)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    mesh = MeshQuad.init_tensor(np.linspace(0, 1, n), np.linspace(0, 1, n))
    mesh = mesh.with_boundaries(
        {
            "left": lambda x: x[0] == 0.0,
            "right": lambda x: x[0] == 1.0,
            "bottom": lambda x: x[1] == 0.0,
            "top": lambda x: x[1] == 1.0,
        }
    )

    basis = InteriorBasis(mesh, ElementQuad1(), intorder=3)
    D = basis.get_dofs(list(mesh.boundaries.keys()))

    mesh_fine = MeshQuad.init_tensor(np.linspace(0, 1, n_fine), np.linspace(0, 1, n_fine))
    mesh_fine = mesh_fine.with_boundaries(
        {
            "left": lambda x: x[0] == 0.0,
            "right": lambda x: x[0] == 1.0,
            "bottom": lambda x: x[1] == 0.0,
            "top": lambda x: x[1] == 1.0,
        }
    )
    basis_fine = InteriorBasis(mesh_fine, ElementQuad1(), intorder=3)
    D_fine = basis_fine.get_dofs(list(mesh_fine.boundaries.keys()))
    m_fine, k_fine = _assemble_l2_h1(basis_fine)

    M = 5
    ks = np.arange(1, M + 1)
    lambdas = 1.0 / (ks**2)
    sigma = 0.1

    @LinearForm
    def load(v, w):
        return 1.0 * v

    def solve_sample(yvec, basis_local, dofs):
        @BilinearForm
        def laplace(u, v, w):
            x, ycoord = w.x
            coeff = np.zeros_like(x)
            for i, k in enumerate(ks):
                coeff += (
                    np.sqrt(lambdas[i])
                    * np.sin(np.pi * k * x)
                    * np.sin(np.pi * k * ycoord)
                    * yvec[i]
                )
            a = np.exp(sigma * coeff)
            return a * dot(grad(u), grad(v))

        A = laplace.assemble(basis_local)
        b = load.assemble(basis_local)
        return solve(*condense(A, b, D=dofs))

    meas = uq.UQMeasurementSet()
    train = []
    for _ in range(ns):
        yvec = rng.uniform(-1.0, 1.0, size=M)
        u = solve_sample(yvec, basis, D)
        meas.add(yvec, u)
        train.append((yvec, u))

    dimensions = [basis.N] + [poly_dim] * M
    tt = uq.uq_ra_adf(
        meas,
        uq.PolynomBasis.Legendre,
        dimensions,
        targeteps=1e-7,
        maxitr=maxitr,
        device=torch.device("cpu"),
        dtype=torch.float64,
        init_rank=init_rank,
        init_noise=1e-2,
        adapt_rank=True,
        rank_increase=rank_increase,
        rank_every=rank_every,
        rank_noise=1e-2,
        rank_max=rank_max,
        update_rule="als",
        als_reg=1e-8,
        als_cg_maxit=als_cg_maxit,
        als_cg_tol=1e-6,
        orthonormal=ORTHONORMAL,
    )

    cores = [c.detach().cpu().numpy() for c in tt.cores]
    train_errs = []
    for yvec, ref in train[:5]:
        pred = _eval_tt(cores, yvec)
        train_errs.append(np.linalg.norm(pred - ref) / np.linalg.norm(ref))

    eval_errs = []
    l2_errs = []
    h1_errs = []
    for _ in range(eval_count):
        yvec = rng.uniform(-1.0, 1.0, size=M)
        ref = solve_sample(yvec, basis, D)
        pred = _eval_tt(cores, yvec)
        eval_errs.append(np.linalg.norm(pred - ref) / np.linalg.norm(ref))

        ref_fine = solve_sample(yvec, basis_fine, D_fine)
        pred_fine = basis.interpolator(pred)(mesh_fine.p)
        l2, h1 = _relative_l2_h1(pred_fine - ref_fine, ref_fine, m_fine, k_fine)
        l2_errs.append(l2)
        h1_errs.append(h1)

    return float(np.mean(train_errs)), float(np.mean(eval_errs)), float(np.mean(l2_errs)), float(np.mean(h1_errs))


def test_uq_adf_darcy_log_normal_skfem_fast():
    train_err, eval_err, l2_err, h1_err = _run_darcy_case(
        n=21,
        n_fine=21,
        ns=80,
        poly_dim=5,
        maxitr=150,
        init_rank=2,
        rank_max=20,
        rank_increase=2,
        rank_every=10,
        als_cg_maxit=10,
        eval_count=2,
    )
    assert train_err < 0.1
    assert eval_err < 0.2
    assert l2_err < 0.1
    assert h1_err < 0.5


def test_uq_adf_darcy_log_normal_skfem():
    if os.environ.get(SLOW_ENV, "").lower() not in ("1", "true", "yes"):
        pytest.skip("set RUN_SLOW_DARCY=1 to run slow Darcy test")
    train_err, eval_err, l2_err, h1_err = _run_darcy_case(
        n=41,
        n_fine=41,
        ns=300,
        poly_dim=7,
        maxitr=300,
        init_rank=6,
        rank_max=80,
        rank_increase=4,
        rank_every=10,
        als_cg_maxit=20,
        eval_count=3,
    )
    assert train_err < 1e-3
    assert eval_err < 1e-2
    assert l2_err < 1e-3
    assert h1_err < 5e-2
