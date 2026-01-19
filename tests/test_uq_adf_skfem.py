import numpy as np
import pytest
import torch

from vmc_reconstruction import uq_adf_torchtt as uq


def _legendre_vals(x, degree):
    return np.array([np.polynomial.legendre.legval(x, [0] * k + [1]) for k in range(degree)], dtype=float)


def _eval_tt(cores, y):
    val = np.asarray(cores[0][0, :, :], dtype=float)
    for dim, yi in enumerate(y, start=1):
        core = cores[dim]
        basis_vals = _legendre_vals(yi, core.shape[1])
        tmp = np.tensordot(core, basis_vals, axes=([1], [0]))
        val = val @ tmp
    return np.squeeze(val)


def test_uq_adf_darcy_log_normal_skfem():
    skfem = pytest.importorskip("skfem")
    pytest.importorskip("scipy")
    from skfem import MeshQuad, ElementQuad1, InteriorBasis, BilinearForm, LinearForm, condense, solve
    from skfem.helpers import dot, grad

    np.random.seed(0)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    n = 9
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

    M = 3
    ks = np.arange(1, M + 1)
    lambdas = 1.0 / (ks**2)
    sigma = 0.1

    @LinearForm
    def load(v, w):
        return 1.0 * v

    def solve_sample(yvec):
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

        A = laplace.assemble(basis)
        b = load.assemble(basis)
        return solve(*condense(A, b, D=D))

    Ns = 200
    poly_dim = 5
    meas = uq.UQMeasurementSet()
    train = []
    for _ in range(Ns):
        yvec = rng.uniform(-1.0, 1.0, size=M)
        u = solve_sample(yvec)
        meas.add(yvec, u)
        train.append((yvec, u))

    dimensions = [basis.N] + [poly_dim] * M
    tt = uq.uq_ra_adf(
        meas,
        uq.PolynomBasis.Legendre,
        dimensions,
        targeteps=1e-6,
        maxitr=400,
        device=torch.device("cpu"),
        dtype=torch.float64,
        init_rank=2,
        init_noise=1e-2,
        adapt_rank=True,
        rank_increase=2,
        rank_every=10,
        rank_noise=1e-2,
        rank_max=20,
    )

    cores = [c.detach().cpu().numpy() for c in tt.cores]
    train_errs = []
    for yvec, ref in train[:5]:
        pred = _eval_tt(cores, yvec)
        train_errs.append(np.linalg.norm(pred - ref) / np.linalg.norm(ref))

    eval_errs = []
    for _ in range(5):
        yvec = rng.uniform(-1.0, 1.0, size=M)
        ref = solve_sample(yvec)
        pred = _eval_tt(cores, yvec)
        eval_errs.append(np.linalg.norm(pred - ref) / np.linalg.norm(ref))

    assert float(np.mean(train_errs)) < 1e-3
    assert float(np.mean(eval_errs)) < 1e-2
