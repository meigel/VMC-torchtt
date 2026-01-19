import os
import sys
import warnings

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

warnings.filterwarnings("ignore", category=UserWarning)

from vmc_reconstruction import uq_adf_torchtt as uq


def _legendre_vals(x, degree):
    return np.array([np.polynomial.legendre.legval(x, [0] * k + [1]) for k in range(degree)], dtype=float)


def _eval_tt(cores, yvec):
    val = np.asarray(cores[0][0, :, :], dtype=float)
    for dim, yi in enumerate(yvec, start=1):
        core = cores[dim]
        basis_vals = _legendre_vals(yi, core.shape[1])
        tmp = np.tensordot(core, basis_vals, axes=([1], [0]))
        val = val @ tmp
    return np.squeeze(val)


def _plot_field(ax, tri, values, title, vmin=None, vmax=None, cmap="viridis"):
    col = ax.tripcolor(tri, values, shading="gouraud", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return col


def main():
    try:
        from skfem import MeshQuad, ElementQuad1, InteriorBasis, BilinearForm, LinearForm, condense, solve
        from skfem.helpers import dot, grad
    except Exception as exc:
        print("skfem not available:", exc)
        return

    rng = np.random.default_rng(0)
    np.random.seed(0)
    torch.manual_seed(0)

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

    def coeff_nodes(yvec):
        x, y = mesh.p[0], mesh.p[1]
        coeff = np.zeros_like(x)
        for i, k in enumerate(ks):
            coeff += (
                np.sqrt(lambdas[i])
                * np.sin(np.pi * k * x)
                * np.sin(np.pi * k * y)
                * yvec[i]
            )
        return np.exp(sigma * coeff)

    def solve_sample(yvec):
        @BilinearForm
        def laplace(u, v, w):
            x, y = w.x
            coeff = np.zeros_like(x)
            for i, k in enumerate(ks):
                coeff += (
                    np.sqrt(lambdas[i])
                    * np.sin(np.pi * k * x)
                    * np.sin(np.pi * k * y)
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
    for _ in range(Ns):
        yvec = rng.uniform(-1.0, 1.0, size=M)
        u = solve_sample(yvec)
        meas.add(yvec, u)

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

    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)

    mesh_tri = mesh.to_meshtri()
    tri = mtri.Triangulation(mesh_tri.p[0], mesh_tri.p[1], mesh_tri.t.T)

    y_samples = [rng.uniform(-1.0, 1.0, size=M) for _ in range(3)]
    u_base = solve_sample(np.zeros(M))
    u_base_nodes = basis.interpolator(u_base)(mesh.p)
    for idx, yvec in enumerate(y_samples, start=1):
        coeff = coeff_nodes(yvec)
        u_ref = solve_sample(yvec)
        u_ref_nodes = basis.interpolator(u_ref)(mesh.p)
        u_pred = _eval_tt(cores, yvec)
        u_pred_nodes = basis.interpolator(u_pred)(mesh.p)
        err = np.abs(u_pred_nodes - u_ref_nodes)

        vmin = float(min(u_ref_nodes.min(), u_pred_nodes.min()))
        vmax = float(max(u_ref_nodes.max(), u_pred_nodes.max()))

        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        c0 = _plot_field(axes[0, 0], tri, coeff, "lognormal a(x,y)", cmap="viridis")
        fig.colorbar(c0, ax=axes[0, 0])
        c1 = _plot_field(axes[0, 1], tri, u_ref_nodes, "reference u", vmin=vmin, vmax=vmax, cmap="viridis")
        fig.colorbar(c1, ax=axes[0, 1])
        c2 = _plot_field(axes[1, 0], tri, u_pred_nodes, "reconstruction u", vmin=vmin, vmax=vmax, cmap="viridis")
        fig.colorbar(c2, ax=axes[1, 0])
        c3 = _plot_field(axes[1, 1], tri, err, "abs error", cmap="magma")
        fig.colorbar(c3, ax=axes[1, 1])
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"darcy2d_sample_{idx}.png"), dpi=150)
        plt.close(fig)

        dev_ref = u_ref_nodes - u_base_nodes
        dev_pred = u_pred_nodes - u_base_nodes
        dvmin = float(min(dev_ref.min(), dev_pred.min()))
        dvmax = float(max(dev_ref.max(), dev_pred.max()))
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        c0 = _plot_field(axes[0], tri, dev_ref, "reference - a=1", vmin=dvmin, vmax=dvmax, cmap="coolwarm")
        fig.colorbar(c0, ax=axes[0])
        c1 = _plot_field(axes[1], tri, dev_pred, "reconstruction - a=1", vmin=dvmin, vmax=dvmax, cmap="coolwarm")
        fig.colorbar(c1, ax=axes[1])
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"darcy2d_sample_{idx}_deviation.png"), dpi=150)
        plt.close(fig)

    print("Saved Darcy 2D plots to", out_dir)


if __name__ == "__main__":
    main()
