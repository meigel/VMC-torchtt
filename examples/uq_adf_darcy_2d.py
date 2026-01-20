import argparse
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


ORTHONORMAL = False


def _legendre_vals(x, degree):
    vals = np.array([np.polynomial.legendre.legval(x, [0] * k + [1]) for k in range(degree)], dtype=float)
    if ORTHONORMAL and degree > 0:
        scale = np.sqrt((2.0 * np.arange(degree) + 1.0) / 2.0)
        vals = vals * scale
    return vals


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


def _mean_rel_err(cores, samples):
    errs = []
    for yvec, ref in samples:
        pred = _eval_tt(cores, yvec)
        denom = np.linalg.norm(ref)
        if denom == 0.0:
            continue
        errs.append(np.linalg.norm(pred - ref) / denom)
    return float(np.mean(errs)) if errs else 0.0


def main(fast=False):
    try:
        from skfem import MeshQuad, ElementQuad1, InteriorBasis, BilinearForm, LinearForm, condense, solve
        from skfem.helpers import dot, grad
    except Exception as exc:
        print("skfem not available:", exc)
        return

    rng = np.random.default_rng(0)
    np.random.seed(0)
    torch.manual_seed(0)

    if fast:
        n = 21
        n_fine = 31
        Ns = 150
        poly_dim = 5
        maxitr = 200
        init_rank = 2
        rank_max = 20
        rank_increase = 2
        rank_every = 10
        als_cg_maxit = 12
        n_plot_samples = 1
        n_val_samples = 1
    else:
        n = 41
        n_fine = 41
        Ns = 300
        poly_dim = 7
        maxitr = 300
        init_rank = 6
        rank_max = 80
        rank_increase = 4
        rank_every = 10
        als_cg_maxit = 20
        n_plot_samples = 3
        n_val_samples = 3

    plot_suffix = "_fast" if fast else ""

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

    def solve_sample(yvec, basis_local=basis, dofs=D):
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

        A = laplace.assemble(basis_local)
        b = load.assemble(basis_local)
        return solve(*condense(A, b, D=dofs))

    meas = uq.UQMeasurementSet()
    train_samples = []
    for _ in range(Ns):
        yvec = rng.uniform(-1.0, 1.0, size=M)
        u = solve_sample(yvec, basis, D)
        meas.add(yvec, u)
        if len(train_samples) < 5:
            train_samples.append((yvec, u))

    val_rng = np.random.default_rng(123)
    val_samples = []
    for _ in range(n_val_samples):
        yvec = val_rng.uniform(-1.0, 1.0, size=M)
        ref_fine = solve_sample(yvec, basis_fine, D_fine)
        val_samples.append((yvec, ref_fine))

    dimensions = [basis.N] + [poly_dim] * M
    history = {"iter": [], "res": [], "train_err": [], "l2_fine": [], "h1_fine": []}

    def _callback(iteration, rel_res, cores, ranks):
        if iteration % 5 != 0:
            return
        cores_np = [c.detach().cpu().numpy() for c in cores]
        history["iter"].append(iteration)
        history["res"].append(rel_res)
        history["train_err"].append(_mean_rel_err(cores_np, train_samples))
        if iteration % 20 == 0:
            l2_vals = []
            h1_vals = []
            for yvec, ref_fine in val_samples:
                pred = _eval_tt(cores_np, yvec)
                pred_fine = basis.interpolator(pred)(mesh_fine.p)
                l2, h1 = _relative_l2_h1(pred_fine - ref_fine, ref_fine, m_fine, k_fine)
                l2_vals.append(l2)
                h1_vals.append(h1)
            history["l2_fine"].append(float(np.mean(l2_vals)))
            history["h1_fine"].append(float(np.mean(h1_vals)))
        else:
            history["l2_fine"].append(np.nan)
            history["h1_fine"].append(np.nan)

    tt = uq.uq_ra_adf(
        meas,
        uq.PolynomBasis.Legendre,
        dimensions,
        targeteps=1e-6,
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
        orthonormal=ORTHONORMAL,
        callback=_callback,
    )
    cores = [c.detach().cpu().numpy() for c in tt.cores]

    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)

    mesh_tri = mesh.to_meshtri()
    tri = mtri.Triangulation(mesh_tri.p[0], mesh_tri.p[1], mesh_tri.t.T)

    y_samples = [rng.uniform(-1.0, 1.0, size=M) for _ in range(n_plot_samples)]
    u_base = solve_sample(np.zeros(M), basis, D)
    u_base_nodes = basis.interpolator(u_base)(mesh.p)
    for idx, yvec in enumerate(y_samples, start=1):
        coeff = coeff_nodes(yvec)
        u_ref = solve_sample(yvec, basis, D)
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
        fig.savefig(os.path.join(out_dir, f"darcy2d_sample_{idx}{plot_suffix}.png"), dpi=150)
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
        fig.savefig(os.path.join(out_dir, f"darcy2d_sample_{idx}_deviation{plot_suffix}.png"), dpi=150)
        plt.close(fig)

    print("Saved Darcy 2D plots to", out_dir)
    if history["iter"]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].semilogy(history["iter"], history["res"], marker="o", lw=1)
        axes[0, 0].set_title("Residual vs iteration")
        axes[0, 0].set_xlabel("iteration")
        axes[0, 0].set_ylabel("relative residual")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].semilogy(history["iter"], history["train_err"], marker="o", lw=1)
        axes[0, 1].set_title("Mean train error")
        axes[0, 1].set_xlabel("iteration")
        axes[0, 1].set_ylabel("relative L2 (coarse)")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].semilogy(history["iter"], history["l2_fine"], marker="o", lw=1)
        axes[1, 0].set_title("Mean L2 error (fine)")
        axes[1, 0].set_xlabel("iteration")
        axes[1, 0].set_ylabel("relative L2")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].semilogy(history["iter"], history["h1_fine"], marker="o", lw=1)
        axes[1, 1].set_title("Mean H1 error (fine)")
        axes[1, 1].set_xlabel("iteration")
        axes[1, 1].set_ylabel("relative H1")
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"darcy2d_convergence{plot_suffix}.png"), dpi=150)
        plt.close(fig)
        print("Saved convergence plot to", os.path.join(out_dir, f"darcy2d_convergence{plot_suffix}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run a faster, lower-accuracy configuration.")
    args = parser.parse_args()
    main(fast=args.fast)
