import os
import sys
import warnings

import math
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _hermite_vals(x, degree):
    vals = np.zeros(degree, dtype=float)
    if degree == 0:
        return vals
    vals[0] = 1.0
    if degree == 1:
        vals[1] = x
        return vals
    vals[1] = x
    for n in range(1, degree - 1):
        vals[n + 1] = x * vals[n] - float(n) * vals[n - 1]
    if ORTHONORMAL:
        scale = np.array([1.0 / math.sqrt(math.factorial(k)) for k in range(degree)], dtype=float)
        vals = vals * scale
    return vals


def _basis_vals(x, degree, basis):
    if basis == uq.PolynomBasis.Legendre:
        return _legendre_vals(x, degree)
    if basis == uq.PolynomBasis.Hermite:
        return _hermite_vals(x, degree)
    raise ValueError("Unknown basis {}".format(basis))


def _eval_tt(cores, y, basis):
    val = np.asarray(cores[0][0, :, :], dtype=float)
    for dim, yi in enumerate(y, start=1):
        core = cores[dim]
        basis_vals = _basis_vals(yi, core.shape[1], basis)
        tmp = np.tensordot(core, basis_vals, axes=([1], [0]))
        val = val @ tmp
    val = np.squeeze(val)
    if np.ndim(val) == 0:
        return float(val)
    return val


def _relative_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def _make_measurements(func, M, Ns, basis, rng):
    measurements = uq.UQMeasurementSet()
    if basis == uq.PolynomBasis.Legendre:
        sampler = lambda: rng.uniform(-1.0, 1.0, size=M)
    elif basis == uq.PolynomBasis.Hermite:
        sampler = lambda: rng.normal(0.0, 1.0, size=M)
    else:
        raise ValueError("Unknown basis {}".format(basis))

    for _ in range(Ns):
        y = sampler()
        val = np.asarray(func(y), dtype=float)
        if val.ndim == 0:
            val = np.array([val], dtype=float)
        measurements.add(y, val)
    return measurements


def _reconstruct(func, M, Ns, poly_dim, basis, maxitr=200, targeteps=1e-6, init_rank=1, init_noise=1e-3):
    measurements = _make_measurements(func, M, Ns, basis, rng=np.random.default_rng(0))
    sample = np.asarray(func(np.zeros(M)), dtype=float)
    if sample.ndim == 0:
        sample = np.array([sample], dtype=float)
    dimensions = [int(sample.size)] + [poly_dim] * M
    res = uq.uq_ra_adf(
        measurements,
        basis,
        dimensions,
        targeteps=targeteps,
        maxitr=maxitr,
        device=torch.device("cpu"),
        dtype=torch.float64,
        init_rank=init_rank,
        init_noise=init_noise,
        orthonormal=ORTHONORMAL,
    )
    return res


def _grid_eval(func, tt, basis, grid):
    cores = [c.detach().cpu().numpy() for c in tt.cores]
    ref = np.zeros((grid.size, grid.size), dtype=float)
    pred = np.zeros_like(ref)
    for i, x in enumerate(grid):
        for j, y in enumerate(grid):
            ref[i, j] = func(np.array([x, y]))
            pred[i, j] = _eval_tt(cores, np.array([x, y]), basis)
    return ref, pred


def _save_heatmap(data, path, title):
    plt.figure(figsize=(5, 4))
    plt.imshow(data, origin="lower", extent=(-1, 1, -1, 1), aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _convergence_plot(xs, ys, path, xlabel):
    plt.figure(figsize=(5, 4))
    plt.plot(xs, ys, marker="o")
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel("Relative L2 error")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def _run_case(name, func, basis, out_dir, threshold):
    M = 2
    eval_rng = np.random.default_rng(42)
    eval_pts = [eval_rng.uniform(-1.0, 1.0, size=M) for _ in range(400)]

    def eval_error(tt_eval):
        cores = [c.detach().cpu().numpy() for c in tt_eval.cores]
        preds = np.array([_eval_tt(cores, y, basis) for y in eval_pts], dtype=float)
        refs = np.array([func(y) for y in eval_pts], dtype=float)
        return _relative_l2(preds, refs)

    config = None
    for init_rank in [1, 2, 4, 8]:
        for poly_dim in [3, 4, 5, 6, 8]:
            for Ns in [200, 400, 800, 1200]:
                tt_try = _reconstruct(
                    func,
                    M,
                    Ns,
                    poly_dim,
                    basis,
                    maxitr=500,
                    targeteps=1e-8,
                    init_rank=init_rank,
                    init_noise=1e-2,
                )
                err = eval_error(tt_try)
                if err < threshold:
                    config = (Ns, poly_dim, init_rank, err)
                    tt = tt_try
                    break
            if config is not None:
                break
        if config is not None:
            break

    if config is None:
        Ns, poly_dim, init_rank = 1200, 8, 8
        tt = _reconstruct(
            func,
            M,
            Ns,
            poly_dim,
            basis,
            maxitr=600,
            targeteps=1e-8,
            init_rank=init_rank,
            init_noise=1e-2,
        )
        err = eval_error(tt)
        config = (Ns, poly_dim, init_rank, err)

    Ns, poly_dim, init_rank, err = config
    print(
        "Selected config [{}]: Ns={}, poly_dim={}, init_rank={}, rel_l2={:.6f}".format(
            name, Ns, poly_dim, init_rank, err
        )
    )

    tt = _reconstruct(
        func, M, Ns, poly_dim, basis, maxitr=600, targeteps=1e-8, init_rank=init_rank, init_noise=1e-2
    )
    grid = np.linspace(-1.0, 1.0, 101)
    ref, pred = _grid_eval(func, tt, basis, grid)
    denom = max(float(np.max(np.abs(ref))), 1e-12)
    rel_err = np.abs(pred - ref) / denom

    _save_heatmap(ref, os.path.join(out_dir, f"{name}_reference.png"), "Reference")
    _save_heatmap(pred, os.path.join(out_dir, f"{name}_reconstruction.png"), "Reconstruction")
    _save_heatmap(rel_err, os.path.join(out_dir, f"{name}_relative_error.png"), "Relative Error")

    Ns_list = [40, 80, 160, 320, 640, 1000]
    errs = []
    for n in Ns_list:
        tt_n = _reconstruct(
            func, M, n, poly_dim, basis, maxitr=500, targeteps=1e-8, init_rank=init_rank, init_noise=1e-2
        )
        errs.append(eval_error(tt_n))
    _convergence_plot(Ns_list, errs, os.path.join(out_dir, f"{name}_convergence_measurements.png"), "Measurements")

    deg_list = [2, 3, 4, 5, 6, 8]
    errs = []
    for deg in deg_list:
        tt_deg = _reconstruct(
            func, M, Ns, deg, basis, maxitr=500, targeteps=1e-8, init_rank=init_rank, init_noise=1e-2
        )
        errs.append(eval_error(tt_deg))
    _convergence_plot(deg_list, errs, os.path.join(out_dir, f"{name}_convergence_poly_degree.png"), "Polynomial degree")

    rmax_list = [2, 4, 8, 12]
    errs = []
    for rmax in rmax_list:
        tt_r = tt.round(eps=1e-12, rmax=rmax)
        errs.append(eval_error(tt_r))
    _convergence_plot(rmax_list, errs, os.path.join(out_dir, f"{name}_convergence_rmax.png"), "Rank cap")


def _save_line_plot(x, ref, pred, path, title):
    plt.figure(figsize=(6, 3.5))
    plt.plot(x, ref, label="reference")
    plt.plot(x, pred, "--", label="reconstruction")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_error_plot(x, err, path, title):
    plt.figure(figsize=(6, 3.5))
    plt.plot(x, err, label="relative error")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("relative error")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_realizations(x, refs, preds, path, title):
    plt.figure(figsize=(6, 3.8))
    for i, (ref, pred) in enumerate(zip(refs, preds), start=1):
        plt.plot(x, ref, label=f"ref {i}")
        plt.plot(x, pred, "--", label=f"reco {i}")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _diffusive_operator(u, a, dx):
    du = np.zeros_like(u)
    du[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
    du[0] = (u[1] - u[0]) / dx
    du[-1] = (u[-1] - u[-2]) / dx
    flux = a * du
    dflux = np.zeros_like(u)
    dflux[1:-1] = (flux[2:] - flux[:-2]) / (2.0 * dx)
    dflux[0] = (flux[1] - flux[0]) / dx
    dflux[-1] = (flux[-1] - flux[-2]) / dx
    return -dflux


def _run_vector_case(name, func, basis, out_dir, threshold, M, x, y_plot, residual_fn=None, y_samples=None):
    eval_rng = np.random.default_rng(7)
    eval_pts = [eval_rng.uniform(-1.0, 1.0, size=M) for _ in range(300)]

    def eval_error(tt_eval):
        cores = [c.detach().cpu().numpy() for c in tt_eval.cores]
        preds = np.stack([_eval_tt(cores, y, basis) for y in eval_pts], axis=0)
        refs = np.stack([func(y) for y in eval_pts], axis=0)
        return _relative_l2(preds.ravel(), refs.ravel())

    config = None
    for init_rank in [1, 2, 4, 8]:
        for poly_dim in [3, 4, 5, 6, 8]:
            for Ns in [200, 400, 800, 1200]:
                tt_try = _reconstruct(
                    func,
                    M,
                    Ns,
                    poly_dim,
                    basis,
                    maxitr=500,
                    targeteps=1e-8,
                    init_rank=init_rank,
                    init_noise=1e-2,
                )
                err = eval_error(tt_try)
                if err < threshold:
                    config = (Ns, poly_dim, init_rank, err)
                    tt = tt_try
                    break
            if config is not None:
                break
        if config is not None:
            break

    if config is None:
        Ns, poly_dim, init_rank = 1200, 8, 8
        tt = _reconstruct(
            func,
            M,
            Ns,
            poly_dim,
            basis,
            maxitr=600,
            targeteps=1e-8,
            init_rank=init_rank,
            init_noise=1e-2,
        )
        err = eval_error(tt)
        config = (Ns, poly_dim, init_rank, err)

    Ns, poly_dim, init_rank, err = config
    print(
        "Selected config [{}]: Ns={}, poly_dim={}, init_rank={}, rel_l2={:.6f}".format(
            name, Ns, poly_dim, init_rank, err
        )
    )

    tt = _reconstruct(
        func, M, Ns, poly_dim, basis, maxitr=600, targeteps=1e-8, init_rank=init_rank, init_noise=1e-2
    )
    cores = [c.detach().cpu().numpy() for c in tt.cores]
    ref = func(y_plot)
    pred = _eval_tt(cores, y_plot, basis)
    denom = max(float(np.max(np.abs(ref))), 1e-12)
    rel_err = np.abs(pred - ref) / denom

    _save_line_plot(x, ref, pred, os.path.join(out_dir, f"{name}_line.png"), "Reference vs Reconstruction")
    _save_error_plot(x, rel_err, os.path.join(out_dir, f"{name}_relative_error.png"), "Relative Error")
    if residual_fn is not None:
        res_ref, rel_ref = residual_fn(ref, y_plot)
        res_pred, rel_pred = residual_fn(pred, y_plot)
        _save_error_plot(x, res_pred, os.path.join(out_dir, f"{name}_residual.png"), "PDE Residual")
        print(
            "Residual [{}]: rel_ref={:.3e} rel_pred={:.3e}".format(
                name, rel_ref, rel_pred
            )
        )
    if y_samples:
        ref_series = [func(y) for y in y_samples]
        pred_series = [_eval_tt(cores, y, basis) for y in y_samples]
        _save_realizations(
            x,
            ref_series,
            pred_series,
            os.path.join(out_dir, f"{name}_realizations.png"),
            "Sample Realizations",
        )

    Ns_list = [40, 80, 160, 320, 640, 1000]
    errs = []
    for n in Ns_list:
        tt_n = _reconstruct(
            func, M, n, poly_dim, basis, maxitr=500, targeteps=1e-8, init_rank=init_rank, init_noise=1e-2
        )
        errs.append(eval_error(tt_n))
    _convergence_plot(Ns_list, errs, os.path.join(out_dir, f"{name}_convergence_measurements.png"), "Measurements")

    deg_list = [2, 3, 4, 5, 6, 8]
    errs = []
    for deg in deg_list:
        tt_deg = _reconstruct(
            func, M, Ns, deg, basis, maxitr=500, targeteps=1e-8, init_rank=init_rank, init_noise=1e-2
        )
        errs.append(eval_error(tt_deg))
    _convergence_plot(deg_list, errs, os.path.join(out_dir, f"{name}_convergence_poly_degree.png"), "Polynomial degree")

    rmax_list = [2, 4, 8, 12]
    errs = []
    for rmax in rmax_list:
        tt_r = tt.round(eps=1e-12, rmax=rmax)
        errs.append(eval_error(tt_r))
    _convergence_plot(rmax_list, errs, os.path.join(out_dir, f"{name}_convergence_rmax.png"), "Rank cap")


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(out_dir, exist_ok=True)

    basis = uq.PolynomBasis.Legendre

    def f_hard(y):
        return np.sin(2.0 * y[0]) + 0.5 * np.cos(3.0 * y[1]) + 0.2 * y[0] * y[1]

    _run_case("uq_adf_hard", f_hard, basis, out_dir, threshold=1e-2)

    x = np.linspace(0.0, 1.0, 128)

    def _kl_basis(xgrid, modes, ell=0.2):
        ks = np.arange(1, modes + 1)
        lambdas = 1.0 / (1.0 + (np.pi * ks * ell) ** 2)
        phi = np.sqrt(2.0) * np.sin(np.pi * ks[:, None] * xgrid[None, :])
        return lambdas, phi

    modes = 3
    lambdas, phi = _kl_basis(x, modes)

    def a_kl(y):
        coeff = np.sum(np.sqrt(lambdas)[:, None] * phi * y[:, None], axis=0)
        return 1.0 + 0.2 * coeff

    def u_kl(y):
        coeff = np.sum(np.sqrt(lambdas)[:, None] * phi * y[:, None], axis=0)
        base = np.sin(np.pi * x)
        return base * (1.0 + 0.15 * coeff) + 0.05 * np.sin(2.0 * np.pi * x) * coeff

    dx = float(x[1] - x[0])

    def _kl_rhs(y):
        return _diffusive_operator(u_kl(y), a_kl(y), dx)

    def _kl_residual(u_vec, y):
        res = _diffusive_operator(u_vec, a_kl(y), dx) - _kl_rhs(y)
        denom = np.linalg.norm(_kl_rhs(y))
        rel = float(np.linalg.norm(res) / denom) if denom > 0 else float(np.linalg.norm(res))
        return res, rel

    rng = np.random.default_rng(123)
    y_plot = np.zeros(modes)
    y_samples = [rng.uniform(-1.0, 1.0, size=modes) for _ in range(3)]
    _run_vector_case(
        "uq_adf_kl",
        u_kl,
        basis,
        out_dir,
        threshold=1e-2,
        M=modes,
        x=x,
        y_plot=y_plot,
        residual_fn=_kl_residual,
        y_samples=y_samples,
    )

    c0 = np.sin(np.pi * x)
    c10 = 0.5 * np.cos(2.0 * np.pi * x)
    c01 = -0.3 * np.sin(2.0 * np.pi * x)
    c20 = 0.2 * np.cos(np.pi * x)

    def f_pc(y):
        y1, y2 = y
        p10 = y1
        p01 = y2
        p20 = 0.5 * (3.0 * y1 ** 2 - 1.0)
        return c0 + c10 * p10 + c01 * p01 + c20 * p20

    rng = np.random.default_rng(456)
    y_plot = np.array([0.2, -0.4])
    y_samples = [rng.uniform(-1.0, 1.0, size=2) for _ in range(3)]
    _run_vector_case(
        "uq_adf_pc",
        f_pc,
        basis,
        out_dir,
        threshold=1e-2,
        M=2,
        x=x,
        y_plot=y_plot,
        y_samples=y_samples,
    )


if __name__ == "__main__":
    main()
