import numpy as np
import torch
import numpy.polynomial.legendre as leg

from vmc_reconstruction import uq_adf_torchtt as uq


def _legendre_basis_vals(x, degree):
    return np.array([leg.legval(x, [0] * k + [1]) for k in range(degree)], dtype=float)


def _hermite_basis_vals(x, degree):
    vals = np.zeros(degree, dtype=float)
    if degree == 0:
        return vals
    vals[0] = 1.0
    if degree == 1:
        return vals
    vals[1] = x
    for n in range(1, degree - 1):
        vals[n + 1] = x * vals[n] - float(n) * vals[n - 1]
    return vals


def _basis_vals(x, degree, basis):
    if basis == uq.PolynomBasis.Legendre:
        return _legendre_basis_vals(x, degree)
    if basis == uq.PolynomBasis.Hermite:
        return _hermite_basis_vals(x, degree)
    raise ValueError("Unknown basis {}".format(basis))


def _eval_tt_scalar(cores, y, basis):
    # Evaluates a scalar-valued TT with polynomial basis in stochastic dims.
    # Assumes physical dimension N=1.
    if torch.is_tensor(cores[0]):
        cores = [c.detach().cpu().numpy() for c in cores]
    val = np.asarray(cores[0][0, 0, :], dtype=float)
    for dim, yi in enumerate(y, start=1):
        core = cores[dim]
        basis_vals = _basis_vals(yi, core.shape[1], basis)
        tmp = np.tensordot(core, basis_vals, axes=([1], [0]))  # (r_left, r_right)
        val = val @ tmp
    return float(np.squeeze(val))


def test_uq_adf_constant_legendre():
    rng = np.random.default_rng(0)
    M = 2
    poly_dim = 3
    Ns = 60
    constant = 2.5

    measurements = uq.UQMeasurementSet()
    for _ in range(Ns):
        y = rng.uniform(-1.0, 1.0, size=M)
        measurements.add(y, np.array([constant], dtype=float))

    dimensions = [1] + [poly_dim] * M
    res = uq.uq_ra_adf(
        measurements,
        uq.PolynomBasis.Legendre,
        dimensions,
        targeteps=1e-6,
        maxitr=200,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    cores = [c.detach().cpu().numpy() for c in res.cores]
    for _ in range(10):
        y = rng.uniform(-1.0, 1.0, size=M)
        pred = _eval_tt_scalar(cores, y, uq.PolynomBasis.Legendre)
        assert abs(pred - constant) < 2e-2


def test_uq_adf_linear_legendre():
    rng = np.random.default_rng(1)
    M = 2
    poly_dim = 3
    Ns = 60
    a0, a1, a2 = 1.2, 0.3, -0.7

    def f(y):
        return a0 + a1 * y[0] + a2 * y[1]

    measurements = uq.UQMeasurementSet()
    for _ in range(Ns):
        y = rng.uniform(-1.0, 1.0, size=M)
        measurements.add(y, np.array([f(y)], dtype=float))

    dimensions = [1] + [poly_dim] * M
    res = uq.uq_ra_adf(
        measurements,
        uq.PolynomBasis.Legendre,
        dimensions,
        targeteps=1e-6,
        maxitr=200,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    cores = [c.detach().cpu().numpy() for c in res.cores]
    errs = []
    for _ in range(30):
        y = rng.uniform(-1.0, 1.0, size=M)
        pred = _eval_tt_scalar(cores, y, uq.PolynomBasis.Legendre)
        errs.append(abs(pred - f(y)))
    assert float(np.mean(errs)) < 5e-2


def test_uq_adf_linear_hermite():
    rng = np.random.default_rng(2)
    M = 2
    poly_dim = 3
    Ns = 80
    a0, a1, a2 = -0.4, 0.9, 0.2

    def f(y):
        return a0 + a1 * y[0] + a2 * y[1]

    measurements = uq.UQMeasurementSet()
    for _ in range(Ns):
        y = rng.normal(0.0, 1.0, size=M)
        measurements.add(y, np.array([f(y)], dtype=float))

    dimensions = [1] + [poly_dim] * M
    res = uq.uq_ra_adf(
        measurements,
        uq.PolynomBasis.Hermite,
        dimensions,
        targeteps=1e-6,
        maxitr=200,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    cores = [c.detach().cpu().numpy() for c in res.cores]
    eval_samples = [rng.normal(0.0, 1.0, size=M) for _ in range(40)]
    mean_val = float(np.mean([f(y) for y in eval_samples]))
    errs = []
    baseline_errs = []
    for y in eval_samples:
        pred = _eval_tt_scalar(cores, y, uq.PolynomBasis.Hermite)
        errs.append(abs(pred - f(y)))
        baseline_errs.append(abs(mean_val - f(y)))
    assert float(np.mean(errs)) < 0.2
    assert float(np.mean(errs)) < 0.35 * float(np.mean(baseline_errs))


def test_uq_adf_noisy_measurements():
    rng = np.random.default_rng(3)
    M = 2
    poly_dim = 3
    Ns = 120
    noise_sigma = 0.05
    a0, a1, a2 = 0.7, -0.2, 0.5

    def f(y):
        return a0 + a1 * y[0] + a2 * y[1]

    measurements = uq.UQMeasurementSet()
    for _ in range(Ns):
        y = rng.uniform(-1.0, 1.0, size=M)
        noise = rng.normal(0.0, noise_sigma)
        measurements.add(y, np.array([f(y) + noise], dtype=float))

    dimensions = [1] + [poly_dim] * M
    res = uq.uq_ra_adf(
        measurements,
        uq.PolynomBasis.Legendre,
        dimensions,
        targeteps=1e-6,
        maxitr=250,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    cores = [c.detach().cpu().numpy() for c in res.cores]
    errs = []
    for _ in range(20):
        y = rng.uniform(-1.0, 1.0, size=M)
        pred = _eval_tt_scalar(cores, y, uq.PolynomBasis.Legendre)
        errs.append(abs(pred - f(y)))
    assert float(np.mean(errs)) < 0.08


def test_uq_adf_initial_random_vectors_regression():
    rng = np.random.default_rng(4)
    M = 2
    poly_dim = 2
    measurements = uq.UQMeasurementSet()
    # Solutions define the mean used in the initial guess.
    for val in [1.0, 2.0, 3.0]:
        measurements.add(rng.uniform(-1.0, 1.0, size=M), np.array([val], dtype=float))
    # Initial solutions correspond to linear terms in the initial guess.
    measurements.add_initial(np.zeros(M), np.array([3.0], dtype=float))
    measurements.add_initial(np.zeros(M), np.array([1.0], dtype=float))

    dimensions = [1] + [poly_dim] * M
    cores = uq._initial_guess_with_linear_terms(
        measurements, dimensions, dtype=torch.float64, device=torch.device("cpu")
    )

    for _ in range(10):
        y = rng.uniform(-1.0, 1.0, size=M)
        pred = _eval_tt_scalar(cores, y, uq.PolynomBasis.Legendre)
        expected = 2.0 + y[1] - y[0]
        assert abs(pred - expected) < 1e-6
