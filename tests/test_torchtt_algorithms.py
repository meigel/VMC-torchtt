import os
import sys

import numpy as np
import pytest
import torch


def _import_torchtt():
    try:
        import torchtt as tntt
        return tntt
    except Exception:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        torchtt_path = os.path.join(repo_root, "torchTT")
        if torchtt_path not in sys.path:
            sys.path.insert(0, torchtt_path)
        try:
            import torchtt as tntt
            return tntt
        except Exception as exc:
            pytest.skip(f"torchtt unavailable: {exc}", allow_module_level=True)


tntt = _import_torchtt()


def test_amen_mm_small():
    torch.manual_seed(0)
    A = tntt.random([(3, 3)] * 3, [1, 2, 2, 1], dtype=torch.float64)
    B = tntt.random([(3, 3)] * 3, [1, 2, 2, 1], dtype=torch.float64)
    ref = (A @ B).round(1e-12)
    out = tntt.amen_mm(A, B, kickrank=4, verbose=False)
    rel = float((out - ref).norm() / ref.norm())
    assert rel < 1e-6


def test_dmrg_fast_matvec():
    torch.manual_seed(1)
    A = tntt.random([(2, 2)] * 3, [1, 2, 2, 1], dtype=torch.float64)
    x = tntt.random([2] * 3, [1, 2, 2, 1], dtype=torch.float64)
    ref = (A @ x).round(1e-12)
    out = A.fast_matvec(x)
    rel = float((out - ref).norm() / ref.norm())
    assert rel < 1e-6


def test_dmrg_cross_approximation():
    torch.manual_seed(2)
    N = [5, 5, 4]

    def func(I):
        I = I.to(dtype=torch.float64)
        return 1.0 / (1.0 + torch.sum(I, dim=1))

    tt = tntt.interpolate.dmrg_cross(func, N, eps=1e-8, nswp=6, kick=2, dtype=torch.float64, rmax=10)
    full = tt.full()

    grid = np.stack(np.meshgrid(*[np.arange(n) for n in N], indexing="ij"), axis=-1)
    vals = 1.0 / (1.0 + grid.sum(axis=-1))
    vals = torch.tensor(vals, dtype=torch.float64)

    rel = float(torch.linalg.norm(full - vals) / torch.linalg.norm(vals))
    assert rel < 1e-6


def test_qtt_roundtrip():
    torch.manual_seed(3)
    x = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    tt = tntt.TT(x)
    qtt = tt.to_qtt(mode_size=2)
    back = qtt.qtt_to_tens(tt.N)
    rel = float(torch.linalg.norm(back.full() - x) / torch.linalg.norm(x))
    assert rel < 1e-12
