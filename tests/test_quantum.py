import os
import sys

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

from torchtt.quantum import MPS, MPO, apply_mpo, expectation_value, dmrg_ground_state, tdvp_evolve


def test_mpo_identity_apply():
    dims = [2, 2]
    psi = MPS.random(dims, [1, 2, 1], dtype=torch.float64).normalize()
    mpo = MPO(tntt.eye(dims, dtype=torch.float64))
    out = apply_mpo(mpo, psi, method="fast", eps=1e-12)
    diff = torch.linalg.norm(out.tt.full() - psi.tt.full())
    assert diff < 1e-8
    energy = expectation_value(mpo, psi)
    assert torch.abs(energy - 1.0) < 1e-8


def test_dmrg_ground_state_smoke():
    dims = [2, 2]
    mpo = MPO(tntt.eye(dims, dtype=torch.float64))
    psi0 = MPS.random(dims, [1, 2, 1], dtype=torch.float64)
    psi, info = dmrg_ground_state(mpo, psi0, max_iter=2, step_size=0.1)
    assert torch.isfinite(info["energy"]).all()
    assert psi.N == dims
    psi2, info2 = dmrg_ground_state(mpo, psi0, method="two_site", sweeps=1, local_solver="lobpcg", local_max_iter=10)
    assert torch.isfinite(info2["energy"]).all()
    assert psi2.N == dims


def test_tdvp_evolve_zero_step():
    dims = [2, 2]
    mpo = MPO(tntt.eye(dims, dtype=torch.float64))
    psi0 = MPS.random(dims, [1, 2, 1], dtype=torch.float64)
    out = tdvp_evolve(mpo, psi0, dt=0.0, steps=2, coeff=-1.0, eps=1e-12)
    diff = torch.linalg.norm(out.tt.full() - psi0.tt.full())
    assert diff < 1e-8
    out2 = tdvp_evolve(mpo, psi0, dt=0.0, steps=1, coeff=-1.0, scheme="two_site", sweeps=1)
    diff2 = torch.linalg.norm(out2.tt.full() - psi0.tt.full())
    assert diff2 < 1e-8
    out3 = tdvp_evolve(
        mpo,
        psi0,
        dt=0.0,
        steps=1,
        coeff=-1.0,
        scheme="two_site",
        sweeps=1,
        local_expm="krylov",
        krylov_dim=6,
    )
    diff3 = torch.linalg.norm(out3.tt.full() - psi0.tt.full())
    assert diff3 < 1e-8
