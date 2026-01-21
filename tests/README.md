# Tests

These tests cover the torchTT-based UQ-ADF reconstruction and lightweight tensor-network helpers.

Run all tests:
- `./venv/bin/python -m pytest -q tests`

Notable suites:
- `tests/test_uq_adf.py`: synthetic UQ-ADF regression checks (Legendre/Hermite).
- `tests/test_uq_adf_skfem.py`: Darcy UQ-ADF with FEM; slow reference runs require
  `RUN_SLOW_DARCY=1`.
- `tests/test_torchtt_algorithms.py`: AMEn, DMRG matvec, cross approximation, QTT round-trip.
- `tests/test_quantum.py`: MPS/MPO helpers, two-site DMRG, TDVP (dense/Krylov).
