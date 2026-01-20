# XERUS TorchTT UQ-ADF

This repo contains legacy Xerus research artifacts (notebooks/figures) plus a
Python reimplementation of the Xerus UQ-ADF routine built on `torchTT`. The
active code lives in `vmc_reconstruction/`, and `torchTT` is tracked as a git
submodule.

## Quick Start
- Initialize the submodule:
  - `git submodule update --init --recursive`
- Create a venv and install dependencies:
  - `python -m venv venv`
  - `./venv/bin/pip install -r vmc_reconstruction/requirements.txt`
  - `./venv/bin/pip install scikit-fem scipy matplotlib`

## Examples
- 2D Darcy reconstruction with convergence plots:
  - `./venv/bin/python examples/uq_adf_darcy_2d.py`
  - Fast mode (quick diagnostics): `./venv/bin/python examples/uq_adf_darcy_2d.py --fast`
- Convergence sweeps and heatmaps:
  - `./venv/bin/python examples/uq_adf_plots.py`

Plots are written to `examples/plots/` (gitignored).

## Tests
- `./venv/bin/python -m pytest -q tests`

Note: `tests/test_uq_adf_skfem.py` is intentionally slow because it solves a
fine-mesh Darcy reference problem.
