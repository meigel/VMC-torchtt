# Repository Guidelines

## Project Structure & Module Organization
- The repo root is mostly research artifacts: `*.ipynb` notebooks and `*.svg` diagrams.
- `vmc_reconstruction/` contains the active Python code and scripts.
- Key subfolders:
  - `vmc_reconstruction/problem/`: PDE/problem definitions loaded via `info.json`.
  - `vmc_reconstruction/field/`: field/basis helpers.
  - `vmc_reconstruction/mesh/`: mesh inputs (`*.xml`) and mesh-specific requirements.
  - `vmc_reconstruction/convert/`: conversion utilities to Xerus-format files.
  - `vmc_reconstruction/plot/` and `vmc_reconstruction/reco_plots/`: plotting utilities and C++ plotting helpers.
  - `vmc_reconstruction/results/`, `vmc_reconstruction/plots/`: generated outputs and experiment artifacts.
  - `vmc_reconstruction/test_series/`: scripted reconstruction sweeps and example configurations.

## Build, Test, and Development Commands
- `pip install -r vmc_reconstruction/requirements.txt` installs Python dependencies (expects FEniCS/PETSc libs to be present).
- `docker build -f vmc_reconstruction/Dockerfile .` builds a full Xerus + FEniCS environment (slow, downloads dependencies).
- `python vmc_reconstruction/run_mc.py <PATH> --dump=10000` samples and writes `samples/` + `moments.npz` under `<PATH>`.
- `python vmc_reconstruction/dump_stiffness.py <PATH>` writes Cholesky data needed for orthogonalization.
- `python vmc_reconstruction/orthogonalize_samples.py <PATH>` produces orthogonalized samples.
- `python vmc_reconstruction/reco.py <PATH> --ortho` reconstructs a TT tensor (omit `--ortho` for raw samples).
- `python vmc_reconstruction/convert/samples_to_xerus_format.py <PATH> <OUT>` exports samples to Xerus format.

## Coding Style & Naming Conventions
- Python scripts use 4-space indentation and `snake_case` for functions/variables; constants are uppercase.
- Keep `# -*- coding: utf-8 -*-` and `from __future__ import ...` headers where present.
- Data layout follows `info.json`, with outputs like `samples/`, `moments.npz`, and `results/<N>/reconstruction.xrs`.

## Testing Guidelines
- There is no automated unit-test framework in this repo.
- Validate changes by running the primary pipelines (`run_mc.py`, `reco.py`) and inspecting generated `moments.npz`/plots.
- `vmc_reconstruction/test_series/reconstruction.py` runs reconstruction sweeps against folders in `vmc_reconstruction/test_series/*/`.

## Commit & Pull Request Guidelines
- No Git history is available in this checkout; use short, imperative messages (e.g., "Add orthogonalized reconstruction plot").
- PRs should describe the experiment/config change, mention updated `info.json` keys, and attach key plots or summary stats.

## Configuration & Data Notes
- Output folders (`results/`, `plots/`) can be large; document when regenerating them.
- For parallel runs, control thread count (e.g., `OMP_NUM_THREADS=1`) to avoid oversubscription.
