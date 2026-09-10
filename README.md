# Simulation Code for Surrogate MPC with Mobile Actuators

This folder is a curated reproducibility package for the two numerical studies in the manuscript:

1. two-dimensional marine-pollutant remediation; and
2. one-dimensional metal-rod cooling.

It contains the minimum connected code paths needed to inspect the PDE solvers, synthetic-data generation, surrogate architectures, training and evaluation procedures, actuator dynamics, MPC objectives, constraint handling, and baseline comparisons. It intentionally omits trained weights, cached trajectories, rendered figures, temporary files, and unrelated development material.

## File map

### Two-dimensional experiment

- `pollution_core.py`: physical/grid parameters, PDE stepping, actuator dynamics, random trajectory generation, and transition-dataset construction.
- `stateful_surrogate.py`: state-aware convolutional surrogate and data/physics/boundary/mass losses.
- `pinn.py`: model training and checkpoint generation.
- `pinn2.py`: unseen-trajectory autoregressive evaluation.
- `one.py`: mobile/fixed actuator MPC, numerical-PDE and surrogate prediction modes, constraints, timing, and result plots.
- `plot_style.py`: plotting defaults shared by both experiments.

### One-dimensional experiment

- `rod_cooling_1d/core.py`: rod PDE, actuator ODE, constraints, and synthetic-data generation.
- `rod_cooling_1d/model.py`: one-dimensional state-aware convolutional surrogate.
- `rod_cooling_1d/train.py`: surrogate training.
- `rod_cooling_1d/evaluate.py`: unseen-trajectory autoregressive evaluation.
- `rod_cooling_1d/control.py`: mobile, fixed, numerical-PDE, surrogate, and SVD-RBFNN MPC experiments.
- `rod_cooling_1d/svd_rbf_baseline.py`: SVD-RBFNN baseline implementation.
- `rod_cooling_1d/train_svd_rbf.py`: SVD-RBFNN baseline training entry point.

### Documentation and checks

- `RELEASE_SCOPE.md`: exact inclusion/exclusion decisions.
- `reference_metrics/`: small JSON outputs supplied only as sanity-check references.
- `smoke_test.py`: fast import, shape, PDE-step, network-forward, and boundary-condition checks.

## Environment

Python 3.11 is recommended. Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the appropriate CPU or CUDA build of PyTorch for the target machine if the default wheel is unsuitable. The exact environment used for the package check is recorded in `environment-tested.txt`.

## Quick package check

```bash
python smoke_test.py
```

This check does not train models or run a full MPC experiment.

## Two-dimensional workflow

Run these commands from the package root.

Train the state-aware physics-informed surrogate:

```bash
python pinn.py
```

This generates `state_aware_pinn_best.pth` and `state_aware_pinn_final.pth`. These generated checkpoints are ignored by Git and are not included in the release package.

Evaluate autoregressive prediction on unseen trajectories:

```bash
python pinn2.py --model-path state_aware_pinn_best.pth --num-test-trajectories 20
```

Run the surrogate-based mobile/fixed MPC comparison:

```bash
python one.py --model-path state_aware_pinn_best.pth --prediction-mode surrogate --controller-mode compare --output-dir outputs/2d_control
```

Run the same controller with the numerical PDE predictor:

```bash
python one.py --prediction-mode pde --controller-mode compare --output-dir outputs/2d_pde_control
```

Relevant command-line settings include `--horizon`, `--max-iter`, `--control-steps`, `--control-stride`, `--surrogate-iters`, and `--surrogate-lr`. Use `python one.py --help` for the complete list.

## One-dimensional workflow

Train the state-aware surrogate:

```bash
python -m rod_cooling_1d.train
```

Evaluate the trained surrogate:

```bash
python -m rod_cooling_1d.evaluate
```

Train the SVD-RBFNN comparison model:

```bash
python -m rod_cooling_1d.train_svd_rbf
```

Run the surrogate-based mobile/fixed/SVD-RBFNN comparison:

```bash
python -m rod_cooling_1d.control --prediction-mode surrogate --controller-mode compare
```

Run the numerical-PDE comparison:

```bash
python -m rod_cooling_1d.control --prediction-mode pde --controller-mode compare --output-dir outputs/rod_pde_control
```

Only the first optimized control/force action is applied at each sampling instant before the horizon is re-optimized.

## Reproducibility notes

- Training and evaluation data are generated synthetically by the included PDE solvers; no private experimental dataset is required.
- Random seeds and default hyperparameters are encoded in the scripts and saved with generated model/config outputs.
- CUDA is optional. Numerical timing depends on hardware, PyTorch build, and solver configuration.
- The reference JSON files are provided for coarse sanity checking, not as a substitute for rerunning the simulations.
- Full experiments can be computationally expensive. Reduce trajectory counts, epochs, control steps, or optimizer iterations for development checks only; do not compare reduced runs directly with manuscript results.
