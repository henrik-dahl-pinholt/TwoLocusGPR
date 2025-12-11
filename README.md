# TwoLocusGPR

Gaussian process regression utilities for multi-dimensional particle-tracking data, with covariance parameterized by mean-squared-displacement (MSD) models. Includes tools to fit MSD parameters, draw prior/posterior samples, score likelihoods, and run anomaly-detection style post-processing.

## Installation

Prereqs: Python 3.9+.

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU: choose the matching CUDA/ROCM JAX wheel from https://github.com/google/jax#pip-installation and install it before or after the requirements.

## Repository Layout

- `GPR.py`: Core Gaussian process regressor using JAX; builds covariances from an MSD function, evaluates log-likelihoods, predicts means/variances, samples priors/posteriors, and offers a simple Metropolis-Hastings sampler.
- `utils.py`: GP numerics and MSD helpers (Cholesky-safe covariance prep with NaN masking, posterior mean/covariance utilities, Gaussian samplers, MSD estimators).
- `MSD_functions.py`: MSD model definitions (Rouse, saturating/soft-min variant, softmin helper).
- `Fit_MSD.py`: End-to-end MSD fitting pipeline with bootstrap covariance estimation, multi-start optimization, and optional profile-likelihood CIs.
- `Posterior_analysis.py`: Post-processing utilities (probability-in-sphere estimates, radius percentiles, block-based anomaly detection on tracks).
- `examples/`: Notebooks and sample data illustrating fitting and posterior prediction workflows.
- `tests/`: Pytest-based sanity checks for GP predictions and parameter recovery.
- `pytest.ini`: Pytest configuration (if provided).

## Quickstart

```python
import numpy as np
import GPR, MSD_functions

# observation times and model
ndims = 3
times = np.linspace(0, 6.0 * 60, 181)  # minutes
regressor = GPR.GPR(times, MSD_functions.Rouse_MSD, ndims)

# parameter vector packs per-dim MSD params followed by per-dim noise
Gamma, J = 2.4e3 * np.sqrt(60) / 3, (357.0) ** 2 / 3
noise = np.array([48.0, 47.0, 120.0])
theta = np.array([Gamma, J] * ndims + noise.tolist())

# draw prior samples and add noise
true_samples, noisy_samples = regressor.sample_prior(theta, num_samples=10, seed=0)

# predict on a finer grid
fine_times = np.linspace(times.min(), times.max(), 500)
mean_pred, var_pred = regressor.Predict(theta, fine_times, noisy_samples)
```

Shapes: tracks are `(n_tracks, n_timepoints, n_dims)`, single tracks `(n_timepoints, n_dims)`, time arrays 1D. Parameters are packed as `[per-dim MSD params..., per-dim noise]` with per-dim MSD params tiled `ndims` times.

## Running Tests

From the repo root:

```bash
pytest
```

## Notes

- Some routines assume finite noise and handle `NaN` observations by inflating diagonal noise and masking the data.
- Use `jax.jit`-compatible pure functions for custom MSD models (signature `msd(t, params)` where `t` and `params` are JAX arrays).
- No license file is present; add one if distribution is intended.
