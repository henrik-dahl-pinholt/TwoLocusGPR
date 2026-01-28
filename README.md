# TwoLocusGPR

TwoLocusGPR provides Gaussian-process regression utilities for two-locus tracking data. It bundles
JAX-powered samplers, Gaussian-process marginal likelihoods, MSD-fitting routines, posterior geometry
helpers, and example notebooks for downstream analysis.

## Installation

```bash
pip install .
```
Note that for GPU functionality, you need [jax](https://docs.jax.dev/en/latest/installation.html#conda-community-supported) installed correctly, which is best set up before installing the package. 

## Notebooks
The workings of the module can be seen from the example notebooks:
- [Examples/Anomaly_detection.ipynb](Examples/Anomaly_detection.ipynb): Detecting anomalies in time-series.
- [Examples/fit_MSD.ipynb](Examples/fit_MSD.ipynb): Fitting MSDs.
- [Examples/Identifying_Pol2_loadings.ipynb](Examples/Identifying_Pol2_loadings.ipynb): Event-level polymerase inference.
- [Examples/Posterior_predictions.ipynb](Examples/Posterior_predictions.ipynb): Predicting posteriors.
