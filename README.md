# TwoLocusGPR

TwoLocusGPR provides Gaussian-process regression utilities for two-locus tracking data. It bundles
JAX-powered samplers, Gaussian-process marginal likelihoods, MSD-fitting routines, posterior geometry
helpers, and example notebooks for downstream analysis.

## Installation

```bash
pip install .
```

## Testing

```bash
python -m pytest
```

## Project Structure

- `TwoLocusGPR/`: Python package with Gaussian-process utilities.
- `tests/`: Pytest suite exercising the main public API.
- `examples/`: Interactive notebooks that demonstrate the workflows used in the associated research.
