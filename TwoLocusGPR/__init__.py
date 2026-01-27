"""Top-level package for TwoLocusGPR."""

from importlib import metadata

try:
    __version__ = metadata.version("TwoLocusGPR")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from .GPR import GPR
from . import Fit_MSD, MSD_functions, Posterior_analysis, utils

__all__ = [
    "GPR",
    "Fit_MSD",
    "MSD_functions",
    "Posterior_analysis",
    "utils",
]
