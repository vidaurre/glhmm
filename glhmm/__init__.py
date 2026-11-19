"""Gaussian-Linear Hidden Markov Model toolbox."""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("glhmm")
except PackageNotFoundError:  # source checkout without an installed distribution
    __version__ = "0.0.0.dev0"

__all__ = [
    "auxiliary",
    "glhmm",
    "graphics",
    "io",
    "palm_functions",
    "prediction",
    "preproc",
    "spectral",
    "statistics",
    "utils",
]
