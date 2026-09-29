# API Reference

Full API documentation for all GLHMM modules.

| Module | What it contains |
|---|---|
| `glhmm.glhmm` | The main HMM class. Create, train, decode, and inspect models. |
| `glhmm.io` | Load and save HMMs and data files. |
| `glhmm.preproc` | Preprocess time series data before training. |
| `glhmm.auxiliary` | Helper functions for building `indices`, formatting data, and other utilities. |
| `glhmm.utils` | General utilities used internally by the toolbox. |
| `glhmm.graphics` | Plotting functions for visualising states, covariances, and training diagnostics. |
| `glhmm.prediction` | Predict behavioural variables from HMM outputs (fractional occupancy, lifetimes, etc.). |
| `glhmm.statistics` | Permutation tests for group differences and other statistical comparisons. |
| `glhmm.spectral` | Spectral analysis of HMM states (used mainly with TDE-HMMs). |

---

```{toctree}
:maxdepth: 1
:hidden:

glhmm
io
preproc
auxiliary
utils
graphics
prediction
statistics
palm_functions
spectral
```
