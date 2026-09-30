# API Reference

Full API documentation for all GLHMM modules.

| Module | What it contains |
|---|---|
| `glhmm.glhmm` | The main HMM class. Create, train, decode, and inspect models. |
| `glhmm.io` | Load and save HMMs and data files. |
| `glhmm.preproc` | Preprocess time series data before training. |
| `glhmm.auxiliary` | Helper functions for building `indices`, formatting data, and other utilities. |
| `glhmm.utils` | Extract summary measures from trained models: fractional occupancy (`get_FO`), life times (`get_life_times`), switching rate, state onsets, and stability training tools for choosing K. |
| `glhmm.graphics` | Plotting functions for visualising states, covariances, and training diagnostics. |
| `glhmm.prediction` | Predict behavioural variables from HMM outputs (fractional occupancy, lifetimes, etc.). |
| `glhmm.statistics` | Statistical tests (`test_across_subjects`, `test_across_trials`, `test_across_sessions_within_subject`, `test_across_state_visits`) and index-building helpers (`get_indices_from_list`, `get_indices_timestamp`). |
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
