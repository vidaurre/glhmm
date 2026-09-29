---
html_theme.sidebar_secondary.remove: true
---

# GLHMM — Gaussian-Linear Hidden Markov Models

```{image} ../logo2_full.png
:alt: GLHMM logo
:width: 300px
:align: center
```

<br>

[![Documentation Status](https://readthedocs.org/projects/glhmm/badge/?version=latest)](https://glhmm.readthedocs.io/en/latest/?badge=latest)

GLHMM is a Python toolbox for fitting Hidden Markov Models (HMMs) to time series data, with a focus on neuroscience. It can identify recurring brain states from fMRI, MEG, EEG, or ECoG recordings, and relate those states to behaviour or other variables.

---

## Where to start

::::{grid} 2
:gutter: 3

:::{grid-item-card} Getting Started
:link: getting_started
:link-type: doc

Install the toolbox, understand the data format, and train your first model in a few lines of code.
:::

:::{grid-item-card} How it works
:link: key_concepts
:link-type: doc

Understand the main parameters — K, covtype, δ, γ, and free energy — before diving into the code.
:::

:::{grid-item-card} Examples
:link: notebooks/tutorial
:link-type: doc

Worked examples for Gaussian HMMs, GLHMMs, TDE-HMMs, and more.
:::

:::{grid-item-card} Large datasets and GPU
:link: training_options
:link-type: doc

Stochastic training for datasets that do not fit in memory, and GPU acceleration to speed up training.
:::

::::

---

## What can GLHMM do?

- **Identify brain states**: find recurring patterns of activity or connectivity that the brain cycles through over time
- **Work with multiple data modalities**: fMRI, MEG, EEG, ECoG, and more
- **Relate brain dynamics to behaviour**: predict cognitive scores, classify clinical groups, or test for statistical relationships
- **Scale to large datasets**: stochastic training processes data file by file without loading everything into memory
- **GUI available**: a code-free interface is available at [github.com/Nick7900/glhmm_protocols](https://github.com/Nick7900/glhmm_protocols)

---

## Citing GLHMM

If you use GLHMM in your research, please cite the toolbox paper:

> Vidaurre et al. (2023). *GLHMM: A Python toolbox for Generalised Linear Hidden Markov Modelling.* Imaging Neuroscience. [https://doi.org/10.1162/imag_a_00460](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00460/127499)

If you use the statistical testing protocols, please also cite:

> Larsen, N.Y., Paulsen, L.B., Ahrends, C., Winkler, A.M. & Vidaurre, D. (2026). *A comprehensive framework for statistical testing of brain dynamics.* Nature Protocols, 21, 3148–3179. [https://doi.org/10.1038/s41596-025-01300-2](https://www.nature.com/articles/s41596-025-01300-2)

The protocols and a code-free GUI for the statistical tests are available at [github.com/Nick7900/glhmm_protocols](https://github.com/Nick7900/glhmm_protocols).

---

```{toctree}
:caption: Getting Started
:maxdepth: 1
:hidden:

getting_started
key_concepts
notebooks/Preprocessing
notebooks/HMM_state_selection_standard
training_options
troubleshooting
```

```{toctree}
:caption: Examples
:maxdepth: 1
:hidden:

notebooks/tutorial
notebooks/GaussianHMM_example
notebooks/GLHMM_example
notebooks/HMM-TDE_vs_HMM-MAR_example
```

```{toctree}
:caption: Prediction
:maxdepth: 1
:hidden:

notebooks/Prediction_tutorial
notebooks/Prediction_split
```

```{toctree}
:caption: Statistical Testing
:maxdepth: 1
:hidden:

statistical_testing
notebooks/Testing_across_subjects
notebooks/Testing_across_sessions_within_subject
notebooks/Testing_across_trials_within_session
notebooks/Testing_across_visits
notebooks/HCP_Testing_across_subjects
notebooks/HCP_multi_level_block_permutation
```

```{toctree}
:caption: API Reference
:maxdepth: 1
:hidden:

modules
glossary
```
