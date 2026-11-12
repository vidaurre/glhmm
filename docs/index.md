---
html_theme.sidebar_secondary.remove: true
---

# GLHMM: Gaussian-Linear Hidden Markov Models

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

:::{grid-item-card} Tutorial
:link: notebooks/tutorial
:link-type: doc

Train a model on a small simulated dataset, check that it converged, and look at the states.
:::

:::{grid-item-card} From data to results
:link: workflow
:link-type: doc

The steps of a full analysis, from preparing the data to testing and prediction, with the right page or notebook for each.
:::

:::{grid-item-card} How it works
:link: key_concepts
:link-type: doc

What K, covtype, δ, γ and the free energy mean, and which settings to use.
:::

:::{grid-item-card} Large datasets
:link: large_datasets
:link-type: doc

Data that don't fit in memory: preprocess, train and decode file by file.
:::

:::{grid-item-card} Training options and GPU
:link: training_options
:link-type: doc

The options that control the training, and how to speed it up on a GPU.
:::

::::

---

## What can GLHMM do?

- **Identify brain states**: find recurring patterns of activity or connectivity that the brain cycles through over time
- **Work with multiple data modalities**: fMRI, MEG, EEG, ECoG, and more
- **Relate brain dynamics to behaviour**: predict cognitive scores, classify clinical groups, or test for statistical relationships
- **Scale to large datasets**: preprocessing, training and decoding can run file by file, without loading everything into memory (see [Large datasets](large_datasets.md))
- **GUI available**: a code-free interface is available at [github.com/Nick7900/glhmm_protocols](https://github.com/Nick7900/glhmm_protocols)

---

## Citing GLHMM

If you use GLHMM in your research, please cite the toolbox paper:

> Vidaurre, D., Masaracchia, L., Larsen, N.Y., Ruijters, L.R.P.T., Alonso, S., Ahrends, C. & Woolrich, M.W. (2025). *The Gaussian-linear hidden Markov model: a Python package.* Imaging Neuroscience, 3, imag_a_00460. [https://doi.org/10.1162/imag_a_00460](https://doi.org/10.1162/imag_a_00460)

If you use the statistical testing protocols, please also cite:

> Larsen, N.Y., Paulsen, L.B., Ahrends, C., Winkler, A.M. & Vidaurre, D. (2026). *A comprehensive framework for statistical testing of brain dynamics.* Nature Protocols, 21, 3148–3179. [https://doi.org/10.1038/s41596-025-01300-2](https://www.nature.com/articles/s41596-025-01300-2)

The protocols and a code-free GUI for the statistical tests are available at [github.com/Nick7900/glhmm_protocols](https://github.com/Nick7900/glhmm_protocols).

---

```{toctree}
:caption: Getting Started
:maxdepth: 1
:hidden:

getting_started
notebooks/tutorial
workflow
key_concepts
notebooks/Preprocessing
training_options
large_datasets
troubleshooting
```

```{toctree}
:caption: Model types
:maxdepth: 1
:hidden:

notebooks/GaussianHMM_example
notebooks/GLHMM_example
notebooks/HMM-TDE_vs_HMM-MAR_example
```

```{toctree}
:caption: Choosing the number of states
:maxdepth: 1
:hidden:

notebooks/HMM_state_selection_standard
notebooks/HMM_state_selection_stochastic
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
