# Getting Started

This page walks you through installing GLHMM, explains what data format the toolbox expects, and shows a minimal working example so you can see the model in action.

---

## Installation

Install GLHMM from PyPI:

```bash
pip install glhmm
```

That is all you need to get started. GPU support is optional. See [Training Options](training_options.md) if you want to speed things up on a GPU.

---

## What data does GLHMM need?

GLHMM works with time series data. For example, fMRI BOLD signals or MEG recordings. The data should be organised as a 2D array where rows are timepoints and columns are brain regions or channels:

```
shape: (n_timepoints, n_channels)
```

If you have a single recording session, you can pass it directly. If you have multiple subjects or sessions, you concatenate them into one array and tell the model where each session starts and ends using an `indices` array:

```python
import numpy as np

# Two sessions: 1000 and 800 timepoints
Y = np.concatenate([session1, session2], axis=0)  # shape (1800, n_channels)

indices = np.array([[0, 1000],
                    [1000, 1800]])
```

Each row in `indices` gives the start and end timepoint of one session.

---

## A minimal example

This fits a 6-state model to a single session of simulated data:

```python
import numpy as np
from glhmm import glhmm

# Simulated data — replace with your own
Y = np.random.randn(5000, 80)  # 5000 timepoints, 80 brain regions

# Set up the model
hmm = glhmm.glhmm(K=6, covtype='full', model_mean='no')

# Train
Gamma, Xi, fe = hmm.train(Y=Y)

print(Gamma.shape)
print(f"Free energy: {fe[-1]:.2f}")
print(f"Most likely state at t=0: {Gamma[0].argmax()}")
```

Expected output:

```
Cycle 1, free energy: -1.23e+07
Cycle 2, free energy: -1.18e+07
...
Converged.
(5000, 6)
Free energy: -1.15e+07
Most likely state at t=0: 3
```

γ is a `(n_timepoints, K)` array. Each row gives the probability of being in each state at that timepoint. The rows sum to 1.

---

## Multiple subjects or sessions

For multiple subjects or sessions, concatenate the data and pass an `indices` array. GLHMM has three helper functions that build `indices` for you:

**If you have a list of arrays** (one per subject or session):

```python
from glhmm.statistics import get_indices_from_list

sessions = [session1, session2, session3]
Y = np.concatenate(sessions, axis=0)
indices = get_indices_from_list(sessions)
```

**If you know the length of each session** (number of timepoints):

```python
from glhmm.auxiliary import make_indices_from_T

T = [1000, 800, 950]
indices = make_indices_from_T(T)
print(indices)
```

```
[[   0 1000]
 [1000 1800]
 [1800 2750]]
```

**If all sessions have the same length:**

```python
from glhmm.statistics import get_indices_timestamp

indices = get_indices_timestamp(n_timestamps=1000, n_subjects=3)
print(indices)
```

```
[[   0 1000]
 [1000 2000]
 [2000 3000]]
```

---

## What to do after training

Once the model has finished training, typical next steps are:

- **Look at the states**: `hmm.get_covariance_matrices()` or `hmm.get_means()` shows what each state captures.
- **Get a hard state assignment**: `np.argmax(Gamma, axis=1)` gives the most likely state at each timepoint.
- **Relate states to behaviour**: see the [Prediction tutorial](notebooks/Prediction_tutorial.ipynb) notebook.
- **Run statistical tests**: see the [Statistical testing](statistical_testing.md) page.
- **Compare models with different K**: see the [State number selection](notebooks/HMM_state_selection_standard.ipynb) notebook.

---

## Tutorials and examples

### Getting started with different model types

| Notebook | What it covers |
|---|---|
| [Tutorial](notebooks/tutorial.ipynb) | Overview of GLHMM and the different model types |
| [Gaussian HMM](notebooks/GaussianHMM_example.ipynb) | Standard HMM on a single set of time series |
| [GLHMM example](notebooks/GLHMM_example.ipynb) | HMM with two sets of time series (brain + behaviour) |
| [TDE-HMM vs MAR-HMM](notebooks/HMM-TDE_vs_HMM-MAR_example.ipynb) | TDE-HMM (lagged covariance, scales to whole-brain MEG) vs MAR-HMM (explicit autoregressive model, better for lower-dimensional recordings where AR dynamics are the scientific focus) |
| [Preprocessing](notebooks/Preprocessing.ipynb) | How to prepare your data before training |

### Choosing the number of states

| Notebook | What it covers |
|---|---|
| [State number selection](notebooks/HMM_state_selection_standard.ipynb) | How to compare models with different K values |

### Relating states to behaviour

| Notebook | What it covers |
|---|---|
| [Prediction tutorial](notebooks/Prediction_tutorial.ipynb) | Predicting behavioural variables from HMM outputs |
| [Prediction split](notebooks/Prediction_split.ipynb) | Prediction with cross-validation splits |

### Statistical testing

| Notebook | What it covers |
|---|---|
| [Testing across subjects](notebooks/Testing_across_subjects.ipynb) | Permutation testing between subjects |
| [Testing across sessions](notebooks/Testing_across_sessions_within_subject.ipynb) | Within-subject testing across sessions |
| [Testing across trials](notebooks/Testing_across_trials_within_session.ipynb) | Testing across trials within a session |
| [Testing across visits](notebooks/Testing_across_visits.ipynb) | Testing brain state measures against simultaneous physiological signals (e.g. heart rate, pupil size, skin conductance) |
| [HCP: testing across subjects](notebooks/HCP_Testing_across_subjects.ipynb) | Testing example with HCP dataset |
| [HCP: multi-level permutation](notebooks/HCP_multi_level_block_permutation.ipynb) | Block permutation for structured HCP data |
