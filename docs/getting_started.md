# Getting Started

How to install glhmm, what format your data should be in, and a small example to train your first model.

---

## Installation

Install GLHMM from PyPI:

```bash
pip install glhmm
```

GPU support is optional, see [Training options](training_options.md).

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

# Random data, replace it with your own
Y = np.random.randn(5000, 80)  # 5000 timepoints, 80 brain regions

# Set up the model (the settings are explained in How it works)
hmm = glhmm.glhmm(K=6, covtype='full', model_mean='no', model_beta='no')

# Train
Gamma, Xi, fe = hmm.train(Y=Y)

print(Gamma.shape)
print(f"Free energy: {fe[-1]:.2f}")
```

While it trains, glhmm prints the free energy of each cycle. It should go down over the cycles. The output looks something like this (your numbers will differ):

```
Init repetition 1 free energy = 748234.89
...
Best repetition: 5
Cycle 1 free energy = 747651.23 at [02:58:50] (0.2s)
Cycle 2 free energy = 747321.25 at [02:58:51] (0.2s)
...
Finished training in 50.29s : active states = 6
(5000, 6)
Free energy: 732731.92
```

Since this is random noise, the model has no real states to find, so it may run all 500 cycles and give a warning that it hasn't converged. With real data it usually stops earlier.

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

## Next steps

- The [Tutorial](notebooks/tutorial.ipynb) trains a model on a small simulated dataset and shows how to check it and look at the states.
- [From data to results](workflow.md) goes through all steps of an analysis and links to the right page or notebook for each.
