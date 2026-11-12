# How it works

This page explains the main ideas behind glhmm and the settings you choose when you set up a model.

---

## What is a Hidden Markov Model?

The basic idea is that the brain does not stay in a fixed state. It moves between different patterns of activity over time. An HMM finds those patterns automatically from your data.

You tell the model how many patterns to look for (that is `K`). The model then learns what each pattern looks like and estimates, at every timepoint, how likely it is that the brain was in each pattern. You never observe the patterns directly. That is why they are called *hidden* states.

---

## K -- how many states?

`K` is the number of states you ask the model to find. It is the first thing you need to decide.

- If K is too small, the model groups different patterns together and misses detail.
- If K is too large, it starts inventing distinctions that are not really there, splitting one real pattern into several nearly identical copies.

There is no single correct answer. A practical approach is to fit the model several times with different values of K (for example 4, 6, 8, 10, 12) and then compare the results. See the [State number selection](notebooks/HMM_state_selection_standard.ipynb) notebook for how to do this.

---

## covtype -- what does each state look like?

`covtype` decides what the covariance of each state looks like, i.e. whether a state also captures how regions co-vary, or only how much each region varies.

| Value | What it does | When to use |
|---|---|---|
| `'full'` | Each state has its own covariance matrix, capturing how brain regions co-activate | Recommended for fMRI and most uses. Not the default, so set it yourself. |
| `'diag'` | Each state only captures activity levels, not region-to-region relationships | Very high-dimensional data |
| `'sharedfull'` | All states share the same covariance structure | Rarely used |

---

## model_mean -- do states differ in their mean activity?

`model_mean` controls whether the model also fits the average activation level of each state, or only the relationships between regions.

| Value | What it means |
|---|---|
| `'no'` | States differ only in their covariance, i.e. in how regions are connected |
| `'state'` | Each state also has its own mean, so states can differ in how active the regions are |

Standardising removes each session's overall mean, but states can still differ in their mean activity. Use `'no'` if you want the states to be defined only by connectivity, and `'state'` if differences in activity matter too. If the states end up following the subjects (each subject mostly in its own state), `'state'` is worth a try. The [Tutorial](notebooks/tutorial.ipynb) uses `'state'`, since its data were simulated with states that differ in their mean.

---

## model_beta -- standard HMM or GLHMM?

`model_beta` decides whether the states also model the relation between your data `Y` and a second set of variables `X`, such as behaviour or task conditions.

| Value | What it means |
|---|---|
| `'no'` | A standard HMM on `Y` only. Use this if you only have one set of time series. |
| `'state'` | A GLHMM: each state also has its own regression from `X` to `Y`. You then have to pass `X` to `train()`. |

`'state'` is the default, so for a standard HMM you need to set `model_beta='no'`. Otherwise training stops with "If you want to model beta, X is needed as an argument".

---

## δ (Dirichlet diagonal) -- how long does the brain stay in each state?

δ is the Dirichlet diagonal. It controls how "sticky" the states are: how long the model expects the brain to remain in a state before switching to another one.

- High δ: the model expects long, stable states (slow switching).
- Low δ: the model expects rapid switching between states.

This is only a prior. The model still learns the transition probabilities from the data.

You set δ when creating the model using the `dirichlet_diag` parameter (default: 10):

```python
hmm = glhmm.glhmm(K=6, covtype='full', model_mean='no', model_beta='no', dirichlet_diag=10)
```

If you are not sure what timescale suits your data, train models with several δ values and compare the state time courses. States that last only 1 or 2 timepoints suggest δ is too low. If the model spends almost all time in one or two states and almost never switches, δ may be too high.

---

## Model types

GLHMM supports several types of models. They differ in what defines a state:

| Model type | What defines a state |
|---|---|
| Gaussian HMM | Its covariance (which regions co-vary) and, with `model_mean='state'`, its mean activity |
| TDE-HMM | The covariance of the time-embedded data, i.e. also across short time lags. So each state has its own spectral content. Scales well to whole-brain MEG or EEG with many channels. |
| MAR-HMM | Its own autoregressive model: how activity at previous timepoints predicts the current one. Each state has roughly D²P parameters (dimensions² × AR order), so with more than about 30 to 50 dimensions it gets expensive and can overfit unless you reduce the data with PCA first. |
| GLHMM | Its own regression from a second set of variables `X` (e.g. behaviour or task conditions) to the data `Y` (`model_beta='state'`) |

Which one to use for your data is shown in [From data to results](workflow.md), with an example notebook for each.

---

## Free energy -- how good is the model?

When you train a model, it returns `fe` (free energy), one value per training cycle. It measures how well the model fits the data, with a penalty for how complex the model is. Lower is better, and training tries to make it as low as possible.

You can use the free energy to compare models with different K. Because of the penalty, it doesn't automatically go down when you add states, but with a lot of data it often keeps going down anyway. So picking the K with the lowest free energy isn't enough. The [State number selection](notebooks/HMM_state_selection_standard.ipynb) notebook shows how to choose K using both the free energy and how stable the states are.

---

## γ -- which state is the brain in?

γ is the main output of training (called `Gamma` in the code). It is a matrix with one row per timepoint and one column per state. Each row sums to 1 and gives the probability that the brain was in each state at that moment.

To get the single most likely state at each timepoint, decode the Viterbi path:

```python
vpath = hmm.decode(None, Y, indices, viterbi=True)   # one column per state, 1 for the active state
state_sequence = np.argmax(vpath, axis=1)            # state number at each timepoint
```

Some common summaries:

```python
from glhmm import utils
FO = utils.get_FO(Gamma, indices)                              # fraction of time spent in each state
mean_lt, median_lt, max_lt = utils.get_life_times(vpath, indices)  # how long each visit to a state lasts
```
