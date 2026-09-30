# How it works

This page explains the main ideas and settings behind GLHMM. If something is unclear when you first run the model, this is a good place to start.

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

`covtype` controls the mathematical shape of each state. In practice, it determines whether the model captures relationships *between* brain regions (not just activity in each region separately).

| Value | What it does | When to use |
|---|---|---|
| `'full'` | Each state has its own covariance matrix, capturing how brain regions co-activate | Default for fMRI and most uses |
| `'diag'` | Each state only captures activity levels, not region-to-region relationships | Very high-dimensional data |
| `'sharedfull'` | All states share the same covariance structure | Rarely used |

For most neuroimaging work, use `covtype='full'`.

---

## model_mean -- do states differ in their mean activity?

`model_mean` controls whether the model also fits the average activation level of each state, or only the relationships between regions.

| Value | What it means |
|---|---|
| `'no'` | States differ only in covariance. Use this for demeaned or standardised data. |
| `'state'` | Each state also has its own mean. Use this when raw activation levels matter. |

For preprocessed neuroimaging data (which is typically demeaned), use `model_mean='no'`.

---

## δ (Dirichlet diagonal) -- how long does the brain stay in each state?

δ is the Dirichlet diagonal. It controls how "sticky" the states are: how long the model expects the brain to remain in a state before switching to another one.

- High δ: the model expects long, stable states (slow switching).
- Low δ: the model expects rapid switching between states.

This is a starting preference, not a fixed rule. The model can update it based on what the data actually shows.

You set δ when creating the model using the `dirichlet_diag` parameter (default: 10):

```python
hmm = glhmm.glhmm(K=6, covtype='full', model_mean='no', dirichlet_diag=10)
```

If you are not sure what timescale suits your data, train models with several δ values and compare the state time courses. States that last only 1 or 2 timepoints suggest δ is too low. If the model spends almost all time in one or two states and almost never switches, δ may be too high.

---

## Model types

GLHMM supports several different types of models. The right choice depends on your data and what you are trying to find:

| Model type | Use this when... |
|---|---|
| Gaussian HMM | Your data is fMRI, or you want states that reflect which brain regions activate together |
| TDE-HMM | Your data is MEG or EEG and you want states defined by transient spectral and spatial patterns. Each state is a different lagged covariance pattern. Scales well to whole-brain recordings with many channels or parcels. |
| MAR-HMM | Your data is MEG or EEG and the autoregressive dynamics themselves are of scientific interest — that is, you want to know how activity at previous timepoints predicts current activity. Each state is an explicit autoregressive model with roughly D²P parameters (effective dimensions² × AR order). With ~10–20 dimensions the model is usually manageable; above ~30–50 dimensions it becomes expensive and prone to overfitting unless the data are first reduced with PCA. |
| GLHMM | You have brain data and a second variable alongside it (such as behavioural scores, reaction times or task conditions), and you want brain states that are linked to that second variable. |

See the [tutorial notebook](notebooks/tutorial.ipynb) for a hands-on overview of each type.

---

## Free energy -- how good is the model?

When you train a model, it returns `fe` (free energy). This is a measure of how well the model fits the data. Higher values mean a better fit.

Free energy is useful for comparing models with different values of K. However, a model with more states will always have a higher free energy, because more states give the model more flexibility. You cannot simply pick the highest free energy. The [State number selection](notebooks/HMM_state_selection_standard.ipynb) notebook shows how to compare models properly, taking both free energy and state stability into account.

---

## γ -- which state is the brain in?

γ is the main output of training (called `Gamma` in the code). It is a matrix with one row per timepoint and one column per state. Each row sums to 1 and gives the probability that the brain was in each state at that moment.

To get a single state label at each timepoint:

```python
state_sequence = np.argmax(Gamma, axis=1)
```

To get the average time spent in each state:

```python
from glhmm import utils
vpath = np.argmax(Gamma, axis=1)  # hard state labels
mean_lt, median_lt, max_lt = utils.get_life_times(vpath, indices)
```
