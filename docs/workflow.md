# From data to results

This page shows the steps of a typical HMM analysis and where to find each of them in the documentation. If you're new to glhmm, do the [Tutorial](notebooks/tutorial.ipynb) first. It goes through steps 1 to 6 with a small dataset.

---

## 1. Prepare your data

glhmm needs a 2D array (timepoints × channels or regions) with all sessions stacked in time, plus an `indices` array with the start and end of each session. [Getting started](getting_started.md) shows how to build both.

Then preprocess the data. At the very least, standardise each session. Depending on your data you may also want to filter, downsample or reduce the dimensions with PCA. The [Preprocessing](notebooks/Preprocessing.ipynb) notebook shows how. For MEG or EEG with a TDE-HMM, the data are also embedded in time at this step (see the [TDE-HMM vs MAR-HMM](notebooks/HMM-TDE_vs_HMM-MAR_example.ipynb) notebook).

## 2. Choose the model

Which model type fits your data depends on what you want the states to capture:

| Your data and question | Model | Example |
|---|---|---|
| fMRI, states as patterns of connectivity or activation | Gaussian HMM | [Gaussian HMM](notebooks/GaussianHMM_example.ipynb) |
| MEG or EEG, states with their own spectral content | TDE-HMM | [TDE-HMM vs MAR-HMM](notebooks/HMM-TDE_vs_HMM-MAR_example.ipynb) |
| MEG or EEG, interested in the autoregressive dynamics themselves | MAR-HMM | [TDE-HMM vs MAR-HMM](notebooks/HMM-TDE_vs_HMM-MAR_example.ipynb) |
| Brain data plus a second variable (e.g. behaviour) that the states should relate to | GLHMM | [GLHMM example](notebooks/GLHMM_example.ipynb) |

Then set the model's settings (`covtype`, `model_mean`, `model_beta` and `dirichlet_diag`). [How it works](key_concepts.md) explains what each one does and which values to use.

## 3. Decide how to train

If your data fit in memory, train on them directly, see [Training options](training_options.md). If not, glhmm can preprocess, train and decode file by file, see [Large datasets](large_datasets.md). Both can run on a GPU if training is slow.

## 4. Train and check that it converged

Train the model and look at the free energy over the training cycles. It should go down and level off. If glhmm warns that the training hasn't converged, train for more cycles. The [Tutorial](notebooks/tutorial.ipynb) shows how to check this for one model, and the state selection notebooks for many models at once.

## 5. Choose the number of states

Train several values of K, several times each, and compare how well they fit and how stable the states are across runs. Choose K before you do any other analysis with the model. See [State number selection](notebooks/HMM_state_selection_standard.ipynb), or the [stochastic version](notebooks/HMM_state_selection_stochastic.ipynb) for large datasets.

## 6. Look at the states

Before you use the states, check what they are:
- what each state looks like (its covariance or mean),
- how the model switches between states (the transition probabilities),
- when each state is active (the Viterbi path),
- summaries per subject, such as the fraction of time in each state (fractional occupancy) and how long the visits last (life times).

If a state only shows up in a few subjects, it may capture differences between subjects rather than a state they share. The [Tutorial](notebooks/tutorial.ipynb) shows all of these plots.

## 7. Relate the states to other variables

With the per-subject summaries you can:
- predict behaviour or other traits, see the [Prediction tutorial](notebooks/Prediction_tutorial.ipynb),
- test whether the states differ between subjects, sessions, trials or conditions, see [Statistical testing](statistical_testing.md), which also helps you pick the right test for your design.

If something goes wrong along the way, see [Troubleshooting](troubleshooting.md).
