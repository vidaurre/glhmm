# Troubleshooting

Common problems and how to fix them.

---

## States collapse to zero

Some states capture almost all the data and others go empty. Their fractional occupancy drops to near zero during training.

**Why it happens:** the random initialisation converged to a solution where probability is concentrated in only a few states, or K is too large for the data.

**What to try:**
- Increase `initrep` (default 5) to try more random starts: `options = {'initrep': 10}`. GLHMM keeps the best result automatically.
- Reduce K and work back up.
- Check that `deactivate_states` is `True` (the default). Empty states are removed automatically, so a final model with fewer than K states is normal.

---

## Model does not converge

Free energy has not reached a stable value within the maximum number of training cycles.

**What to try:**
- Increase `cyc` (the default is 500): `options = {'cyc': 1000}`.
- Loosen the stopping criterion: `options = {'tol': 1e-3}` (the default is `1e-4`). This stops training earlier, when free energy changes by less than `tol` between cycles.
- For stochastic training, free energy fluctuates more between cycles by design because each cycle only sees a subset of the data. This is normal and not a sign that something is wrong.

---

## `indices` given as session lengths

```
ValueError: negative dimensions are not allowed
```

You get this error if `indices` is a list of session lengths, e.g. `[1000, 800]`. It must be a 2D array with the start and end of each session. Use one of the helper functions to build it:

```python
from glhmm.auxiliary import make_indices_from_T
indices = make_indices_from_T([1000, 800, 950])  # lengths to start/end pairs
```

See [Getting Started](getting_started.md) for all three helper functions.

---

## Memory error on large datasets

The full data matrix does not fit in RAM.

glhmm can preprocess, train and decode file by file, so the data never have to fit in memory. See [Large datasets](large_datasets.md).

---

## Training on files fails

```
UnboundLocalError: local variable 'dat' referenced before assignment
```

`train()` and `decode()` read `.npz` or `.mat` files, not `.npy`. Run the files through `preproc.preprocess_data(files=..., combine_outputs=False)` first, which saves them as `.npz`. See [Large datasets](large_datasets.md).

---

## GPU is not available

If `nvidia-smi` isn't recognised in the terminal, your machine has no NVIDIA GPU or its driver isn't installed, so glhmm can't use a GPU there. This is common on laptops. GPU training is usually done on a compute cluster.

---

## CuPy is not installed

```
ModuleNotFoundError: No module named 'cupy'
```

glhmm needs CuPy to use the GPU. [Training options](training_options.md) shows how to install the right version for your CUDA.

---

## Multiple CuPy versions installed

```
UserWarning: CuPy may not function correctly because multiple CuPy packages are installed
```

Having more than one CuPy package installed at the same time causes conflicts. Remove all of them and reinstall only the one that matches your CUDA version:

```bash
pip uninstall cupy-cuda11x cupy-cuda12x
pip install cupy-cuda12x   # replace with your CUDA version
```

---

## CUDA driver version is insufficient

```
CUDARuntimeError: cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version
```

Your NVIDIA driver is too old for the CUDA version CuPy was built for. Either install an older CuPy version that matches your driver, or update your NVIDIA GPU driver from the NVIDIA website. Run `nvidia-smi` to see the maximum CUDA version your current driver supports, then install the matching CuPy package.

---

## NaN values in γ or free energy

Training produces `nan` values.

**Most likely cause:** the data is not scaled. Very large or very small values cause numerical errors during training. Standardise your data before training:

```python
from glhmm import preproc
Y_proc, indices, log = preproc.preprocess_data(data=Y, indices=indices, standardise=True)
```

A second cause is near-perfectly correlated channels. When two channels carry almost identical signals, the model cannot estimate a valid covariance matrix. Try reducing dimensionality (for example with PCA) before training.
