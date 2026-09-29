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
- Increase `cyc`: `options = {'cyc': 200}`.
- Loosen the stopping criterion: `options = {'tol': 1e-3}` (the default is `1e-4`). This stops training earlier, when free energy changes by less than `tol` between cycles.
- For stochastic training, free energy fluctuates more between cycles by design because each cycle only sees a subset of the data. This is normal and not a sign that something is wrong.

---

## `indices` shape error

```
ValueError: indices must have shape (n_sessions, 2)
```

`indices` must be a 2D array of start/end pairs, not a list of lengths. Use one of the helper functions to build it:

```python
from glhmm.auxiliary import make_indices_from_T
indices = make_indices_from_T([1000, 800, 950])  # lengths to start/end pairs
```

See [Getting Started](getting_started.md) for all three helper functions.

---

## Memory error on large datasets

The full data matrix does not fit in RAM.

Switch to stochastic training, which reads one mini-batch of files at a time:

```python
options = {'stochastic': True, 'Nbatch': 10}
Gamma, Xi, fe = hmm.train(files=files, options=options)
```

Your data must be saved as individual files (one per subject). See [Training Options](training_options.md) for the full setup.

---

## GPU is not available

Before installing anything, check whether your machine has an NVIDIA GPU by running this in your terminal:

```bash
nvidia-smi
```

If the command is not recognised, your machine does not have an NVIDIA GPU or the driver is not installed. GPU acceleration will not work on this machine. This is common on laptops — GPU training is typically done on a compute cluster.

If `nvidia-smi` runs, note the CUDA version shown in the top-right corner (for example `CUDA Version: 12.4`). You need this to install the right CuPy version.

---

## CuPy is not installed

```
ModuleNotFoundError: No module named 'cupy'
```

Install CuPy matching the CUDA version shown by `nvidia-smi`:

```bash
pip install cupy-cuda12x   # CUDA 12.x
pip install cupy-cuda11x   # CUDA 11.x
```

See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for other versions.

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
Y_proc = preproc.preprocess_data(data=Y, indices=indices, standardise=True)
```

A second cause is near-perfectly correlated channels. When two channels carry almost identical signals, the model cannot estimate a valid covariance matrix. Try reducing dimensionality (for example with PCA) before training.
