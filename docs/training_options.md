# Large datasets and GPU

How to train GLHMM on datasets that do not fit in memory, and how to speed up training with a GPU.

---

## Standard training

Pass `Y` and `indices` directly for full-batch training:

```python
hmm = glhmm.glhmm(K=6, covtype='full', model_mean='no')
Gamma, Xi, fe = hmm.train(Y=Y, indices=indices)
```

This works well for most fMRI datasets. For large MEG or EEG datasets that do not fit in RAM, use stochastic training instead.

---

## Stochastic training

Stochastic training reads your data one small batch of files at a time. The full dataset never needs to be in memory at once. This is the recommended approach when you have many subjects or long recordings.

### What you need

- Your data saved as individual files, one per subject or session (for example `.npy` files).
- Pass a list of file paths to `train()` using the `files` argument instead of `Y` and `indices`.

### Step 1: preprocess your files

Before training, preprocess each file separately and save the results. Pass `files` to `preproc.preprocess_data()` and point it at an output directory:

```python
from glhmm import preproc
import glob

files = sorted(glob.glob('/data/subjects/*/data.npy'))

preproc.preprocess_data(
    files=files,
    standardise=True,
    onpower=False,
    output_dir='/data/preprocessed/'
)

preprocessed_files = sorted(glob.glob('/data/preprocessed/*.npy'))
```

### Step 2: train the model

Pass the preprocessed file list to `train()`:

```python
from glhmm import glhmm

hmm = glhmm.glhmm(K=6, covtype='full', model_mean='no')

options = {'stochastic': True, 'Nbatch': 10}

Gamma, Xi, fe = hmm.train(files=preprocessed_files, options=options)
```

After training, γ is not automatically computed for the full dataset. Call `hmm.decode` afterwards to get the complete state time courses:

```python
Gamma, Xi, _ = hmm.decode(None, None, files=preprocessed_files)
```

### Key options

| Option | Default | What it does |
|---|---|---|
| `stochastic` | `False` | Set to `True` to enable stochastic training |
| `Nbatch` | 10 | Number of files per mini-batch. A good starting point is around 10% of your total files; larger batches give more stable updates but use more memory. |
| `initNbatch` | Same as `Nbatch` | Files used during initialisation |
| `cyc` | `100` | Maximum number of training cycles |
| `initcyc` | `25` | Training cycles during initialisation |
| `forget_rate` | `0.75` | How much earlier batches influence the model. Lower values make the model adapt more quickly to recent data; higher values give earlier batches more lasting influence. |
| `base_weights` | `0.25` | Minimum weight given to any batch |

---

## GPU acceleration

GLHMM can use a GPU to speed up training. This is useful when training is slow because K is large, the dataset is large, or both.

### Do you have a compatible GPU?

Run this in your terminal before installing anything:

```bash
nvidia-smi
```

If the command is not found, you do not have an NVIDIA GPU or the driver is not installed. GPU acceleration will not work.

If it runs, it prints the GPU model and a CUDA version number (for example `CUDA Version: 12.4`). Note that version number — you need it to install the right CuPy build.

### Installing CuPy

GPU support requires [CuPy](https://cupy.dev/), the GPU-accelerated equivalent of NumPy. Install the version that matches the CUDA version shown by `nvidia-smi`:

```bash
pip install cupy-cuda12x   # for CUDA 12.x
pip install cupy-cuda11x   # for CUDA 11.x
```

See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for other versions.

After installing, confirm CuPy can see your GPU by running this in Python:

```python
import cupy as cp
print(cp.cuda.runtime.getDeviceCount())  # should print 1 or more
```

If you get `ModuleNotFoundError: No module named 'cupy'`, CuPy is not installed yet. Run the `pip install` command above first.

### How to enable it

Add `gpu_acceleration` to your options:

```python
options = {
    'gpu_acceleration': 1,
    'gpuChunks': 1,       # increase this if you run out of GPU memory
    'verbose': True,
}

Gamma, Xi, fe = hmm.train(Y=Y, indices=indices, options=options)
```

| Value | What it does |
|---|---|
| `0` | No GPU (default) |
| `1` | GPU-accelerated forward-backward pass |
| `2` | Full GPU computation -- faster, but uses more GPU memory |

If you run out of GPU memory, increase `gpuChunks` to split the computation into smaller pieces:

```python
options = {
    'gpu_acceleration': 2,
    'gpuChunks': 4,
}
```

Note: GPU acceleration does not work together with `serial=True`. If you enable both, GPU acceleration is automatically disabled.

---

## Other options

| Option | Default | What it does |
|---|---|---|
| `cyc` | `100` | Maximum number of training cycles |
| `tol` | `1e-4` | Stop training when free energy changes by less than this |
| `verbose` | `True` | Print progress during training |
| `initrep` | `5` | Number of random initialisations to try. The best one is kept. |
| `deactivate_states` | `True` | Automatically remove states that go empty during training |
| `serial` | `False` | Process sessions one at a time (uses less memory, but slower) |
