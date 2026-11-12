# Training options

This page is about how the training runs: the options you can set, and how to run it on a GPU. If your data are too big to fit in memory, see [Large datasets](large_datasets.md), which trains file by file with stochastic training.

---

## Training

Pass the data `Y` and the start and end of each session (`indices`) to `train()`:

```python
from glhmm import glhmm

hmm = glhmm.glhmm(K=6, covtype='full', model_mean='no', model_beta='no')
Gamma, Xi, fe = hmm.train(Y=Y, indices=indices)
```

The settings in `glhmm.glhmm()` define the model and are explained in [How it works](key_concepts.md).

You can change how the training runs with an `options` dictionary, e.g. `hmm.train(Y=Y, indices=indices, options={'cyc': 1000})`. The main options are:

| Option | Default | What it does |
|---|---|---|
| `cyc` | `500` | Maximum number of training cycles. Training stops earlier once the free energy has converged. If it reaches `cyc` first, glhmm gives a warning. |
| `tol` | `1e-4` | Training stops when the free energy changes by less than this |
| `initrep` | `5` | Number of random initialisations to try. The best one is kept. |
| `deactivate_states` | `True` | Removes states that are hardly used during training. When you compare different K, set it to `False`, so every run keeps all K states. |
| `serial` | `False` | Process the sessions one at a time. Uses less memory, but is slower. |
| `verbose` | `True` | Print progress during training |

These options also apply to stochastic training. The [State number selection](notebooks/HMM_state_selection_standard.ipynb) notebook shows a full example.

---

## Training on a GPU

glhmm can run part of the training on an NVIDIA GPU, which helps when training is slow because K is large, the dataset is large, or both. It works the same way with stochastic training.

### 1. Check that you have a GPU

Run this in a terminal:

```bash
nvidia-smi
```

If the command isn't found, you either don't have an NVIDIA GPU or its driver isn't installed, and GPU training won't work. If it runs, it prints your GPU and a CUDA version, e.g. `CUDA Version: 12.4`. You'll need that number in the next step.

### 2. Install CuPy

glhmm uses [CuPy](https://cupy.dev/) to run on the GPU. Install the version that matches your CUDA version:

```bash
pip install cupy-cuda12x   # for CUDA 12.x
pip install cupy-cuda11x   # for CUDA 11.x
```

For other versions, see the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html). To check that CuPy can see your GPU, run this in Python:

```python
import cupy as cp
print(cp.cuda.runtime.getDeviceCount())  # should print 1 or more
```

### 3. Turn it on

Add `gpu_acceleration` to the options:

```python
options = {'gpu_acceleration': 1}
Gamma, Xi, fe = hmm.train(Y=Y, indices=indices, options=options)
```

With stochastic training, add it to the stochastic options.

| `gpu_acceleration` | What it does |
|---|---|
| `0` | No GPU (default) |
| `1` | The forward-backward pass runs on the GPU |
| `2` | More of the computation runs on the GPU. Faster, but uses more GPU memory. |

If you run out of GPU memory, set `gpuChunks` to split the sessions into smaller chunks, e.g. `{'gpu_acceleration': 2, 'gpuChunks': 4}`.

The GPU isn't used together with `serial=True`. If you set both, glhmm turns the GPU off.
