# Large datasets

If your data don't fit in memory, e.g. MEG or EEG from many subjects, glhmm can do each step file by file. Only a few files are in memory at a time, during preprocessing, training and decoding. The [State number selection with stochastic training](notebooks/HMM_state_selection_stochastic.ipynb) notebook goes through this with an example dataset.

---

## 1. Save one file per subject or session

Each file holds one array of shape `(n_timepoints, n_channels)`, e.g. as `.npy`. The files should all have the same channels in the same order.

## 2. Preprocess file by file

Pass the list of files to `preprocess_data()`. It processes them one at a time and saves the results in a new folder. Set `combine_outputs=False` to get one output file per input file, otherwise they're merged into one:

```python
from glhmm import preproc
import glob

files = sorted(glob.glob('/data/subjects/*.npy'))

preprocessed_files, log = preproc.preprocess_data(
    files=files,
    standardise=True,
    combine_outputs=False,
    output_dir='/data/preprocessed/'
)
```

The output files are `.npz` files, which is what `train()` and `decode()` read. Don't pass the original `.npy` files to `train()`, it can't read them.

Most preprocessing options work in this mode too, such as filtering, downsampling, PCA and TDE embedding (`lags`). PCA is computed across all files, so all subjects end up in the same space. ICA doesn't work file by file. See the [Preprocessing](notebooks/Preprocessing.ipynb) notebook for the options.

## 3. Train with stochastic training

Stochastic training picks a small batch of files in every cycle, updates the model with them, and moves on to the next batch. Pass the list of files instead of `Y` and `indices`, and set `stochastic` to `True`:

```python
from glhmm import glhmm

hmm = glhmm.glhmm(K=6, covtype='full', model_mean='no', model_beta='no')
options = {'stochastic': True, 'Nbatch': 10}

Gamma, Xi, fe = hmm.train(files=preprocessed_files, options=options)
```

The model settings are explained in [How it works](key_concepts.md). `model_mean='no'` is the usual choice for MEG or EEG with TDE, where the covariance of each state holds its spectral content.

The most important options are:

| Option | Default | What it does |
|---|---|---|
| `Nbatch` | `10` (or half the files, if fewer) | Number of files per batch. Use at least 10 if you can. Larger batches make the training less noisy, but use more memory. |
| `cyc` | `250` | Number of training cycles. Each file is used about `cyc` × `Nbatch` / (number of files) times. |
| `forget_rate` | `0.75` | How quickly the updates get smaller over the cycles. Should be between 0.5 and 1. |
| `base_weights` | `0.25` | Makes files that haven't been used much more likely to be picked, so all files are used about equally often |
| `initNbatch` | same as `Nbatch` | Number of files per batch during initialisation |
| `initcyc` | `25` | Number of cycles during initialisation |

The general training options on [Training options](training_options.md) apply as well.

## 4. Check that it converged

Stochastic training usually runs all `cyc` cycles, because the free energy goes up and down a bit from batch to batch. If the free energy is still going down at the end, glhmm gives a warning. Then increase `cyc`, or `Nbatch` if memory allows, and train again. You can plot the free energy (`fe`) with `graphics.plot_free_energy`.

## 5. Decode one file at a time

With stochastic training, the `Gamma` and `Xi` that `train()` returns are empty, so you need to decode the files afterwards. `hmm.decode(None, None, files=preprocessed_files)` would load all files at once, so with large data, decode them one at a time and keep only what you need. Here we get the Viterbi path of each file and the fraction of time in each state (fractional occupancy):

```python
import numpy as np
from pathlib import Path

FO = []
for f in preprocessed_files:
    vpath = hmm.decode(None, None, files=[f], viterbi=True)          # timepoints x states, one-hot
    np.save(Path(f).with_name(Path(f).stem + '_vpath.npy'), vpath)   # save it, if you need it later
    FO.append(vpath.mean(axis=0))                                   # fraction of time in each state
FO = np.array(FO)   # files x states
```

If you need the state probabilities instead, use `Gamma, Xi, _ = hmm.decode(None, None, files=[f])`.

## If memory is still a problem

- Use a smaller `Nbatch`. The training gets noisier, so you may need more `cyc`.
- Set `serial=True` to process the sessions in a batch one after another (this can't be combined with the GPU).
- On a GPU, increase `gpuChunks` (see [Training options](training_options.md)).
