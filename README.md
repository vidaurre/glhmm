# GLHMM

[![Documentation Status](https://readthedocs.org/projects/glhmm/badge/?version=latest)](https://glhmm.readthedocs.io/en/latest/?badge=latest)

The GLHMM toolbox provides facilities to fit a variety of Hidden Markov models (HMM) based on the Gaussian distribution, which we generalise as the Gaussian-Linear HMM. 
Crucially, the toolbox has a focus on finding associations at various levels between brain data (EEG, MEG, fMRI, ECoG, etc) and non-brain data, such as behavioural or physiological variables.

## Important links

- Official source code repo: <https://github.com/vidaurre/glhmm>
- GLHMM documentation: <https://glhmm.readthedocs.io/en/latest/index.html>

## Dependencies

The required dependencies to use glhmm are:

- Python >= 3.6
- NumPy
- numba
- scikit-learn
- scipy
- matplotlib
- seaborn

## Installation

- To install from the repo, use the following command:

```bash
pip install --user git+https://github.com/vidaurre/glhmm
