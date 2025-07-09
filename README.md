![Overview Image](logo2_full.png)

[![Documentation Status](https://readthedocs.org/projects/glhmm/badge/?version=latest)](https://glhmm.readthedocs.io/en/latest/?badge=latest)

The GLHMM toolbox provides facilities to fit a variety of Hidden Markov models (HMM) based on the Gaussian distribution, which we generalise as the Gaussian-Linear HMM. 
Crucially, the toolbox has a focus on finding associations at various levels between brain data (EEG, MEG, fMRI, ECoG, etc) and non-brain data, such as behavioural or physiological variables.

## Important links

- Official source code repo: <https://github.com/vidaurre/glhmm>
- GLHMM documentation: <https://glhmm.readthedocs.io/en/latest/index.html>
- Paper: <https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00460/127499>

## Dependencies

The required dependencies to use glhmm are:

- Python >= 3.10
- NumPy
- numba
- scikit-learn
- scipy
- matplotlib
- seaborn
- pickle
- scikit-learn

- cupy (only when using GPU acceleration; requires manual install)
- h5py

## Installation

- To install the latest development version from the repository, use the following command:
```bash
pip install git+https://github.com/vidaurre/glhmm
```

- Alternatively, to install the latest stable release from PyPI, use the command:
```bash
pip install glhmm
```

## Graphical User Interface (GUI)
In addition to using the GLHMM toolbox as a Python package, a graphical user interface (GUI) is now available. The GUI offers an intuitive, code-free way to load data, train models, run statistical tests, and visualise results.

#### Access the GUI
To access the GUI, visit the companion repository: 
**[https://github.com/Nick7900/glhmm_protocols](https://github.com/Nick7900/glhmm_protocols)**

The GUI is built using Streamlit and can be launched locally. Instructions for setup and use are provided in that repository.

#### Video Tutorial
An introductory walkthrough of the GUI is available here:  
[GLHMM GUI Tutorial – YouTube](https://www.youtube.com/watch?v=XPcoK5zCPtU&t=1497s)


