#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing functions - General/Gaussian Linear Hidden Markov Model
@author: Diego Vidaurre 2023
"""

import math
import numpy as np
import warnings
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy import signal
import scipy.io
import os
from pathlib import Path
from scipy.io import loadmat

from . import auxiliary
from .auxiliary import make_indices_from_T

# import auxiliary

def apply_pca(X,d,whitening=False, exact=True):
    """Applies PCA to the input data X.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_parcels)
        The input data to be transformed.
    d : int or float
        If int, the number of components to keep.
        If float, the percentage of explained variance to keep.
        If array-like of shape (n_parcels, n_components), the transformation matrix.
    exact : bool, default=True
        Whether to use full SVD solver for PCA.

    Returns:
    --------
    X : array-like of shape (n_samples, n_components)
        The transformed data after applying PCA.
    pcamodel : sklearn estimator
        The estimated PCA model
    """
    if type(d) is np.ndarray:
        X -= np.mean(X,axis=0)
        X = X @ d
        whitening = True
        if whitening: X /= np.std(X,axis=0)
        return X, None

    svd_solver = 'full' if exact else 'auto'
    if d >= 1: 
        pcamodel = PCA(n_components=d,svd_solver=svd_solver)
        pcamodel.fit(X)
        X = pcamodel.transform(X)
    else: 
        pcamodel = PCA(svd_solver=svd_solver)
        pcamodel.fit(X)
        ncomp = np.where(np.cumsum(pcamodel.explained_variance_ratio_)>=d)[0][0] + 1
        X = pcamodel.transform(X)
        X = X[:,0:ncomp]
        d = ncomp

    # sign convention equal to Matlab's
    flip_signs = np.ones(d)
    values = np.zeros(d)
    for j in range(d):
        jj = np.where(np.abs(X[:,j]) == np.abs(np.max(X[:,j])) )[0][0]
        values[j] = X[jj,j]
        if X[jj,j] < 0: 
            X[:,j] *= -1 
            flip_signs[j] = -1
            print("Flipping sign of component %d" % j)

    return X, pcamodel

def apply_ica(X,d,algorithm='parallel'):
    """Applies ICA to the input data X.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_parcels)
        The input data to be transformed.
    d : int or float
        If int, the number of components to keep.
        If float, the percentage of explained variance to keep (according to a PCA decomposition)
    algorithm : {"parallel", "deflation"}, default="parallel"
        Specify which algorithm to use for FastICA.

    Returns:
    --------
    X : array-like of shape (n_samples, n_components)
        The transformed data after applying ICA.
    icamode : sklearn estimator
        The estimated ICA model
    """

    if d < 1:
        pcamodel = PCA()
        pcamodel.fit(X)
        ncomp = np.where(np.cumsum(pcamodel.explained_variance_ratio_)>=d)[0][0] + 1
    else: 
        ncomp = d

    icamodel = FastICA(n_components=ncomp,whiten='unit-variance',algorithm=algorithm)
    icamodel.fit(X)
    X = icamodel.transform(X)

    # sign convention equal to Matlab's
    for j in range(ncomp):
        jj = np.where(np.abs(X[:,j]) == np.abs(np.max(X[:,j])) )[0][0]
        if X[jj,j] < 0: X[:,j] *= -1

    return X, icamodel

def dampen_peaks(X,strength=5):
    """Applies dampening of extreme peaks to the input data X, at the group level.
    If the absolute value of X goes beyond 2 standard deviation of X, 
    it gets substituted by the logarithm of the absolute value of X.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_parcels)
        The input data to be transformed.
    strength : positive int
        The strength of dampening. This value refers to the base of the logarithm to use. 
        The bigger the base, the stronger the dampening.

    Returns:
    --------
    X_transformed : array-like of shape (n_samples, n_parcels)
        The transformed data after applying extreme peak dampening.
    """
    
    x_mask = np.abs(X)>2*np.std(X)
    X_transformed = X.copy()
    X_transformed[x_mask] = np.sign(X[x_mask])*(2*np.std(X) - np.log(2*np.std(X))/np.log(strength) + 
                                                np.log(np.abs(X[x_mask]))/np.log(strength))

    return X_transformed



def load_X(file_path):
    """
    Load a data array from a file.

    Parameters:
    -----------
    INPUT_FILE_PATH : str or Path
        Path to the input file (.npy, .npz, or .mat).

    Returns:
    --------
    X : ndarray of shape (n_samples, n_features)
        The loaded data array, reshaped to 2D if needed.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.npy':
        X = np.load(file_path)
    elif ext == '.npz':
        data = np.load(file_path)
        X = data[data.files[0]]
    elif ext == '.mat':
        mat = loadmat(file_path)
        keys = [k for k in mat if not k.startswith("__")]
        if not keys:
            raise ValueError(f"No data found in {file_path}")
        X = mat[keys[0]]
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    return np.array(X)


def resolve_files(files, file_type="npz"):
    supported_types = {"npz", "npy", "mat"}
    ext = f".{file_type.lower().lstrip('.')}"
    if file_type not in supported_types:
        raise ValueError(f"'file_type' must be one of {supported_types}")

    if isinstance(files, (str, Path)) and Path(files).is_dir():
        directory = Path(files)
        return sorted(str(p) for p in directory.glob(f"*{ext}") if p.is_file())

    if isinstance(files, (list, tuple)) and all(isinstance(p, (str, Path)) for p in files):
        return [str(p) for p in files]

    raise ValueError("The 'files' argument must be a list of file paths or a directory containing data files.")


def highdim_pca(C, n_components=None):
    """
    Perform PCA on a high-dimensional correlation or covariance matrix.

    Parameters:
    -----------
    C : ndarray of shape (p, p)
        The input correlation or covariance matrix.
    n_components : int or float or None
        Number of components or proportion of explained variance to retain.

    Returns:
    --------
    eigvecs : ndarray of shape (p, n_components)
        The principal component directions.
    eigvals : ndarray of shape (n_components,)
        The corresponding eigenvalues indicating variance explained.
    """
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    if isinstance(n_components, int):
        eigvecs = eigvecs[:, :n_components]
        eigvals = eigvals[:n_components]
    elif isinstance(n_components, float):
        explained = np.cumsum(eigvals) / np.sum(eigvals)
        k = np.searchsorted(explained, n_components) + 1
        eigvecs = eigvecs[:, :k]
        eigvals = eigvals[:k]
    elif n_components is not None:
        raise ValueError("Invalid type for n_components")

    # Apply Matlab-like sign convention (apply_pca)
    for j in range(eigvecs.shape[1]):
        max_idx = np.argmax(np.abs(eigvecs[:, j]))
        if eigvecs[max_idx, j] < 0:
            eigvecs[:, j] *= -1

    return eigvecs, eigvals

def preprocess_data(data = None,indices = None,
        fs = 1, # frequency of the data
        dampen_extreme_peaks=None, # it can be None, True, or an int with the strength of dampening
        standardise=True, # True / False
        filter=None, # Tuple with low-pass high-pass thresholds, or None
        detrend=False, # True / False
        onpower=False, # True / False
        onphase=False, # True / False
        pca=None, # Number of principal components, % explained variance, or None
        exact_pca=True, # related to how to run PCA
        ica=None, # Number of independent components, % explained variance, or None (if specified, pca is not used)
        ica_algorithm='parallel', # related to how to run PCA
        post_standardise=None, # True / False, standardise the ICA/PCA components?
        downsample=None, # new frequency, or None
        files=None, # list of files to be preprocessed
        output_dir=None,
        file_name = None,
        file_type= "npy",
        lags=None,
        autoregressive_order= None
        ):
    
    """
    Preprocess the input data or files, with support for stochastic training and optional TDE embedding.

    Parameters:
    -----------
    data : array-like, optional
        Raw input data of shape (n_samples, n_parcels), used for in-memory processing.
    indices : array-like of shape (n_sessions, 2), optional
        Start and end indices of each session (only required for in-memory data).
    fs : int or float, default=1
        Sampling frequency of the data.
    dampen_extreme_peaks : int, bool, or None, default=None
        Dampens extreme peaks in the data. If True, uses default strength of 5. If int, specifies strength.
    standardise : bool, default=True
        Whether to standardise (zero-mean, unit-variance) each session.
    filter : tuple of two floats or None, default=None
        Bandpass (low, high), lowpass (0, high), or highpass (low, None) filter.
    detrend : bool, default=False
        Whether to linearly detrend the data.
    onpower : bool, default=False
        Whether to extract signal power using the Hilbert transform.
    onphase : bool, default=False
        Whether to extract phase using the Hilbert transform. If both `onpower` and `onphase` are True,
        power and phase are concatenated.
    pca : int, float, array-like, or None, default=None
        PCA dimensionality reduction. If int, number of components. If float, proportion of variance to retain.
        If array, treated as precomputed PCA matrix.
    exact_pca : bool, default=True
        Whether to use full SVD in PCA (only relevant for in-memory mode).
    ica : int or float or None, default=None
        ICA dimensionality reduction. If int, number of components. If float, proportion of variance to retain.
    ica_algorithm : str, default='parallel'
        ICA algorithm to use (e.g., 'parallel', 'deflation').
    post_standardise : bool or None, default=None
        Whether to standardise data after PCA or ICA. Defaults to True if ICA is used.
    downsample : int or float or None, default=None
        New sampling frequency. If None, no downsampling.
    files : list of str or Path, optional
        If set, enables file-based preprocessing with one file at a time for stochastic training.
    output_dir : str or Path, optional
        Directory to save processed files (only used in file mode).
    file_name : str or None, optional
        Optional suffix to append to output filenames.
    file_type : str, default="npy"
        File format type for loading files.
    lags : list, optional
        If specified, applies temporal delay embedding (TDE) using these lags.
        This prepares the data for use with a Time-Delay Embedded HMM (HMM-TDE).
        This should be a list of integers indicating how many time steps before and after to include.
        For example, use:
            lags = np.arange(-7, 8)
        to include 15 lagged versions of the signal: from 7 time steps before to 7 time steps after.
    autoregressive_order : int, default=None
        Number of lags to include.

    Returns:
    --------
    For in-memory mode:
        data : np.ndarray
            The preprocessed (and optionally embedded/reduced) data.
        indices_new : np.ndarray
            Updated indices after preprocessing and embedding.
        log : dict
            Dictionary containing the preprocessing parameters and models.

    For file-based mode:
        output_file_paths : list of str
            List of paths to saved preprocessed files.
        log : dict
            Dictionary with accumulated preprocessing parameters and PCA statistics.
    """
    if lags is not None and autoregressive_order is not None:
        raise ValueError("Specify either `lags` (for TDE) or `autoregressive_order` (for AR), not both.")

    if autoregressive_order is not None:
        if not isinstance(autoregressive_order, int):
            raise ValueError("`autoregressive_order` must be an integer.")
        if autoregressive_order < 1:
            raise ValueError("`autoregressive_order` must be >= 1.")
    
    # Validate lags if specified
    if lags is not None:
        if not isinstance(lags, (list, np.ndarray)):
            raise ValueError("`lags` must be a list or NumPy array of integers.")
        lags = np.asarray(lags)
        if lags.ndim != 1 or not np.issubdtype(lags.dtype, np.integer):
            raise ValueError("`lags` must be a 1D array or list of integers.")
        if np.any(lags != np.sort(lags)):
            raise ValueError("`lags` must be sorted in ascending order.")
        else:
            print("Applying TDE embedding.")
    # file-based preprocessing for stochastic learning
    if files is not None:
        if ica is not None:
            raise NotImplementedError("ICA cannot be applied in file-wise mode without concatenating data.")
        log = {**locals()}
        del(log["data"], log["indices"], log["files"])
        files = resolve_files(files, file_type=file_type)
        INPUT_FILE_PATHS = [str(p) for p in files]
        if output_dir is None:
            OUTPUT_DIR_PATH = Path(files[0]).parent
        else:
            OUTPUT_DIR_PATH = Path(output_dir)
        OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)

        TEMP_PATHS = []
        all_indices = []
        first = True
        total_T = 0

        if pca is not None or ica is not None:
            accumulate_stats = True
        else:
            accumulate_stats = False

        # Step 1 + 2: Preprocess each file (except PCA), optionally apply TDE, save temp, accumulate PCA stats
        for INPUT_FILE_PATH in INPUT_FILE_PATHS:
            X = load_X(INPUT_FILE_PATH)
            T = X.shape[0]
            indices = np.array([[0, T]])
            p = X.shape[1]

            if dampen_extreme_peaks:
                X -= np.mean(X, axis=0)
                strength = dampen_extreme_peaks if isinstance(dampen_extreme_peaks, int) else 5
                X = dampen_peaks(X, strength)

            if standardise:
                X -= np.mean(X, axis=0)
                X /= np.std(X, axis=0)

            if filter is not None:
                filterorder = 6
                if filter[0] == 0:
                    sos = signal.butter(filterorder, filter[1], 'lowpass', output='sos', fs=fs)
                elif filter[1] is None:
                    sos = signal.butter(filterorder, filter[0], 'highpass', output='sos', fs=fs)
                else:
                    sos = signal.butter(filterorder, filter, 'bandpass', output='sos', fs=fs)
                X = signal.sosfilt(sos, X, axis=0)

            if detrend:
                X = signal.detrend(X, axis=0)

            if onpower and not onphase:
                X = np.abs(signal.hilbert(X, axis=0))

            if onphase and not onpower:
                X = np.unwrap(np.angle(signal.hilbert(X, axis=0)), axis=0)

            if onpower and onphase:
                analytic = signal.hilbert(X, axis=0)
                X_power = np.abs(analytic)
                X_phase = np.unwrap(np.angle(analytic), axis=0)
                X = np.concatenate((X_power, X_phase), axis=1)
                p = X.shape[1]


            if accumulate_stats==False and post_standardise:
                X -= np.mean(X, axis=0)
                X /= np.std(X, axis=0)

            if downsample is not None:
                factor = downsample / fs
                indices_new = np.array([[0, int(np.ceil(T * factor))]])
                gcd = math.gcd(int(downsample), int(fs))
                X = signal.resample_poly(X, int(downsample // gcd), int(fs // gcd), axis=0)
                indices = indices_new

            if autoregressive_order is not None:
                X, Y, indices_new, _ = build_data_autoregressive(
                    data=X,
                    indices=indices,
                    autoregressive_order=autoregressive_order,
                    center_data=post_standardise  # Use same logic
                )
            elif lags is not None:
                X, indices_new = build_data_tde(X, indices, lags)
            else:
                indices_new = indices
                Y = X  # fallback for consistency

            TEMP_PATH = OUTPUT_DIR_PATH / f"temp_{Path(INPUT_FILE_PATH).stem}.npy"
            np.save(TEMP_PATH, X)
            TEMP_PATHS.append(str(TEMP_PATH))
            all_indices.append(indices_new)

            if accumulate_stats:
            # Accumulate PCA stats
                T, p = X.shape
                total_T += T
                if first:
                    meanX = np.zeros(p)
                    sum_squares_X = np.zeros(p)
                    C = np.zeros((p, p))
                    first = False
                meanX += X.sum(axis=0)
                sum_squares_X += np.sum(X ** 2, axis=0)
                C += X.T @ X  # Gram matrix
        
        # Run PCA if requested
        if accumulate_stats:
            meanX /= total_T #  Global mean vector.
            varX = sum_squares_X / total_T - meanX ** 2 # Variance using E[X²] - (E[X])² identity
            stdX = np.sqrt(varX)
            stdX[stdX == 0] = 1  # Avoid division by zero for flat signals (std = 0); they won't be scaled during correlation normalization
            C = C / total_T - np.outer(meanX, meanX) # center the Gram matrix => covariance
            C = C / np.outer(stdX, stdX) # normalizes the covariance matrix => correlation matrix. 
        
        if pca is not None:
            pca_matrix, _ = highdim_pca(C, n_components=pca)

        # Save files with PCA/ICA applied
        OUTPUT_FILE_PATHS = []
        for path, INPUT_FILE_PATH, indices in zip(TEMP_PATHS, INPUT_FILE_PATHS, all_indices):
            X = np.load(path)
            if pca is not None:
                X = (X - meanX) / stdX
                X = X @ pca_matrix # Apply PCA
                if post_standardise:
                    X -= np.mean(X, axis=0)
                    X /= np.std(X, axis=0)
                log['meanX'] = meanX
                log['stdX'] = stdX
                log['pca_matrix'] = pca_matrix
    
            # Save the result regardless of PCA
            BASE_FILENAME = Path(INPUT_FILE_PATH).stem
            append_name = f"_{file_name}" if isinstance(file_name, str) else "_preprocessed"
            OUTPUT_FILE_PATH = OUTPUT_DIR_PATH / f"{BASE_FILENAME}{append_name}.npz"
            np.savez(OUTPUT_FILE_PATH, X=np.empty((0,)), Y=Y, indices=indices)
            OUTPUT_FILE_PATHS.append(str(OUTPUT_FILE_PATH))
            os.remove(path)

        log_suffix = f"_{file_name}" if isinstance(file_name, str) else ""
        log_file_path = OUTPUT_DIR_PATH / f"log_preprocessing{log_suffix}.npz"
        np.savez(log_file_path, **log)

        return OUTPUT_FILE_PATHS, log
   
    # In memory preprocessing
    p = data.shape[1]
    N = indices.shape[0]
    log = {**locals()}
    del(log["data"], log["indices"])

    data = np.copy(data)
    
    if dampen_extreme_peaks: 
        # center data first, per subject
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            data[t,:] -= np.mean(data[t,:],axis=0)
        # then dampen peaks at the group level    
        if isinstance(dampen_extreme_peaks,int):
            strength = dampen_extreme_peaks
        else:
            strength = 5
        data = dampen_peaks(data,strength)           
            
    if standardise:
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            data[t,:] -= np.mean(data[t,:],axis=0)
            data[t,:] /= np.std(data[t,:],axis=0)   

    if filter != None:
        filterorder = 6
        if filter[0] == 0: # low-pass
            sos = signal.butter(filterorder, filter[1], 'lowpass', output='sos', fs = fs)
        elif filter[1] == None: # high-pass
            sos = signal.butter(filterorder, filter[0], 'highpass', output='sos', fs = fs)
        else:
            sos = signal.butter(filterorder, filter, 'bandpass', output='sos', fs = fs)
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1])
            data[t,:] = signal.sosfilt(sos, data[t,:], axis=0)
    
    if detrend:
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            data[t,:] = signal.detrend(data[t,:], axis=0)       

    if onpower and not onphase:
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            data[t,:] = np.abs(signal.hilbert(data[t,:], axis=0))

    if onphase and not onpower:
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            data[t,:] = np.unwrap(np.angle(signal.hilbert(data[t,:], axis=0)))

    if onpower and onphase:
        data = np.concatenate((data,data),1)
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            analytical_signal = signal.hilbert(data[t,:p], axis=0)
            data[t,:p] = np.abs(analytical_signal)
            data[t,p:] = np.unwrap(np.angle(analytical_signal))
        p = data.shape[1]

    if autoregressive_order is not None:
        X, Y, indices, _ = build_data_autoregressive(
            data=data,
            indices=indices,
            autoregressive_order=autoregressive_order,
            center_data=post_standardise
        )
        data = Y  # Use Y as the transformed signal
    elif lags is not None:
        data, indices = build_data_tde(data, indices, lags)

    if (pca != None) and (ica is None):
        data, pcamodel = apply_pca(data,pca,exact_pca)
        p = data.shape[1]
        log["pcamodel"] = pcamodel

    if ica != None:
        data, icamodel = apply_ica(data,ica,ica_algorithm)
        p = data.shape[1]
        log["icamodel"] = icamodel       

    if post_standardise is None:
        if ica: post_standardise = True
        else: post_standardise = False

    if (pca or ica) and post_standardise:
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            data[t,:] -= np.mean(data[t,:],axis=0)
            data[t,:] /= np.std(data[t,:],axis=0)          
        
    if downsample != None:
        factor = downsample / fs
        Tnew = np.ceil(factor * (indices[:,1]-indices[:,0])).astype(int)
        indices_new = auxiliary.make_indices_from_T(Tnew)
        data_new = np.zeros((np.sum(Tnew),p))
        gcd = math.gcd(downsample,fs)
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            tnew = np.arange(indices_new[j,0],indices_new[j,1]) 
            data_new[tnew,:] = signal.resample_poly(data[t,:], downsample/gcd, fs/gcd)
            # Tjnew = tnew.shape[0]
            # data_new[tnew,:] = signal.resample(data[t,:], Tjnew)     
        data = data_new
    else: indices_new = indices

    return data,indices_new,log


def build_data_autoregressive(data,indices,autoregressive_order=1,
        connectivity=None,center_data=True,  files=None, output_dir=None, file_name=None):
    """
    Builds X and Y for the autoregressive model. Supports both in-memory and file-based input.
    Saves output when processing files.

    Parameters:
    -----------
    data : ndarray, shape (n_samples, n_parcels)
        In-memory time series data.
    indices : ndarray, shape (n_sessions, 2)
        Session boundaries in data.
    autoregressive_order : int
        Number of lags to include.
    connectivity : ndarray, optional
        Mask of shape (n_parcels, n_parcels).
    center_data : bool
        Whether to mean-center X and Y.
    files : list of str or Path, optional
        Input `.npz` or `.mat` files to process.
    output_dir : str or Path, optional
        Directory to save processed files.
    file_name : str, optional
        Custom suffix to append to each output file name.

    Returns:
    --------
    X : array-like of shape (n_samples - n_sessions*autoregressive_order, n_parcels*autoregressive_order)
        The timeseries of set of variables 1 (i.e., the regressors).
    Y : array-like of shape (n_samples - n_sessions*autoregressive_order, n_parcels)
        The timeseries of set of variables 2 (i.e., variables to predict, targets).
    indices_new : array-like of shape (n_sessions, 2)
        The new array of start and end indices for each trial/session.
    connectivity_new : array-like of shape (n_parcels*autoregressive_order, n_parcels)
        The new connectivity matrix indicating which regressors should be used for each variable.

    """
    if files is not None:
        output_paths = []
        log = {"autoregressive_order": autoregressive_order}

        if output_dir is None:
            output_dir = Path(files[0]).parent
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            _, data, indices = load_files([file], do_only_indices=False)
            X, Y, indices_new, conn_new = build_data_autoregressive(
                data=data,
                indices=indices,
                autoregressive_order=autoregressive_order,
                connectivity=connectivity,
                center_data=center_data,
                files=None
            )

            base_name = Path(file).stem
            suffix = f"_{file_name}" if isinstance(file_name, str) else "_ar"
            output_file = output_dir / f"{base_name}{suffix}.npz"

            # Save AR data: put Y as main, X optional (for legacy)
            np.savez(output_file, X=np.empty((0,)), Y=Y, indices=indices_new)
            output_paths.append(str(output_file))

        # Add optional metadata
        if connectivity is not None:
            log["connectivity_shape"] = connectivity.shape
        log["n_files"] = len(files)
        log["output_dir"] = str(output_dir)

        log_file = output_dir / f"log_ar{suffix}.npz"
        np.savez(log_file, **log)

        return output_paths, log

    else:
        T,p = data.shape
        N = indices.shape[0]

        if autoregressive_order == 0:
            warnings.warn("autoregressive_order is 0 so nothing to be done")
            return np.empty(0),data,indices,connectivity
        
        X = np.zeros((T - N*autoregressive_order,p*autoregressive_order))
        Y = np.zeros((T - N*autoregressive_order,p))
        indices_new = np.zeros((N,2))

        for j in range(N):
            ind_1 = np.arange(indices[j,0]+autoregressive_order,indices[j,1],dtype=np.int64)
            ind_2 = np.arange(indices[j,0],indices[j,1]-autoregressive_order,dtype=np.int64) \
                - j * autoregressive_order
            Y[ind_2,:] = data[ind_1,:]
            for i in range(autoregressive_order):
                ind_3 = np.arange(indices[j,0]+autoregressive_order-(i+1),indices[j,1]-(i+1),dtype=np.int64)
                ind_ch = np.arange(p) + i * p
                X[ind_2,ind_ch[:,np.newaxis]] = data[ind_3,:].T
            indices_new[j,0] = ind_2[0]
            indices_new[j,1] = ind_2[-1] + 1

        # center
        if center_data:
            Y -= np.mean(Y,axis=0)
            X -= np.mean(X,axis=0)

        if connectivity is not None:
            # connectivity_new : (regressors by regressed) 
            connectivity_new = np.zeros((autoregressive_order*p,p))
            for i in range(autoregressive_order):
                ind_ch = np.arange(p) + i * p
                connectivity_new[ind_ch,:] = connectivity
            # regress out when asked
            for j in range(p):
                jj = np.where(connectivity_new[:,j]==0)[0]
                if len(jj)==0: continue
                b = np.linalg.inv(X[:,jj].T @ X[:,jj] + 0.001 * np.eye(len(jj))) \
                    @ (X[:,jj].T @ Y[:,j])
                Y[:,j] -= X[:,jj] @ b
            # remove unused variables
            active_X = np.zeros(p,dtype=bool)
            active_Y = np.zeros(p,dtype=bool)
            for j in range(p):
                active_X[j] = np.sum(connectivity[j,:]==1) > 0
                active_Y[j] = np.sum(connectivity[:,j]==1) > 0
            active_X = np.tile(active_X,autoregressive_order)
            active_X = np.where(active_X)[0]
            active_Y = np.where(active_Y)[0]
            Y = Y[:,active_Y]
            X = X[:,active_X]
            connectivity_new = connectivity_new[active_X,active_Y[:,np.newaxis]].T
        else: connectivity_new = None

    return X,Y,indices_new,connectivity_new


def build_data_partial_connectivity(X,Y,connectivity=None,center_data=True):
    """Builds X and Y for the partial connectivity model, 
    essentially regressing out things when indicated in connectivity,
    and getting rid of regressors / regressed variables that are not used;
    it return connectivity with the right dimensions as well. 

    Parameters:
    -----------
    X : np.ndarray of shape (n_samples, n_parcels)
        The timeseries of set of variables 1 (i.e., the regressors).
    Y : np.ndarray of shape (n_samples, n_parcels)
        The timeseries of set of variables 2 (i.e., variables to predict, targets).
    connectivity : np.ndarray of shape (n_parcels, n_parcels), optional, default=None
        A binary matrix indicating which regressors affect which targets (i.e., variables to predict). 
    center_data : bool, default=True
        Center data to zero mean.

    Returns:
    --------
    X_new : np.ndarray of shape (n_samples, n_active_parcels)
        The timeseries of set of variables 1 (i.e., the regressors) after removing unused predictors and regressing out 
        the effects indicated in connectivity.
    Y_new : np.ndarray of shape (n_samples, n_active_parcels)
        The timeseries of set of variables 2 (i.e., variables to predict, targets) after removing unused targets and regressing out 
        the effects indicated in connectivity.
    connectivity_new : np.ndarray of shape (n_active_parcels, n_active_parcels), optional, default=None
        A binary matrix indicating which regressors affect which targets
        The matrix has the same structure as `connectivity` after removing unused predictors and targets.
    """

    X_new = np.copy(X)
    Y_new = np.copy(Y)

    if connectivity is not None:
        p = X.shape[1]
        q = Y.shape[1]
        # regress out when asked
        for j in range(q):
            jj = np.where(connectivity[:,j]==0)[0]
            if len(jj)==0: continue
            b = np.linalg.inv(X[:,jj].T @ X[:,jj] + 0.001 * np.eye(len(jj))) \
                @ (X[:,jj].T @ Y[:,j])
            Y_new[:,j] -= X[:,jj] @ b
        # remove unused variables
        active_X = np.zeros(p,dtype=bool)
        for j in range(p): active_X[j] = np.sum(connectivity[j,:]==1) > 0
        active_Y = np.zeros(q,dtype=bool)
        for j in range(q): active_Y[j] = np.sum(connectivity[:,j]==1) > 0
        active_X = np.where(active_X)[0]
        active_Y = np.where(active_Y)[0]
        Y = Y[:,active_Y]
        X = X[:,active_X]
        # copy of connectivity
        connectivity_new = np.copy(connectivity)
        connectivity_new = connectivity_new[active_X,active_Y[:,np.newaxis]].T
    else: connectivity_new = None

    # center
    if center_data:
        Y_new = Y_new - np.mean(Y_new,axis=0)
        X_new = X_new - np.mean(X_new,axis=0)

    return X_new,Y_new,connectivity_new


def build_data_tde(data=None, indices=None, lags=None, pca=None, standardise_pc=True, files=None, output_dir=None, file_name=None):
    """
    Builds delay-embedded data for TDE-HMM. Supports in-memory or file-based input.

    Parameters:
    -----------
    data : ndarray or None
        Raw data (n_samples, n_parcels) to embed in memory.
    indices : ndarray or None
        Start and end indices for each session (n_sessions, 2).
    lags : list or array-like
        Lags to apply for temporal embedding.
    pca : int, float, array or None
        PCA options.
    standardise_pc : bool
        Whether to standardise PCA components.
    files : list of str or Path, optional
        If set, reads files instead of using `data`/`indices`.
    output_dir : str or Path, optional
        Where to save output files if using file input.
    file_name : str or None, optional
        Custom string to append to each output file name before extension.

    Returns:
    --------
    If using files: list of output file paths, and log dictionary if PCA is applied.
    If using in-memory data: X_emb, indices_emb (+ pcamodel if PCA).
    """

    if files is not None:
        OUTPUT_FILE_PATHS = []
        # Determine output directory
        if output_dir is None:
            output_dir = Path(files[0]).parent
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # accumulate statistics for global PCA 
        if pca is not None:
            first = True
            total_T = 0
            for file in files:
                _, data, indices = load_files([file], do_only_indices=False)
                T, p = data.shape
                N = indices.shape[0]
                L = len(lags)
                minlag, maxlag = np.min(lags), np.max(lags)
                rwindow = maxlag - minlag
                # Apply delay embedding to each file individually
                X = np.zeros((T - N * rwindow, p * L))
                for j in range(N):
                    ind_1 = np.arange(indices[j, 0], indices[j, 1])
                    ind_2 = np.arange(indices[j, 0], indices[j, 1] - rwindow) - j * rwindow
                    for i, l in enumerate(lags):
                        X_l = np.roll(data[ind_1], l, axis=0)
                        X_l = X_l[-minlag:-maxlag]
                        X[ind_2, i * p:(i + 1) * p] = X_l

                # Initialize accumulators on the first file
                if first:
                    D = X.shape[1]
                    sum_X = np.zeros(D)
                    sumsq_X = np.zeros(D)
                    C = np.zeros((D, D))
                    first = False
                 # Accumulate total samples, means, and covariance
                total_T += X.shape[0]
                sum_X += X.sum(axis=0)
                sumsq_X += np.einsum("ij,ij->j", X, X)
                C += X.T @ X
            # Compute global mean and standard deviation
            meanX = sum_X / total_T
            varX = sumsq_X / total_T - meanX ** 2
            stdX = np.sqrt(varX)
            stdX[stdX == 0] = 1
            # Normalize the covariance matrix to compute correlation
            C = C / total_T - np.outer(meanX, meanX)
            C = C / np.outer(stdX, stdX)

            # Compute PCA matrix using updated highdim_pca
            pca_matrix, _ = highdim_pca(C, n_components=pca)

        # apply TDE and shared PCA 
        for file in files:
            _, data, indices = load_files([file], do_only_indices=False)
            T, p = data.shape
            N = indices.shape[0]
            L = len(lags)
            minlag, maxlag = np.min(lags), np.max(lags)
            rwindow = maxlag - minlag

            X = np.zeros((T - N * rwindow, p * L))
            indices_new = np.zeros((N, 2), dtype=int)

            for j in range(N):
                ind_1 = np.arange(indices[j, 0], indices[j, 1])
                ind_2 = np.arange(indices[j, 0], indices[j, 1] - rwindow) - j * rwindow
                for i, l in enumerate(lags):
                    X_l = np.roll(data[ind_1], l, axis=0)
                    X_l = X_l[-minlag:-maxlag]
                    X[ind_2, i * p:(i + 1) * p] = X_l
                indices_new[j] = [ind_2[0], ind_2[-1] + 1]

            X -= np.mean(X, axis=0)
            X /= np.std(X, axis=0)

            if pca is not None:
                X = (X - meanX) / stdX
                X = X @ pca_matrix
                if standardise_pc:
                    X /= np.std(X, axis=0)

            # Save output for this file
            base_filename = Path(file).stem
            append_name = f"_{file_name}" if isinstance(file_name, str) else ("_pca_tde" if pca is not None else "_tde")
            output_file = output_dir / f"{base_filename}{append_name}.npz"
            np.savez(output_file, X=np.empty((0,)), Y=X, indices=indices_new)
            OUTPUT_FILE_PATHS.append(str(output_file))

        # Prepare log for reproducibility
        log = {"n_lags": len(lags)}
        if pca is not None:
            log.update({
                "meanX": meanX,
                "stdX": stdX,
                "pca_matrix": pca_matrix,
                "n_components": pca_matrix.shape[1]
            })

        # Save log
        log_suffix = f"_{file_name}" if isinstance(file_name, str) else ""
        log_file_path = output_dir / f"log_tde{log_suffix}.npz"
        np.savez(log_file_path, **log)

        return OUTPUT_FILE_PATHS, log


    # In-memory mode
    T, p = data.shape
    N = indices.shape[0]

    L = len(lags)
    minlag = np.min(lags)
    maxlag = np.max(lags)
    rwindow = maxlag - minlag

    X = np.zeros((T - N * rwindow, p * L))
    indices_new = np.zeros((N, 2), dtype=int)

    for j in range(N):
        ind_1 = np.arange(indices[j, 0], indices[j, 1], dtype=np.int64)
        ind_2 = np.arange(indices[j, 0], indices[j, 1] - rwindow, dtype=np.int64) - j * rwindow
        for i in range(L):
            l = lags[i]
            X_l = np.roll(data[ind_1, :], l, axis=0)
            X_l = X_l[-minlag:-maxlag, :]
            ind_ch = np.arange(i, L * p, L)
            X[ind_2, ind_ch[:, np.newaxis]] = X_l.T
        indices_new[j, 0] = ind_2[0]
        indices_new[j, 1] = ind_2[-1] + 1

    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    if pca is not None:
        X, pcamodel = apply_pca(X, pca, standardise_pc)
        return X, indices_new, pcamodel
    else:
        return X, indices_new




def load_files(files,I=None,do_only_indices=False):        
    # Convert Path objects to strings if needed
    files = [str(f) if isinstance(f, Path) else f for f in files]
    X = []
    Y = []
    indices = []
    sum_T = 0

    if I is None:
        I = np.arange(len(files))
    elif type(I) is int:
        I = np.array([I])

    for ij in range(I.shape[0]):

        j = I[ij]

        # if type(files[j]) is tuple:
        #     if len(files[j][0]) > 0: # X
        #         if files[j][0][-4:] == '.npy':
        #             X.append(np.load(files[j][0]))
        #         elif files[j][0][-4:] == '.txt':

        if files[j][-4:] == '.mat':
            dat = scipy.io.loadmat(files[j])

        elif files[j][-4:] == '.npz':
            dat = np.load(files[j])
            
        if not do_only_indices:
            if ('X' in dat) and (not 'Y' in dat): 
                Y.append(dat["X"])
            else:
                if 'X' in dat: X.append(dat["X"])
                Y.append(dat["Y"])
        if 'indices' in dat: 
            indices.append(dat['indices'])
        elif 'T' in dat:
            indices.append(auxiliary.make_indices_from_T(dat['T']) + sum_T)
        else:
            ind = np.zeros((1,2)).astype(int)
            ind[0,0] = 0
            ind[0,1] = Y[-1].shape[0]
            indices.append(ind + sum_T)

        sum_T += dat["Y"].shape[0]

    if not do_only_indices:
        if len(X) > 0: X = np.concatenate(X)
        Y = np.concatenate(Y)
    indices = np.concatenate(indices)
    if len(indices.shape) == 1: indices = np.expand_dims(indices,axis=0)
    if len(X) == 0: X = None

    return X,Y,indices