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
    for j in range(d):
        jj = np.where(np.abs(X[:,j]) == np.abs(np.max(X[:,j])) )[0][0]
        if X[jj,j] < 0: X[:,j] *= -1

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

    return eigvecs, eigvals


def preprocess_data(data = None,indices = None,
        fs = 1, # frequency of the data
        dampen_extreme_peaks=None, # it can be None, True, or an int with the strength of dampening
        standardise=True, # True / False
        filter=None, # Tuple with low-pass high-pass thresholds, or None
        notch_filter = None,
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
        file_type= "npy"
        ):
    
    """
    Preprocess the input data or files.

    Parameters:
    -----------
    data : array-like or list of file paths
        If array-like of shape (n_samples, n_parcels), the raw input data to be preprocessed in memory.
        If list of file paths, raw data files to be preprocessed individually.
    indices : array-like of shape (n_sessions, 2), optional
        The start and end indices of each trial/session in the input data (only used when data is array-like).
    fs : int or float, default=1
        Sampling frequency of the input data.
    notch_filter: Base notch frequency (e.g., 50 or 60 Hz)
    dampen_extreme_peaks : int, True, or None, default=None
        Whether to dampen extreme peaks in the data. 
        If int, specifies the dampening strength.
        If True, default strength of 5 is used.
        If None, no dampening is applied.
    standardise : bool, default=True
        Whether to standardize the input data (zero mean, unit variance).
    filter : tuple of length 2 or None, default=None
        Filtering thresholds. 
        If tuple, (low-pass, high-pass) values.
        If None, no filtering is applied.
    detrend : bool, default=False
        Whether to detrend the input data.
    onpower : bool, default=False
        Whether to calculate signal power using the Hilbert transform.
    onphase : bool, default=False
        Whether to calculate signal phase using the Hilbert transform.
        If both `onpower` and `onphase` are True, power and phase are concatenated.
    pca : int, float, or None, default=None
        If int, number of PCA components to retain.
        If float, proportion of explained variance to retain.
        If None, no PCA is applied.
    exact_pca : bool, default=True
        Whether to use full SVD for PCA (only relevant for in-memory mode).
    ica : int, float, or None, default=None
        Whether to apply ICA instead of PCA.
        If int, number of independent components.
        If float, proportion of explained variance.
        If None, no ICA is applied.
    ica_algorithm : {'parallel', 'deflation'}, default='parallel'
        ICA algorithm to use.
    post_standardise : bool or None, default=None
        Whether to standardize again after PCA/ICA. 
        Defaults to True if ICA is used.
    downsample : int, float, or None, default=None
        New sampling frequency if downsampling.
        If None, no downsampling is applied.
    output_folder : str or Path, optional
        If data is file-based, directory where preprocessed files are saved. 
        If None, files are saved in the same folder as input files.

    Returns:
    --------
    If input is an array:
        data : array-like of shape (n_samples_processed, n_parcels)
            Preprocessed data.    
        indices_new : array-like of shape (n_sessions_processed, 2)
            Updated indices after processing.
        log : dict
            Information about the preprocessing steps and any fitted models (e.g., PCA, ICA).

    If input is a list of files:
        output_file_paths : list of str
            Paths to saved preprocessed files.
        log : dict
            Dictionary containing preprocessing statistics (mean, std, PCA matrix, etc.) used.
    """

    # New mode: file-based preprocessing for stochastic learning
    if files is not None:
        files = resolve_files(files, file_type=file_type)
        INPUT_FILE_PATHS = [str(p) for p in files]
        if output_dir is None:
            OUTPUT_DIR_PATH = Path(files[0]).parent
        else:
            OUTPUT_DIR_PATH = output_dir
        OUTPUT_DIR_PATH = Path(OUTPUT_DIR_PATH)
        OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)

        first = True
        total_T = 0
        # collect aggregated statistics - One file at the time
        for INPUT_FILE_PATH in INPUT_FILE_PATHS:
            X = load_X(INPUT_FILE_PATH)
            T, p = X.shape
            total_T += T
            if first:
                meanX = np.zeros(p)
                sum_squares_X = np.zeros(p)
                C = np.zeros((p, p))
                first = False
            meanX += X.sum(axis=0)
            sum_squares_X += np.einsum("ij,ij->j", X, X)
            C += X.T @ X  # Gram matrix.
            del X

        meanX /= total_T #  Global mean vector.
        varX = sum_squares_X / total_T - meanX ** 2 # Variance using E[X²] - (E[X])² identity
        stdX = np.sqrt(varX)
        # Avoid division by zero for flat signals (std = 0); they won't be scaled during correlation normalization
        stdX[stdX == 0] = 1

        C = C / total_T - np.outer(meanX, meanX) # center the Gram matrix => covariance
        C = C / np.outer(stdX, stdX) # normalizes the covariance matrix => correlation matrix.
        pca_matrix, _ = highdim_pca(C, n_components=pca) # run PCA on that

        OUTPUT_FILE_PATHS = []

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

            if notch_filter is not None:
                for freq in notch_filter:
                    b, a = signal.iirnotch(freq, Q=30, fs=fs)
                    X = signal.lfilter(b, a, X, axis=0)

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

            if pca is not None:
                X = (X - meanX) / stdX
                X = X @ pca_matrix # Project onto shared PCA basis

            if ica is not None:
                X, icamodel = apply_ica(X, ica, ica_algorithm)
                p = X.shape[1]

            if post_standardise is None:
                post_standardise = bool(ica)

            if (pca or ica) and post_standardise:
                X -= np.mean(X, axis=0)
                X /= np.std(X, axis=0)

            if downsample is not None:
                factor = downsample / fs
                indices_new = np.array([[0, int(np.ceil(T * factor))]])
                gcd = math.gcd(int(downsample), int(fs))
                X = signal.resample_poly(X, int(downsample // gcd), int(fs // gcd), axis=0)
                indices = indices_new

            BASE_FILENAME = Path(INPUT_FILE_PATH).stem
            append_name = f"_{file_name}" if isinstance(file_name, str) else "_preprocessed"
            OUTPUT_FILE_PATH = OUTPUT_DIR_PATH / f"{BASE_FILENAME}{append_name}.npz"
            np.savez(OUTPUT_FILE_PATH, X=np.empty((0,)), Y=X, indices=indices)
            OUTPUT_FILE_PATHS.append(str(OUTPUT_FILE_PATH))

        log = {
            "meanX": meanX,
            "stdX": stdX,
            "pca_matrix": pca_matrix,
            "total_T": total_T,
            "n_components": pca_matrix.shape[1]
        }
        
        # Save the log file in the same output directory
        log_suffix = f"_{file_name}" if isinstance(file_name, str) else ""
        log_file_path = OUTPUT_DIR_PATH / f"log_preprocessing_{log_suffix}.npz"
        np.savez(log_file_path, **log)
        return OUTPUT_FILE_PATHS, log

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
        connectivity=None,center_data=True):
    """Builds X and Y for the autoregressive model, 
    as well as an adapted indices array and predefined connectivity 
    matrix in the right format. X and Y are centered by default.
    
    Parameters:
    -----------
    data : array-like of shape (n_samples,n_parcels)
        The data timeseries.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.
    autoregressive_order : int, optional, default=1
        The number of lags to include in the autoregressive model.
    connectivity : array-like of shape (n_parcels, n_parcels), optional, default=None
        The matrix indicating which regressors should be used for each variable.
    center_data : bool, optional, default=True
        If True, the data will be centered.

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

    else:
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
