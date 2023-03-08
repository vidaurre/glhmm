#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing functions - General/Gaussian Linear Hidden Markov Model
@author: Diego Vidaurre 2023
"""

import numpy as np
import warnings
from sklearn.decomposition import PCA
from scipy import signal

from . import auxiliary

def apply_pca(X,d,whitening=False,exact=True):

    if type(d) is np.ndarray:
        X -= np.mean(X,axis=0)
        X = X @ d
        if whitening: X /= np.std(X,axis=0)
        return X

    if exact: svd_solver = 'full'
    if d >= 1: 
        pcamodel = PCA(n_components=d,whiten=whitening,svd_solver=svd_solver)
        pcamodel.fit(X)
        X = pcamodel.transform(X)
    else: 
        pcamodel = PCA(whiten=whitening,svd_solver=svd_solver)
        pcamodel.fit(X)
        ncomp = np.where(np.cumsum(pcamodel.explained_variance_ratio_)>=d)[0][0] + 1
        X = pcamodel.transform(X)
        X = X[:,0:ncomp]
        d = ncomp

    # sign convention equal to Matlab's
    for j in range(d):
        jj = np.where(np.abs(X[:,j]) == np.abs(np.max(X[:,j])) )[0][0]
        if X[jj,j] < 0: X[:,j] *= -1

    return X


def preprocess_data(data,indices,
        fs = 1, # frequency of the data
        standardise=True, # True / False
        filter=None, # Tuple with low-pass high-pass thresholds, or None
        detrend=True, # True / False
        onpower=False, # True / False
        pca=None, # Number of components, % explained variance, or None
        whitening=False, # True / False
        exact_pca=True,
        downsample=None # new frequency, or None
        ):

    p = data.shape[1]
    N = indices.shape[0]

    data = np.copy(data)

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

    if onpower:
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            data[t,:] = np.abs(signal.hilbert(data[t,:], axis=0))

    if pca != None:
        data = apply_pca(data,pca,whitening,exact_pca)
        
    if downsample != None:
        factor = downsample / fs
        Tnew = np.ceil(factor * (indices[:,1]-indices[:,0])).astype(int)
        indices_new = auxiliary.make_indices_from_T(Tnew)
        data_new = np.zeros((np.sum(Tnew),p))
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            tnew = np.arange(indices_new[j,0],indices_new[j,1]) 
            Tjnew = tnew.shape[0]
            data_new[tnew,:] = signal.resample(data[t,:], Tjnew, axis=0)     
        data = data_new
    else: indices_new = indices

    return data,indices_new


def build_data_autoregressive(data,indices,autoregressive_order=1,
        connectivity=None,center_data=True):
    """
    Build X and Y for the autoregressive model, 
    as well as an adapted indices array and predefined connectivity 
    matrix in the right format.
    X and Y are centered by default. 
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
    """
    Build X and Y for the partial connectivity model, 
    essentially regressing out things when indicated in connectivity,
    and getting rid of regressors / regressed variables that are not used;
    it return connectivity with the right dimensions as well. 
    X and Y are centered by default. 
    The returned X, Y and connectivity are new copies (not links)
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


def build_data_tde(data,indices,lags,pca=None,standardise_pc=True):
    """
    Build X for the temporal delay embedded HMM, as well as an adapted indices array
    X is centered by default. 
    PCA can be run optionally: if pca >=1, that is the number of components;
    if pca < 1, that is explained variance;
    if pca is a numpy array, then it is a precomputed PCA projection matrix;
    if pca is None, then no PCA is run.
    """

    T,p = data.shape
    N = indices.shape[0]
    
    L = len(lags)
    minlag = np.min(lags)
    maxlag = np.max(lags)
    rwindow = maxlag-minlag

    X = np.zeros((T - N*rwindow,p*L))
    indices_new = np.zeros((N,2)).astype(int)

    # Embedding
    for j in range(N):
        ind_1 = np.arange(indices[j,0],indices[j,1],dtype=np.int64)
        ind_2 = np.arange(indices[j,0],indices[j,1]-rwindow,dtype=np.int64) - j * rwindow
        for i in range(L):
            l = lags[i]
            X_l = np.roll(data[ind_1,:],l,axis=0)
            X_l = X_l[-minlag:-maxlag,:]
            ind_ch = np.arange(i,L*p,L)
            X[ind_2,ind_ch[:,np.newaxis]] = X_l.T
        indices_new[j,0] = ind_2[0]
        indices_new[j,1] = ind_2[-1] + 1

    # Standardise (in Matlab's HMM-MAR we only centered pre-embedding)
    # note that this is done for the entire data set and not per sessions
    X -= np.mean(X,axis=0)
    X /= np.std(X,axis=0)

    # PCA and whitening 
    if pca is not None:
        X = apply_pca(X,pca,standardise_pc)

    return X,indices_new


def load_files(files,I=None,do_only_indices=False):        

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
            indices.append(make_indices_from_T(dat['T']) + sum_T)
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


