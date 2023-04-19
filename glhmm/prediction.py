#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kernel prediction from Gaussian Linear Hidden Markov Model
@author: Christine Ahrends 2023
"""

import numpy as np
import sys
from . import glhmm


def compute_gradient(hmm, Y, incl_Pi=True, incl_P=True, incl_Mu=False, incl_Sigma=False):
    """Computes the gradient of the log-likelihood for timeseries Y
    with respect to specified HMM parameters
    
    Parameters:
    -----------
    hmm : HMM object
        An instance of the HMM class, estimated on the group-level
    Y : array-like of shape (n_samples, n_variables_2)
        (subject- or session-level) timeseries data
    incl_Pi : bool, default=True
        whether to compute gradient w.r.t state probabilities
    incl_P : bool, default=True
        whether to compute gradient w.r.t. transition probabilities
    incl_Mu : bool, default=False
        whether to compute gradient w.r.t state means
        (only possible if state means were estimated during training)
    incl_Sigma : bool, default=False
        whether to compute gradient w.r.t. state covariances
        (for now only for full covariance matrix)

    Returns:
    --------
    hmmgrad : array of shape (sum(len(requested_parameters)))

    Raises:
    -------
    Exception
        If the model has not been trained or if requested parameters do not exist
        (e.g. if Mu is requested but state means were not estimated)

    Note:
    ----
    Does not include gradient computation for X and beta

    """

    if not hmm.trained:
        raise Exception("The model has not yet been trained")

    Pi = hmm.Pi
    P = hmm.P
    K = hmm.hyperparameters["K"] # number of states
    q = Y.shape[1] # number of variables (regions/parcels)

    subHMM, subGamma, subXi_tmp = hmm.dual_estimate(X=None, Y=Y, for_kernel=True)
    GammaT = subGamma.transpose()
    subXi = np.mean(subXi_tmp, axis=0)

    Xisum = np.sum(subXi, axis=1)
    Xisumreal = Xisum
    # make sure that values are not getting too close to 0
    for i in range(K):
        Xisumreal[i] = max(Xisum[i], sys.float_info.min)

    XiT = subXi.transpose()
    Xi_tmp = XiT / Xisumreal
    Xi = Xi_tmp.transpose()

    # gradient w.r.t. state prior:
    if incl_Pi:
        Pireal = Pi
        # make sure that values are not getting too close to 0
        for i in range(K):
            Pireal[i] = max(Pi[i], sys.float_info.min)
        
        dPi = GammaT[:,0] / Pireal

    # gradient w.r.t. transition probabilities:
    if incl_P:
        Preal = P
        # make sure that values are not getting too close to 0
        for i in range(K):
            for j in range(K):
                Preal[i,j] = max(P[i,j], sys.float_info.min)
    
        dP = Xi / Preal

    if (incl_Mu) or (incl_Sigma):
        Sigma = np.zeros(shape=(q, q, K))
        invSigma = np.zeros(shape=(q, q, K))
        for k in range(K):
            Sigma[:,:,k] = hmm.get_covariance_matrix(k=k)
            invSigma[:,:,k] = hmm.get_inverse_covariance_matrix(k=k)

        # gradient w.r.t. state means
        if incl_Mu:
            if hmm.hyperparameters["model_mean"] == 'no':
                raise Exception("Cannot compute gradient w.r.t state means because state means were not modelled")
            
            Mu = hmm.get_means()
            dMu = np.zeros(shape=(q, K))
            for k in range(K):
                Xi_V = Y - Mu[:,k]
                Xi_VT = Xi_V.transpose()
                iSigmacond = np.matmul(invSigma[:,:,k], Xi_VT)
                GamiSigmacond = GammaT[k,:]*iSigmacond
                dMu[:,k] = np.sum(GamiSigmacond, axis=1)
        
        # gradient w.r.t. state covariances
        if incl_Sigma:
            dSigma = np.zeros(shape=(q, q, K))
            YT = Y.transpose()
            for k in range(K):
                Xi_V = Y- Mu[:,k]
                Xi_VT = Xi_V.transpose()
                Gamma_tmp = -sum(GammaT[k,:]/2)
                GamiSigma = Gamma_tmp * invSigma[:,:,k]
                GamXi = GammaT[k,:] * Xi_VT
                GamXi2 = np.matmul(GamXi, Xi_V)
                iSigmaGamXi = 0.5 * np.matmul(invSigma[:,:,k], GamXi2)
                iSigmaGamXi2 = np.matmul(iSigmaGamXi, invSigma[:,:,k])
                dSigma[:,:,k] = GamiSigma + iSigmaGamXi2

    hmmgrad = np.empty(shape=0)

    if incl_Pi:
        hmmgrad = -dPi

    if incl_P:
        dP_flat = np.ndarray.flatten(dP)
        hmmgrad = np.concatenate((hmmgrad, -dP_flat))

    if incl_Mu:
        dMu_flat = np.ndarray.flatten(dMu)
        hmmgrad = np.concatenate((hmmgrad, -dMu_flat))
    
    if incl_Sigma:
        dSigma_flat = np.ndarray.flatten(dSigma)
        hmmgrad = np.concatenate((hmmgrad, -dSigma_flat))
        
    return hmmgrad

def hmm_kernel(hmm, Y, indices, type='Fisher', shape='linear', incl_Pi=True, incl_P=True, incl_Mu=False, incl_Sigma=False, tau=None, return_feat=False, return_dist=False):
    """Constructs a kernel from an HMM, as well as the respective feature matrix 
    and/or distance matrix
    
    Parameters:
    -----------
    hmm : HMM object
        An instance of the HMM class, estimated on the group-level
    Y : array-like of shape (n_samples, n_variables_2)
        (group-level) timeseries data
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data. 
        Note that kernel cannot be computed if indices=None
    type : str, optional
        The type of kernel to be constructed
        (default: 'Fisher')
    shape : str, optional
        The shape of kernel to be constructed, either 'linear' or 'Gaussian'
        (default: 'linear')
    incl_Pi : bool, default=True
        whether to include state probabilities in kernel construction
    incl_P : bool, default=True
        whether to include transition probabilities in kernel construction
    incl_Mu : bool, default=False
        whether to include state means in kernel construction
        (only possible if state means were estimated during training)
    incl_Sigma : bool, default=False
        whether to include state covariances in kernel construction
        (for now only for full covariance matrix)
    return_feat : bool, default=False
        whether to return also the feature matrix
    return_dist : bool, default=False
        whether to return also the distance matrix

    Returns:
    --------
    kernel : array of shape (n_sessions, n_sessions)
        HMM Kernel for subjects/sessions contained in Y
    feat : array of shape (n_sessions, sum(len(requested_parameters)))
        Feature matrix for subjects/sessions contained in Y for requested parameters
    dist : array of shape (n_sessions, n_sessions)
        Distance matrix for subjects/sessions contained in Y

    Raises:
    -------
    Exception
        If the model has not been trained or if requested parameters do not exist
        (e.g. if Mu is requested but state means were not estimated)
        If kernel other than Fisher kernel is requested

    Note:
    ----
    Does not include X and beta in kernel construction
    Only Fisher kernel implemented at this point
    """

    S = indices.shape[0]
    K = hmm.hyperparameters["K"]
    q = Y.shape[1]
    
    feat_dim = incl_Pi*K + incl_P*K*K + incl_Mu*K*q + incl_Sigma*K*q*q
    feat = np.zeros(shape=(S,feat_dim))
    for s in range(S):
        Y_s = Y[indices[s,0]:indices[s,1],:]
        if type=='Fisher':
            feat[s,:] = compute_gradient(hmm=hmm, Y=Y_s, incl_Pi=incl_Pi, incl_P=incl_P, incl_Mu=incl_Mu, incl_Sigma=incl_Sigma)
        else:
            raise Exception("This kernel is not yet implemented. Use type='Fisher' instead")


    if shape=='linear':
        featT = feat.transpose()
        kernel = feat @ featT # inner product

    if shape=='Gaussian':
        dist = np.zeros(shape=(S,S))
        for i in range(S):
            for j in range(S):
                dist[i,j] = np.sqrt(sum(abs(feat[i,:]-feat[j,:])**2))**2
            
            
        if not tau:
            tau = 1
        
        kernel = np.exp(-dist/(2*tau**2))

    if return_feat and not return_dist:
        return kernel, feat
    elif return_dist and not return_feat:
        return kernel, dist 
    elif return_feat and return_dist:
        return kernel, feat, dist
    else:   
        return kernel
        