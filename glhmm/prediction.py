#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kernel prediction from Gaussian Linear Hidden Markov Model
@author: Christine Ahrends 2023
"""

import numpy as np
import sys
import sklearn
import igraph as ig
from . import glhmm


def compute_gradient(hmm, Y, incl_Pi=True, incl_P=True, incl_Mu=True, incl_Sigma=True):
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
    if not hmm.trained:
        raise Exception("The model has not yet been trained")

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
        if shape=='linear':
            raise Exception("Distance matrix is not defined for linear kernel")
        else:
            return kernel, dist 
    elif return_feat and return_dist:
        if shape=='linear':
            raise Exception("Distance matrix is not defined for linear kernel")
        else:
            return kernel, feat, dist
    else:   
        return kernel
    
def get_summ_features(hmm, Y, indices, metrics):
    # Note: lifetimes uses the mean lifetimes (change code if you want to use median or max lifetimes instead)
    if not set(metrics).issubset(['FO', 'switching_rate', 'lifetimes']):
            raise Exception('Requested summary metrics are not recognised. Use one or more of FO, switching rate, or lifetimes (for now)')

    features_tmp = np.zeros(shape=(indices.shape[0],1))

    # get Gamma or vpath from hmm:
    if 'FO' in metrics or 'switching_rate' in metrics:
        Gamma,_,_ = hmm.decode(X=None, Y=Y, indices=indices, viterbi=False)
    
    if 'lifetimes' in metrics:
        vpath = hmm.decode(X=None, Y=Y, indices=indices, viterbi=True)

    if 'FO' in metrics:
        FO = glhmm.utils.get_FO(Gamma, indices)
        features_tmp = np.append(features_tmp, FO, axis=1)
    if 'switching_rate' in metrics:
        SR = glhmm.utils.get_switching_rate(Gamma, indices)
        features_tmp = np.append(features_tmp, SR, axis=1)
    if 'lifetimes' in metrics:
        LT,_,_ = glhmm.utils.get_life_times(vpath, indices)
        features_tmp = np.append(features_tmp, LT, axis=1)

    features = features_tmp[:,1:]

    return features
    
def get_groups(group_structure):
    # make sure diagonal if family matrix is 1:
    cs2 = group_structure
    np.fill_diagonal(cs2, 1)
    # create adjacency graph
    csG = ig.Graph.Adjacency(cs2)
    # get groups (connected components in adjacency matrix)
    groups = csG.connected_components()
    cs_tmp = groups.membership
    cs = np.asarray(cs_tmp)

    return cs
        
def predictPhenotype(hmm, Y, behav, indices, method='Fisherkernel', estimator='KernelRidge', options=None):
    """Predict phenotype from HMM
    
    Parameters:
    -----------
    hmm : trained (group-level) HMM
    Y : (group) timeseries
    behav : phenotype to be predicted
    indices : indices indicating beginning and end of each subject's timeseries
    method : either 'Fisherkernel' or 'summary_metrics' (default='Fisherkernel')
    estimator : sklearn estimator to be used for prediction (default='KernelRidge')
    options :

    Returns:
    --------
    behav_pred : predicted phenotype
    
    """
    if not hmm.trained:
        raise Exception("The model has not yet been trained")

    if behav is None:
        raise Exception("Phenotype to be predicted needs to be provided")
    
    if indices is None: 
        raise Exception("To predict phenotype from HMM, indices need to be provided")
    
    if options is None: 
        options = {}
    else:
        if method=='Fisherkernel':
            if not options['shape']:
                shape='linear' 
            else:
                shape=options['shape']
            if options['incl_Pi']:
                incl_Pi = options['incl_Pi']
            else:
                incl_Pi = True
            if options['incl_P']:
                incl_P = options['incl_P']
            else:
                incl_P = True
            if options['incl_Mu']:
                incl_Mu = options['incl_Mu']
            else:
                incl_Mu = True
            if options['incl_Sigma']:
                incl_Sigma = options['incl_Sigma']:
            else:
                incl_Sigma = True
        
        if method=='summary':
            if not options['metrics']:
                metrics = ['FO', 'switching_rate', 'lifetimes']
            else:
                metrics = options['metrics']

        if not options['CVscheme']:
            CVscheme = 'KFold'
        else:
            CVscheme = options['CVscheme']

        if not options['nfolds']:
            nfolds = 5
        else:
            nfolds = options['nfolds']

        if not options['family_structure']:
            do_groupKFold = False
            allcs = None
        else:
            do_groupKFold = True
            allcs = options['family_structure']
            CVscheme = 'GroupKFold'

        if not options['confounds']:
            confounds = None
        else:
            confounds = options['confounds']

    N = indices.shape[0] # number of samples

    # get features/kernel from HMM to be predicted from (default: Fisher kernel):
    if method=='Fisherkernel':
        if shape=='linear':
            tau=None
        elif shape=='Gaussian':
            tau=options['tau']
        Xin = hmm_kernel(hmm, Y, indices, type='Fisher', shape=shape, incl_Pi=incl_Pi, incl_P=incl_P, incl_Mu=incl_Mu, incl_Sigma=incl_Sigma, tau=tau, return_feat=False, return_dist=False)
    # alternative: predict from HMM summary metrics
    elif method=='summary':
        Xin = get_summ_features(hmm, Y, indices, metrics)  
            
    # create CV folds
    if do_groupKFold: # when using family/group structure - use GroupKFold
        cs = get_groups(allcs)
        cvfolds = sklearn.model_selection.GroupKFold(n_splits=nfolds)
        cvfolds.get_n_splits(Y, behav, cs)
    elif CVscheme=='KFold': # when not using family/group structure
        cvfolds = sklearn.model_selection.KFold(n_splits=nfolds)
        cvfolds.get_n_splits(Y, behav)

    behav_pred = np.zeros(shape=N)

    if estimator=='KernelRidge':
        if not options['alpha']:
            alphas = np.logspace(-4, -1, 6)
        else:
            alphas = options['alpha']
        
        model = sklearn.kernel_ridge.KernelRidge(kernel="precomputed")
        for train, test in cvfolds.split(Xin, behav, groups=cs):
            model_tuned = sklearn.model_selection.GridSearchCV(estimator=model, param_grid=dict(alpha=alphas), cv=cvfolds)
            model_tuned.fit(Xin[train, train.reshape(-1,1)], behav[train], groups=cs[train])
            behav_pred[test] = model_tuned.predict(Xin[train, test.reshape(-1,1)])

    return behav_pred
