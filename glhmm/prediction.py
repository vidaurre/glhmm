#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction from Gaussian Linear Hidden Markov Model
@author: Christine Ahrends 2023
"""

import numpy as np
import sys
from sklearn import preprocessing, model_selection, kernel_ridge, linear_model, svm
from sklearn import metrics as ms
import igraph as ig
from . import glhmm, utils


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

    Notes:
    ------
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

    Notes:
    ------
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
    """Util function to get summary features from HMM. 
    Output can be used as input features for ML

    Parameters:
    -----------
    hmm : HMM object
        An instance of the HMM class, estimated on the group-level
    Y : array-like of shape (n_samples, n_variables_2)
        (group-level) timeseries data
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data. 
        Note that kernel cannot be computed if indices=None  
    metrics : list
        names of metrics to be extracted. For now, this should be one or more 
        of 'FO', 'switching_rate', 'lifetimes'    

    Returns:
    --------
    features : array-like of shape (n_samples, n_features)
        The HMM summary metrics collected into a feature matrix

    """

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
        FO = utils.get_FO(Gamma, indices)
        features_tmp = np.append(features_tmp, FO, axis=1)
    if 'switching_rate' in metrics:
        SR = utils.get_switching_rate(Gamma, indices)
        features_tmp = np.append(features_tmp, SR, axis=1)
    if 'lifetimes' in metrics:
        LT,_,_ = utils.get_life_times(vpath, indices)
        features_tmp = np.append(features_tmp, LT, axis=1)

    features = features_tmp[:,1:]

    return features
    
def get_groups(group_structure):
    """Util function to get groups from group structure matrix such as family structure. 
    Output can be used to make sure groups/families are not split across folds during 
    cross validation, e.g. using sklearn's GroupKFold. Groups are defined as components 
    in the adjacency matrix.

    Parameter:
    ----------
    group_structure : array-like of shape (n_samples, n_samples)
        a matrix specifying the structure of the dataset, with positive
        values indicating relations between samples and zeros indicating no relations.
        Note: The diagonal will be set to 1
    
    Returns:
    --------
    cs : array-like of shape (n_samples,)
        1D array containing the group each sample belongs to

    """

    # make sure diagonal of family matrix is 1:
    cs2 = group_structure
    np.fill_diagonal(cs2, 1)
    # create adjacency graph
    csG = ig.Graph.Adjacency(cs2)
    # get groups (connected components in adjacency matrix)
    groups = csG.connected_components()
    cs_tmp = groups.membership
    cs = np.asarray(cs_tmp)

    return cs

def deconfound(Y, confX, betaY=None, my=None):
    """Deconfound
    """
    if betaY is None:
        betaY = np.zeros(shape=Y.shape)
        my = np.mean(Y)
        Y = Y - my
        if confX.ndim==1:
            confX = np.reshape(confX, (-1,1))
        confXT = confX.T
        betaY_tmp = np.linalg.lstsq((np.matmul(confXT, confX)) + 0.00001 * np.identity(confX.shape[1]), np.matmul(confXT, Y))
        betaY = betaY_tmp[0]
    
    res = Y - np.matmul(confX,betaY)
    Y = res
    
    return betaY, my, Y

def reconfound(Y, conf, betaY, my):
    """Reconfound
    """
    if conf.ndim==1:
        conf = np.reshape(conf, (-1,1))
    Y = Y + np.matmul(conf, betaY) + my
    return Y
        
def predict_phenotype(hmm, Y, behav, indices, predictor='Fisherkernel', estimator='KernelRidge', options=None):
    """Predict phenotype from HMM
    This uses either the Fisher kernel (default) or a set of HMM summary metrics
    to predict a phenotype, in a nested cross-validated way
    Estimators so far include: Kernel Ridge Regression and Ridge Regression
    Cross-validation strategies so far include: KFold and GroupKFold
    Hyperparameter optimization strategies so far include: only grid search
    
    Parameters:
    -----------
    hmm : HMM object
        An instance of the HMM class, estimated on the group-level
    Y : array-like of shape (n_samples, n_variables_2)
        (group-level) timeseries data
    behav : array-like of shape (n_samples,)
        phenotype to be predicted
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data. 
        Note that this function does not work if indices=None  
    predictor : char (optional, default to 'Fisherkernel')
        What to predict from, either 'Fisherkernel' or 'summary_metrics' (default='Fisherkernel')
    estimator : char (optional, default to 'KernelRidge')
        Model to be used for prediction (default='KernelRidge')
        This should be the name of a sklearn base estimator
        (for now either 'KernelRidge' or 'Ridge')
    options : dict (optional, default to None)
        general relevant options are:
            'CVscheme': char, which CVscheme to use (default: 'GroupKFold' if group structure is specified, otherwise: KFold)
            'nfolds': int, number of folds k for (outer and inner) k-fold CV loops
            'group_structure': ndarray of (n_samples, n_samples), matrix specifying group structure: positive values if samples are related, zeros otherwise
            'confounds': array-like of shape (n_sessions,) or (n_sessions, n_confounds) containing confounding variables
            'return_scores': bool, whether to return also the model scores of each fold
            'return_models': bool, whether to return also the trained models of each fold
            'return_hyperparams': bool, whether to return also the optimised hyperparameters of each fold
            possible hyperparameters for model, e.g. 'alpha' for (kernel) ridge regression
        for Fisher kernel, relevant options are: 
            'shape': char, either 'linear' or 'Gaussian' (TO DO)
            'incl_Pi': bool, whether to include the gradient w.r.t. the initial state probabilities when computing the Fisher kernel
            'incl_P': bool, whether to include the gradient w.r.t. the transition probabilities
            'incl_Mu': bool, whether to include the gradient w.r.t. the state means (note that this only works if means were not set to 0 when training HMM)
            'incl_Sigma': bool, whether to include the gradient w.r.t. the state covariances
        for summary metrics, relevant options are:
            'metrics': list of char, containing metrics to be included as features

    Returns:
    --------
    results : dict
        containing
        'behav_pred': predicted phenotype on test sets
        'corr': correlation coefficient between predicted and actual values
        (if requested):
        'scores': the model scores of each fold
        'models': the trained models from each fold
        'hyperparams': the optimised hyperparameters of each fold

    Raises:
    -------
    Exception
        If the model has not been trained or if necessary input is missing
    
    """

    # check conditions
    if not hmm.trained:
        raise Exception("The model has not yet been trained")

    if behav is None:
        raise Exception("Phenotype to be predicted needs to be provided")
    
    if indices is None: 
        raise Exception("To predict phenotype from HMM, indices need to be provided")
    
    # check options or set default:
    if options is None: 
        options = {}   
    # necessary options for Fisher kernel:
    if predictor=='Fisherkernel':
        if not options['shape']:
            shape='linear' 
        else:
            shape=options['shape']
        if 'incl_Pi' in options:
            incl_Pi = options['incl_Pi']
        else:
            incl_Pi = True
        if 'incl_P' in options:
            incl_P = options['incl_P']
        else:
            incl_P = True
        if 'incl_Mu' in options:
            incl_Mu = options['incl_Mu']
        else:
            incl_Mu = False
        if 'incl_Sigma' in options:
            incl_Sigma = options['incl_Sigma']
        else:
            incl_Sigma = True   
        estimator='KernelRidge'
    # necessary options for summary metrics
    if predictor=='summary_metrics':
        if not 'metrics' in options:
            metrics = ['FO', 'switching_rate', 'lifetimes']
        else:
            metrics = options['metrics']
        estimator='Ridge'
    # other necessary options
    if not 'CVscheme' in options:
        CVscheme = 'KFold'
    else:
        CVscheme = options['CVscheme']

    if not 'nfolds' in options:
        nfolds = 5
    else:
        nfolds = options['nfolds']

    if not 'group_structure' in options:
        do_groupKFold = False
        allcs = None
    else:
        do_groupKFold = True
        allcs = options['group_structure']
        CVscheme = 'GroupKFold'

    if not 'confounds' in options:
        confounds = None
        deconfounding = False
    else:
        confounds = options['confounds']
        if confounds.ndim==1:
            confounds = confounds.reshape((-1,1))
        deconfounding = True

    N = indices.shape[0] # number of samples

    # get features/kernel from HMM to be predicted from (default: Fisher kernel):
    if predictor=='Fisherkernel':
        if shape=='linear':
            tau=None
        elif shape=='Gaussian':
            tau=options['tau']
        Xin = hmm_kernel(hmm, Y, indices, type='Fisher', shape=shape, incl_Pi=incl_Pi, incl_P=incl_P, incl_Mu=incl_Mu, incl_Sigma=incl_Sigma, tau=tau, return_feat=False, return_dist=False)
    # alternative: predict from HMM summary metrics
    elif predictor=='summary_metrics':
        Xin = get_summ_features(hmm, Y, indices, metrics)  
            
    # create CV folds
    if do_groupKFold: # when using family/group structure - use GroupKFold
        cs = get_groups(allcs)
        cvfolds = model_selection.GroupKFold(n_splits=nfolds)
        cvfolds.get_n_splits(Y, behav, cs)
    elif CVscheme=='KFold': # when not using family/group structure
        cvfolds = model_selection.KFold(n_splits=nfolds)
        cvfolds.get_n_splits(Y, behav)

    # create empty return structures
    behav_pred = np.zeros(shape=N)  
    behav_mean = np.zeros(shape=N)
    if deconfounding:
        behavD = np.zeros(shape=N)
        behav_predD = np.zeros(shape=N)
        behav_meanD = np.zeros(shape=N)
    # optional return: 
    if 'return_scores' in options and options['return_scores']==True:
        scores = list()
        return_scores = True
        if deconfounding:
            scores_deconf = list()
    else:
        return_scores = False
    if 'return_models'in options and options['return_models']==True:
        models = list()
        return_models = True
    else:
        return_models = False
    if 'return_hyperparams' in options and options['return_hyperparams']==True:
        hyperparams = list()
        return_hyperparams = True
    else:
        return_hyperparams = False

    # main prediction:
    # KRR (default for Fisher kernel):
    if estimator=='KernelRidge':
        if not 'alpha' in options:
            alphas = np.logspace(-4, -1, 6)
        else:
            alphas = options['alpha']

        model = kernel_ridge.KernelRidge(kernel="precomputed")

        if do_groupKFold:
            for train, test in cvfolds.split(Xin, behav, groups=cs):
                behav_train = behav[train]
                behav_mean[test] = np.mean(behav_train)
                # deconfounding:
                if deconfounding:
                    confounds_train = confounds[train,:]
                    CbetaY, CinterceptY, behav_train = deconfound(behav_train, confounds_train)
                # train model and make predictions:
                model_tuned = model_selection.GridSearchCV(estimator=model, param_grid=dict(alpha=alphas), cv=cvfolds)
                model_tuned.fit(Xin[train, train.reshape(-1,1)], behav_train, groups=cs[train])
                behav_pred[test] = model_tuned.predict(Xin[train, test.reshape(-1,1)])
                # in deconfounded space
                behav_predD[test] = behav_pred[test]
                behavD[test] = behav[test]
                behav_meanD[test] = behav_mean[test]
                if deconfounding:
                    _,_,behavD[test] = deconfound(behavD[test], confounds[test,:], CbetaY, CinterceptY)
                    behav_predD[test] = reconfound(behav_predD[test], confounds[test,:], CbetaY, CinterceptY)
                    behav_meanD[test] = reconfound(behav_meanD[test], confounds[test,:], CbetaY, CinterceptY)
                # get additional output
                if return_scores:
                    scores.append(model_tuned.score(Xin[train, test.reshape(-1,1)], behav[test]))
                    if deconfounding:
                        scores_deconf.append(model_tuned.score(Xin[train, test.reshape(-1,1)], behavD[test]))
                if return_models:
                    models.append(model_tuned)
                if return_hyperparams:
                    hyperparams.append(model_tuned.best_estimator_.alpha)
        else: # KFold CV not accounting for family structure
            for train, test in cvfolds.split(Xin, behav):
                behav_train = behav[train]
                behav_mean[test] = np.mean(behav_train)
                # deconfounding:
                if deconfounding:
                    confounds_train = confounds[train,:]
                    CbetaY, CinterceptY, behav_train = deconfound(behav_train, confounds_train)
                # train model and make predictions:
                model_tuned = model_selection.GridSearchCV(estimator=model, param_grid=dict(alpha=alphas), cv=cvfolds)
                model_tuned.fit(Xin[train, train.reshape(-1,1)], behav_train)
                behav_pred[test] = model_tuned.predict(Xin[train, test.reshape(-1,1)])
                # in deconfounded space
                behav_predD[test] = behav_pred[test]
                behavD[test] = behav[test]
                behav_meanD[test] = behav_mean[test]
                if deconfounding:
                    _,_,behavD[test] = deconfound(behavD[test], confounds[test,:], CbetaY, CinterceptY)
                    behav_predD[test] = reconfound(behav_predD[test], confounds[test,:], CbetaY, CinterceptY)
                    behav_meanD[test] = reconfound(behav_meanD[test], confounds[test,:], CbetaY, CinterceptY)
                # get additional output
                if return_scores:
                    scores.append(model_tuned.score(Xin[train, test.reshape(-1,1)], behav[test]))
                    if deconfounding:
                        scores_deconf.append(model_tuned.score(Xin[train, test.reshape(-1,1)], behavD[test]))
                if return_models:
                    models.append(model_tuned)
                if return_hyperparams:
                    hyperparams.append(model_tuned.best_estimator_.alpha)
    
    # RR (default for summary metrics):
    elif estimator=='Ridge':
        if not 'alpha' in options:
            alphas = np.logspace(-4, -1, 6)
        else:
            alphas = options['alpha']

        model = linear_model.Ridge()

        if do_groupKFold:
            for train, test in cvfolds.split(Xin, behav, groups=cs):
                behav_train = behav[train]
                behav_mean[test] = np.mean(behav_train)
                # deconfounding:
                if deconfounding:
                    confounds_train = confounds[train,:]
                    CbetaY, CinterceptY, behav_train = deconfound(behav_train, confounds_train)
                # train model and make predictions:
                model_tuned = model_selection.GridSearchCV(estimator=model, param_grid=dict(alpha=alphas), cv=cvfolds)
                model_tuned.fit(Xin[train, :], behav_train, groups=cs[train])
                behav_pred[test] = model_tuned.predict(Xin[test,:])
                # in deconfounded space
                behav_predD[test] = behav_pred[test]
                behavD[test] = behav[test]
                behav_meanD[test] = behav_mean[test]
                if deconfounding:
                    _,_,behavD[test] = deconfound(behavD[test], confounds[test,:], CbetaY, CinterceptY)
                    behav_predD[test] = reconfound(behav_predD[test], confounds[test,:], CbetaY, CinterceptY)
                    behav_meanD[test] = reconfound(behav_meanD[test], confounds[test,:], CbetaY, CinterceptY)
                # get additional output
                if return_scores:
                    scores.append(model_tuned.score(Xin[test,:], behav[test]))
                    if deconfounding:
                        scores_deconf.append(model_tuned.score(Xin[test,:], behavD[test]))
                if return_models:
                    models.append(model_tuned)
                if return_hyperparams:
                    hyperparams.append(model_tuned.best_estimator_.alpha)
        else: # KFold CV not using family structure
            for train, test in cvfolds.split(Xin, behav):
                behav_train = behav[train]
                behav_mean[test] = np.mean(behav_train)
                # deconfounding:
                if deconfounding:
                    confounds_train = confounds[train,:]
                    CbetaY, CinterceptY, behav_train = deconfound(behav_train, confounds_train)
                # train model and make predictions:
                model_tuned = model_selection.GridSearchCV(estimator=model, param_grid=dict(alpha=alphas), cv=cvfolds)
                model_tuned.fit(Xin[train, :], behav_train)
                behav_pred[test] = model_tuned.predict(Xin[test,:])
                # in deconfounded space
                behav_predD[test] = behav_pred[test]
                behavD[test] = behav[test]
                behav_meanD[test] = behav_mean[test]
                if deconfounding:
                    _,_,behavD[test] = deconfound(behavD[test], confounds[test,:], CbetaY, CinterceptY)
                    behav_predD[test] = reconfound(behav_predD[test], confounds[test,:], CbetaY, CinterceptY)
                    behav_meanD[test] = reconfound(behav_meanD[test], confounds[test,:], CbetaY, CinterceptY)
                # get additional output
                if return_scores:
                    scores.append(model_tuned.score(Xin[test,:], behav[test]))
                    if deconfounding:
                        scores_deconf.append(model_tuned.score(Xin[test,:], behavD[test]))
                if return_models:
                    models.append(model_tuned)
                if return_hyperparams:
                    hyperparams.append(model_tuned.best_estimator_.alpha)
    
    # get correlation coefficient between model-predicted and actual values
    corr = np.corrcoef(behav_pred, behav)[0,1]
    if deconfounding:
        corr_deconf = np.corrcoef(behav_predD, behavD)[0,1]

    # aggregate results and optional returns
    results = {}
    results['behav_pred'] = behav_pred
    results['corr'] = corr
    if deconfounding:
        results['behav_predD'] = behav_predD
        results['corr_deconf'] = corr_deconf
    if return_scores:
        results['scores'] = scores
        if deconfounding:
            results['scores_deconf'] = scores_deconf
    if return_models:
        results['models'] = models
    if return_hyperparams:
        results['hyperparams'] = hyperparams
    
    return results

def classify_phenotype(hmm, Y, behav, indices, predictor='FisherKernel', estimator='SVM', options=None):
    """Classify phenotype from HMM
    This uses either the Fisher kernel (default) or a set of HMM summary metrics
    to make a classification, in a nested cross-validated way
    Estimators so far include: SVM and Logistic Regression
    Cross-validation strategies so far include: KFold and GroupKFold
    Hyperparameter optimization strategies so far include: only grid search
    
    Parameters:
    -----------
    hmm : HMM object
        An instance of the HMM class, estimated on the group-level
    Y : array-like of shape (n_samples, n_variables_2)
        (group-level) timeseries data
    behav : array-like of shape (n_samples,)
        phenotype labels to be predicted
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data. 
        Note that this function does not work if indices=None  
    predictor : char (optional, default to 'Fisherkernel')
        What to predict from, either 'Fisherkernel' or 'summary_metrics' (default='Fisherkernel')
    estimator : char (optional, default to 'KernelRidge')
        Model to be used for prediction (default='KernelRidge')
        This should be the name of a sklearn base estimator
        (for now either 'KernelRidge' or 'Ridge')
    options : dict (optional, default to None)
        general relevant options are:
            'CVscheme': char, which CVscheme to use (default: 'GroupKFold' if group structure is specified, otherwise: KFold)
            'nfolds': int, number of folds k for (outer and inner) k-fold CV loops
            'group_structure': ndarray of (n_samples, n_samples), matrix specifying group structure: positive values if samples are related, zeros otherwise
            'return_scores': bool, whether to return also the model scores of each fold
            'return_models': bool, whether to return also the trained models of each fold
            'return_hyperparams': bool, whether to return also the optimised hyperparameters of each fold
            possible hyperparameters for model, e.g. 'alpha' for (kernel) ridge regression
            'return_prob': bool, whether to return also the estimated probabilities
        for Fisher kernel, relevant options are: 
            'shape': char, either 'linear' or 'Gaussian' (TO DO)
            'incl_Pi': bool, whether to include the gradient w.r.t. the initial state probabilities when computing the Fisher kernel
            'incl_P': bool, whether to include the gradient w.r.t. the transition probabilities
            'incl_Mu': bool, whether to include the gradient w.r.t. the state means (note that this only works if means were not set to 0 when training HMM)
            'incl_Sigma': bool, whether to include the gradient w.r.t. the state covariances
        for summary metrics, relevant options are:
            'metrics': list of char, containing metrics to be included as features

    Returns:
    --------
    results : dict
        containing
        'behav_pred': predicted phenotype on test sets
        'acc': overall accuracy
        (if requested):
        'behav_prob': predicted probabilities of each class on test set
        'scores': the model scores of each fold
        'models': the trained models from each fold
        'hyperparams': the optimised hyperparameters of each fold

    Raises:
    -------
    Exception
        If the model has not been trained or if necessary input is missing
    
    """

    # check conditions
    if not hmm.trained:
        raise Exception("The model has not yet been trained")

    if behav is None:
        raise Exception("Phenotype to be predicted needs to be provided")
    elif behav.ndim>1 or np.unique(behav).shape[0]>2:
        behav = preprocessing.LabelBinarizer().fit_transform(behav)
    
    if indices is None: 
        raise Exception("To predict phenotype from HMM, indices need to be provided")
    
    # check options or set default:
    if options is None: 
        options = {}   
    # necessary options for Fisher kernel:
    if predictor=='Fisherkernel':
        if not options['shape']:
            shape='linear' 
        else:
            shape=options['shape']
        if 'incl_Pi' in options:
            incl_Pi = options['incl_Pi']
        else:
            incl_Pi = True
        if 'incl_P' in options:
            incl_P = options['incl_P']
        else:
            incl_P = True
        if 'incl_Mu' in options:
            incl_Mu = options['incl_Mu']
        else:
            incl_Mu = False
        if 'incl_Sigma' in options:
            incl_Sigma = options['incl_Sigma']
        else:
            incl_Sigma = True   
        estimator='SVM'
    # necessary options for summary metrics
    if predictor=='summary_metrics':
        if not 'metrics' in options:
            metrics = ['FO', 'switching_rate', 'lifetimes']
        else:
            metrics = options['metrics']
        estimator='LogisticRegression'
    # other necessary options
    if not 'CVscheme' in options:
        CVscheme = 'KFold'
    else:
        CVscheme = options['CVscheme']

    if not 'nfolds' in options:
        nfolds = 5
    else:
        nfolds = options['nfolds']

    if not 'group_structure' in options:
        do_groupKFold = False
        allcs = None
    else:
        do_groupKFold = True
        allcs = options['group_structure']
        CVscheme = 'GroupKFold'

    if 'confounds' in options:
        raise Exception("Deconfounding is not implemented for classification, use prediction instead or remove confounds")

    N = indices.shape[0] # number of samples

    # get features/kernel from HMM to be predicted from (default: Fisher kernel):
    if predictor=='Fisherkernel':
        if shape=='linear':
            tau=None
        elif shape=='Gaussian':
            tau=options['tau']
        Xin = hmm_kernel(hmm, Y, indices, type='Fisher', shape=shape, incl_Pi=incl_Pi, incl_P=incl_P, incl_Mu=incl_Mu, incl_Sigma=incl_Sigma, tau=tau, return_feat=False, return_dist=False)
    # alternative: predict from HMM summary metrics
    elif predictor=='summary_metrics':
        Xin = get_summ_features(hmm, Y, indices, metrics)  
            
    # create CV folds
    if do_groupKFold: # when using family/group structure - use GroupKFold
        cs = get_groups(allcs)
        cvfolds = model_selection.GroupKFold(n_splits=nfolds)
        cvfolds.get_n_splits(Y, behav, cs)
    elif CVscheme=='KFold': # when not using family/group structure
        cvfolds = model_selection.KFold(n_splits=nfolds)
        cvfolds.get_n_splits(Y, behav)

    # create empty return structures
    behav_pred = np.zeros(shape=N)  

    # optional return: 
    if 'return_scores' in options and options['return_scores']==True:
        scores = list()
        return_scores = True
    else:
        return_scores = False
    if 'return_models'in options and options['return_models']==True:
        models = list()
        return_models = True
    else:
        return_models = False
    if 'return_hyperparams' in options and options['return_hyperparams']==True:
        hyperparams = list()
        return_hyperparams = True
    else:
        return_hyperparams = False

    if 'return_prob' in options and options['return_prob']==True:
        return_prob = True
        behav_prob = np.zeros(shape=(N,2))
    else:
        return_prob = False

    # main classification:
    # SVM (default for Fisher kernel):
    if estimator=='SVM':
        if not 'C' in options:
            Cs =  np.logspace(-10, 0, 10)
        else:
            Cs = options['C']

        model = svm.SVC(kernel="precomputed")
        if return_prob:
            model = svm.SVC(kernel="precomputed", probability=True)

        if do_groupKFold:
            for train, test in cvfolds.split(Xin, behav, groups=cs):
                behav_train = behav[train]
                # train model and make predictions:
                model_tuned = model_selection.GridSearchCV(estimator=model, param_grid=dict(C=Cs), cv=cvfolds)
                model_tuned.fit(Xin[train, train.reshape(-1,1)], behav_train, groups=cs[train])
                behav_pred[test] = model_tuned.predict(Xin[train, test.reshape(-1,1)])
                if return_prob:
                    behav_prob[test,:] = model_tuned.predict_proba(Xin[train, test.reshape(-1,1)])
                # get additional output
                if return_scores:
                    scores.append(model_tuned.score(Xin[train, test.reshape(-1,1)], behav[test]))
                if return_models:
                    models.append(model_tuned)
                if return_hyperparams:
                    hyperparams.append(model_tuned.best_estimator_.C)
        else: # KFold CV not accounting for family structure
            for train, test in cvfolds.split(Xin, behav):
                behav_train = behav[train]
                # train model and make predictions:
                model_tuned = model_selection.GridSearchCV(estimator=model, param_grid=dict(C=Cs), cv=cvfolds)
                model_tuned.fit(Xin[train, train.reshape(-1,1)], behav_train)
                behav_pred[test] = model_tuned.predict(Xin[train, test.reshape(-1,1)])
                if return_prob:
                    behav_prob[test,:] = model_tuned.predict_proba(Xin[train, test.reshape(-1,1)])
                # get additional output
                if return_scores:
                    scores.append(model_tuned.score(Xin[train, test.reshape(-1,1)], behav[test]))
                if return_models:
                    models.append(model_tuned)
                if return_hyperparams:
                    hyperparams.append(model_tuned.best_estimator_.C)
    
    # Logistic Regression (default for summary metrics):
    elif estimator=='LogisticRegression':
        if not 'C' in options:
            Cs = np.logspace(-10, 0, 10)
        else:
            Cs = options['C']

        model = linear_model.LogisticRegression()

        if do_groupKFold:
            for train, test in cvfolds.split(Xin, behav, groups=cs):
                behav_train = behav[train]
                # train model and make predictions:
                model_tuned = model_selection.GridSearchCV(estimator=model, param_grid=dict(C=Cs), cv=cvfolds)
                model_tuned.fit(Xin[train, :], behav_train, groups=cs[train])
                behav_pred[test] = model_tuned.predict(Xin[test,:])
                if return_prob:
                    behav_prob[test,:] = model_tuned.predict_proba(Xin[test,:])
                # get additional output
                if return_scores:
                    scores.append(model_tuned.score(Xin[test,:], behav[test]))
                if return_models:
                    models.append(model_tuned)
                if return_hyperparams:
                    hyperparams.append(model_tuned.best_estimator_.C)
        else: # KFold CV not using family structure
            for train, test in cvfolds.split(Xin, behav):
                behav_train = behav[train]
                # train model and make predictions:
                model_tuned = model_selection.GridSearchCV(estimator=model, param_grid=dict(C=Cs), cv=cvfolds)
                model_tuned.fit(Xin[train, :], behav_train)
                behav_pred[test] = model_tuned.predict(Xin[test,:])
                if return_prob:
                    behav_prob[test,:] = model_tuned.predict_proba(Xin[test,:])
                # get additional output
                if return_scores:
                    scores.append(model_tuned.score(Xin[test,:], behav[test]))
                if return_models:
                    models.append(model_tuned)
                if return_hyperparams:
                    hyperparams.append(model_tuned.best_estimator_.C)
    
    # get overall accuracy of model-predicted classes
    acc = ms.accuracy_score(behav_pred, behav)

    # aggregate results and optional returns
    results = {}
    results['behav_pred'] = behav_pred
    if return_prob:
        results['behav_prob'] = behav_prob
    results['acc'] = acc
    if return_scores:
        results['scores'] = scores
    if return_models:
        results['models'] = models
    if return_hyperparams:
        results['hyperparams'] = hyperparams
    
    return results

# TO DO: 
# add betas (gradient, prediction)
# option for deconfounding X
# add options for different hyperparameter optimisation
# fix Gaussian Fisher kernel to do proper optimisation for tau
