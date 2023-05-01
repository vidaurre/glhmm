#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input/output functions - Gaussian Linear Hidden Markov Model
@author: Diego Vidaurre 2023
"""

import numpy as np
import scipy.special
import scipy.io
import pickle

from . import glhmm
from . import auxiliary

def load_files(files,I=None,do_only_indices=False):
    """
    Loads data from files and returns the loaded data, indices, and individual indices for each file.
    """       

    X = []
    Y = []
    indices = []
    indices_individual = []
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
            ind = dat['indices']
        elif 'T' in dat:
            ind = auxiliary.make_indices_from_T(dat['T'])
        else:
            ind = np.zeros((1,2)).astype(int)
            ind[0,0] = 0
            ind[0,1] = Y[-1].shape[0]
        if len(ind.shape) == 1: ind = np.expand_dims(ind,axis=0)
        indices_individual.append(ind)
        indices.append(ind + sum_T)

        sum_T += dat["Y"].shape[0]

    if not do_only_indices:
        if len(X) > 0: X = np.concatenate(X)
        Y = np.concatenate(Y)
    indices = np.concatenate(indices)
    if len(indices.shape) == 1: indices = np.expand_dims(indices,axis=0)
    if len(X) == 0: X = None

    return X,Y,indices,indices_individual


def read_flattened_hmm_mat(file):
    """
    Reads a MATLAB file containing hidden Markov model (HMM) parameters, 
    and initializes a Gaussian linear hidden Markov model (GLHMM) using those parameters.
    """
    
    hmm_mat = scipy.io.loadmat(file)

    K = hmm_mat["K"][0][0]
    covtype = hmm_mat["train"]["covtype"][0][0][0]
    zeromean = hmm_mat["train"]["zeromean"][0][0][0][0]
    if not zeromean: model_mean = 'state'
    else: model_mean = 'no'
    if "state_0_Mu_W" in hmm_mat: 
        if (model_mean == 'state') and (hmm_mat["state_0_Mu_W"].shape[0] == 1):
            model_beta = 'no'
        elif hmm_mat["state_0_Mu_W"].shape[0] == 0:
            model_beta = 'no'
        else:
            model_beta = 'state'
    else: 
        model_beta = 'no'
    dirichlet_diag = hmm_mat["train"]["DirichletDiag"][0][0][0][0]
    connectivity = hmm_mat["train"]["S"][0][0]
    Pstructure = np.array(hmm_mat["train"]["Pstructure"][0][0], dtype=bool)
    Pistructure = np.squeeze(np.array(hmm_mat["train"]["Pistructure"][0][0], dtype=bool))

    shared_covmat = (covtype == 'shareddiag') or (covtype == 'sharedfull')
    diagonal_covmat = (covtype == 'shareddiag') or (covtype == 'diag') 

    if "prior_Omega_Gam_rate" in hmm_mat:
        prior_Omega_Gam_rate = hmm_mat["prior_Omega_Gam_rate"]
        prior_Omega_Gam_shape = hmm_mat["prior_Omega_Gam_shape"][0][0]
    else:
        prior_Omega_Gam_rate = hmm_mat["state_0_prior_Omega_Gam_rate"]
        prior_Omega_Gam_shape = hmm_mat["state_0_prior_Omega_Gam_shape"][0][0]    
    if diagonal_covmat: prior_Omega_Gam_rate = np.squeeze(prior_Omega_Gam_rate)
    q = prior_Omega_Gam_rate.shape[0]

    if "state_0_Mu_W" in hmm_mat:
        p = hmm_mat["state_0_Mu_W"].shape[0]
        if model_mean == 'state': p -= 1
    else: p = 0

    hmm = glhmm.glhmm(
        K=K,
        covtype=covtype,
        model_mean=model_mean,
        model_beta=model_beta,
        dirichlet_diag=dirichlet_diag,
        connectivity=connectivity,
        Pstructure=Pstructure,
        Pistructure=Pistructure
        )

    # mean 
    if model_mean == 'state':
        hmm.mean = []
        for k in range(K):
            hmm.mean.append({})
            Sigma_W = np.squeeze(hmm_mat["state_" + str(k) + "_S_W"])
            Mu_W = np.squeeze(hmm_mat["state_" + str(k) + "_Mu_W"])
            if model_beta == 'state':
                if q==1: hmm.mean[k]["Mu"] = np.array(Mu_W[0])
                else: hmm.mean[k]["Mu"] = Mu_W[0,:]
            else: 
                if q==1: hmm.mean[k]["Mu"] = np.array(Mu_W)
                else: hmm.mean[k]["Mu"] = Mu_W
            if diagonal_covmat:
                if model_beta == 'state':
                    if q==1: hmm.mean[k]["Sigma"] = np.array([[Sigma_W[0,0]]])
                    else: hmm.mean[k]["Sigma"] = np.diag(Sigma_W[:,0,0])
                else:
                    if q==1: hmm.mean[k]["Sigma"] = np.array([[Sigma_W]])
                    hmm.mean[k]["Sigma"] = np.diag(Sigma_W)
            else:
                if q==1: np.array([[Sigma_W[0,0]]])
                else: hmm.mean[k]["Sigma"] = Sigma_W[0:q,0:q]

    # beta
    if model_beta == 'state':
        if model_mean == 'state': j0 = 1
        else: j0 = 0
        hmm.beta = []
        for k in range(K):
            hmm.beta.append({})
            Sigma_W = hmm_mat["state_" + str(k) + "_S_W"]
            Mu_W = hmm_mat["state_" + str(k) + "_Mu_W"]
            hmm.beta[k]["Mu"] = np.zeros((p,q))
            hmm.beta[k]["Mu"][:,:] = Mu_W[j0:,:]
            if diagonal_covmat:
                hmm.beta[k]["Sigma"] = np.zeros((p,p,q))
                if q==1:
                    hmm.beta[k]["Sigma"][:,:,0] = Sigma_W[j0:,j0:]
                else:
                    for j in range(q):
                        hmm.beta[k]["Sigma"][:,:,j] = Sigma_W[j,j0:,j0:]
            else:
                hmm.beta[k]["Sigma"] = Sigma_W[(j0*q):,(j0*q):]

    hmm._glhmm__init_priors_sub(prior_Omega_Gam_rate,prior_Omega_Gam_shape,p,q)
    hmm._glhmm__update_priors()

    # covmatrix
    hmm.Sigma = []
    if diagonal_covmat and shared_covmat:
        hmm.Sigma.append({})
        hmm.Sigma[0]["rate"] = np.zeros(q)
        hmm.Sigma[0]["rate"][:] = hmm_mat["Omega_Gam_rate"]
        hmm.Sigma[0]["shape"] = hmm_mat["Omega_Gam_shape"][0][0]
    elif diagonal_covmat and not shared_covmat:
        for k in range(K):
            hmm.Sigma.append({})
            hmm.Sigma[k]["rate"] = np.zeros(q)
            hmm.Sigma[k]["rate"][:] = hmm_mat["state_" + str(k) + "_Omega_Gam_rate"]
            hmm.Sigma[k]["shape"] = hmm_mat["state_" + str(k) + "_Omega_Gam_shape"][0][0]
    elif not diagonal_covmat and shared_covmat:
        hmm.Sigma.append({})
        hmm.Sigma[0]["rate"] = hmm_mat["Omega_Gam_rate"]
        hmm.Sigma[0]["irate"] = hmm_mat["Omega_Gam_irate"]
        hmm.Sigma[0]["shape"] = hmm_mat["Omega_Gam_shape"][0][0]
    else: # not diagonal_covmat and not shared_covmat
        for k in range(K):
            hmm.Sigma.append({})
            hmm.Sigma[k]["rate"] = hmm_mat["state_" + str(k) + "_Omega_Gam_rate"]
            hmm.Sigma[k]["irate"] = hmm_mat["state_" + str(k) + "_Omega_Gam_irate"]
            hmm.Sigma[k]["shape"] = hmm_mat["state_" + str(k) + "_Omega_Gam_shape"][0][0]

    #hmm.init_dynamics()
    hmm.P = hmm_mat["P"]
    hmm.Pi = np.squeeze(hmm_mat["Pi"])
    hmm.Dir2d_alpha = hmm_mat["Dir2d_alpha"]
    hmm.Dir_alpha = np.squeeze(hmm_mat["Dir_alpha"])
    
    hmm.trained = True
    
    return hmm


def save_hmm(hmm, filename):
    """
    Saves a glhmm object on filename
    """
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(hmm, outp, pickle.HIGHEST_PROTOCOL)


def load_hmm(filename):
    """
    Loads a glhmm object from filename
    """
    with open(filename, 'rb') as inp:
        hmm = pickle.load(inp)
    return hmm



