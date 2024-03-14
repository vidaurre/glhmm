#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian Linear Hidden Markov Model
@author: Diego Vidaurre 2023
"""

import numpy as np
import math
import scipy
import scipy.special
import scipy.spatial
import sys
import warnings
import copy
import time

# import auxiliary
# import io
# import utils

from . import auxiliary
from . import io
from . import utils


class glhmm():
    """ Gaussian Linear Hidden Markov Model class to decode stimulus from data.
    
    Attributes:
    -----------
    K : int, default=10
        number of states in the model.
    covtype : str, {'shareddiag', 'diag','sharedfull','full'}, default 'shareddiag'
        Type of covariance matrix. Choose 'shareddiag' to have one diagonal covariance matrix for all states, 
        or 'diag' to have a diagonal full covariance matrix for each state, 
        or 'sharedfull' to have a shared full covariance matrix for all states,
        or 'full' to have a full covariance matrix for each state.
    model_mean : str, {'state', 'shared', 'no'}, default 'state'
        Model for the mean. If 'state', the mean will be modelled state-dependent.
        If 'shared', the mean will be modelled globally (shared between all states).
        If 'no' the mean of the timeseries will not be used to drive the states.
    model_beta : str, {'state', 'shared', 'no'}, default 'state'
        Model for the beta. If 'state', the regression coefficients will be modelled state-dependent.
        If 'shared', the regression coefficients will be modelled globally (shared between all states).
        If 'no' the regression coefficients will not be used to drive the states.
    dirichlet_diag : float, default=10
        The value of the diagonal of the Dirichlet distribution for the transition probabilities. 
        The higher the value, the more persistent the states will be. 
        Note that this value is relative; the prior competes with the data, so if the timeseries is very long, 
        the `dirichlet_diag` may have little effect unless it is set to a very large value.  
    connectivity : array_like of shape (n_states, n_states), optional
        Matrix of binary values defining the connectivity of the states. 
        This parameter can only be used with a diagonal covariance matrix (i.e., `covtype='diag'`).
    Pstructure : array_like, optional
        Binary matrix defining the allowed transitions between states.
        The default is a (n_states, n_states) matrix of all ones, allowing all possible transitions between states.
    Pistructure : array_like, optional
        Binary vector defining the allowed initial states.
        The default is a (n_states,) vector of all ones, allowing all states to be used as initial states.

    Notes:
    ------
    This class requires the following modules: numpy, math, scipy, sys, warnings, copy, and time.
    """

    ### Private methods

    def __init__(self,
        K=10, # model options
        covtype='shareddiag', 
        model_mean='state',
        model_beta='state',
        dirichlet_diag=10,
        connectivity=None,
        Pstructure=None,
        Pistructure=None
    ):

        if (connectivity is not None) and not ((covtype == 'shareddiag') or (covtype == 'diag')):
            warnings.warn('Parameter connectivity can only be used with a diagonal covariance matrix')
            connectivity = None

        self.hyperparameters = {}
        self.hyperparameters["K"] = K
        self.hyperparameters["covtype"] = covtype
        self.hyperparameters["model_mean"] = model_mean
        self.hyperparameters["model_beta"] = model_beta
        self.hyperparameters["dirichlet_diag"] = dirichlet_diag
        self.hyperparameters["connectivity"] = connectivity
        if Pstructure is None:
            self.hyperparameters["Pstructure"] = np.ones((K,K), dtype=bool)
        else:
            self.hyperparameters["Pstructure"] = Pstructure
        if Pistructure is None:
            self.hyperparameters["Pistructure"] = np.ones((K,), dtype=bool)       
        else:
            self.hyperparameters["Pistructure"] = Pistructure


        self.beta = None
        self.mean = None
        self.alpha_beta = None
        self.alpha_mean = None
        self.Sigma = None
        self.active_states = np.ones(K,dtype=bool)
        self.trained = False
        
    ## Private methods

    def __forward_backward(self,L,indices):
        """
        Calculate state time courses for a collection of segments
        """        

        ind = indices
        if len(ind.shape) == 1:
            ind = np.expand_dims(indices,axis=0)

        T,K = L.shape
        N = ind.shape[0]
        Gamma = np.zeros((T,K))
        Xi = np.zeros((T-N,K,K))
        scale = np.zeros(T)
        indices_Xi = auxiliary.Gamma_indices_to_Xi_indices(ind)

        for j in range(N):

            tt = range(ind[j,0],ind[j,1])
            tt_xi = range(indices_Xi[j,0],indices_Xi[j,1])

            a,b,sc = auxiliary.compute_alpha_beta(L[tt,:],self.Pi,self.P)

            scale[tt] = sc
            Gamma[tt,:] = b * a
            Xi[tt_xi,:,:] = np.matmul( np.expand_dims(a[0:-1,:],axis=2), \
                np.expand_dims((b[1:,:] * L[tt[1:],:]),axis=1)) * self.P

            # repeat if a Nan is produced, scaling the loglikelood
            if np.any(np.isinf(Gamma)) or np.any(np.isinf(Xi)):
                LL = np.log(L[tt,:])
                t = np.all(LL<0,axis=1)
                LL[t,:] = LL[t,:] -  np.expand_dims(np.max(LL[t,:],axis=1), axis=1)
                a,b,_ = auxiliary.compute_alpha_beta(np.exp(LL),self.Pi,self.P)
                Gamma[tt,:] = b * a
                Xi[tt_xi,:,:] = np.matmul( np.expand_dims(a[0:-1,:],axis=2), \
                    np.expand_dims((b[1:,:] * L[tt[1:],:]),axis=1)) * self.P

            Gamma[tt,:] = Gamma[tt,:] / np.expand_dims(np.sum(Gamma[tt,:],axis=1), axis=1)
            Xi[tt_xi,:,:] = Xi[tt_xi,:,:] / np.expand_dims(np.sum(Xi[tt_xi,:,:],axis=(1,2)),axis=(1,2))

        return Gamma,Xi,scale


    def __forward_backward_vp(self,L,indices):
        """
        Calculate viterbi path for a collection of segments
        """        

        ind = indices
        if len(ind.shape) == 1:
            ind = np.expand_dims(indices,axis=0)

        T,K = L.shape
        N = ind.shape[0]
        vpath = np.zeros((T,K))
        
        for j in range(N):
            tt = range(ind[j,0],ind[j,1])
            qstar = auxiliary.compute_qstar(L[tt,:],self.Pi,self.P)
            vpath[tt,:] = qstar

        return vpath   
    

    def __loglikelihood_k(self,X,Y,L,k,cache):

        T,q = Y.shape
        if self.hyperparameters["model_beta"] != 'no': p = X.shape[1]
        else: p = 0
        shared_covmat = (self.hyperparameters["covtype"] == 'shareddiag') or \
                        (self.hyperparameters["covtype"] == 'sharedfull')
        diagonal_covmat = (self.hyperparameters["covtype"] == 'shareddiag') or \
                        (self.hyperparameters["covtype"] == 'diag')
        k_mean,k_beta = k,k
        if self.hyperparameters["model_mean"] == 'shared': k_mean = 0
        if self.hyperparameters["model_beta"] == 'shared': k_beta = 0

        constant = - q / 2 * math.log(2*math.pi) #+ q / 2 

        if (k==0) and shared_covmat:

            if diagonal_covmat:
                PsiWish_alphasum = 0.5 * q * scipy.special.psi(self.Sigma[0]['shape']) 
                ldetWishB = 0
                for j in range(q):
                    ldetWishB += np.log(self.Sigma[0]['rate'][j])
                ldetWishB = - 0.5 * ldetWishB
                C = self.Sigma[0]['shape'] / self.Sigma[0]['rate']
                
            else:
                PsiWish_alphasum = 0
                for j in range(1,q+1):
                    PsiWish_alphasum += scipy.special.psi(0.5 * (self.Sigma[0]['shape'] + 1 - j))
                PsiWish_alphasum = 0.5 * PsiWish_alphasum
                (s, logdet) = np.linalg.slogdet(self.Sigma[0]['rate'])
                ldetWishB = - 0.5 * s * logdet
                C = self.Sigma[0]['shape'] * self.Sigma[0]['irate']

            cache["PsiWish_alphasum"] = PsiWish_alphasum
            cache["ldetWishB"] = ldetWishB
            cache["C"] = C

        elif shared_covmat:

            PsiWish_alphasum = cache["PsiWish_alphasum"]
            ldetWishB = cache["ldetWishB"]
            C = cache["C"]

        elif diagonal_covmat: # not shared_covmat

            PsiWish_alphasum = 0.5 * q * scipy.special.psi(self.Sigma[k]['shape']) 
            ldetWishB = 0
            for j in range(q):
                ldetWishB += np.log(self.Sigma[k]['rate'][j])
            ldetWishB = - 0.5 * ldetWishB
            C = self.Sigma[k]['shape'] / self.Sigma[k]['rate']
            diagonal_covmat = True

        else: # not shared_covmat, full matrix

            PsiWish_alphasum = 0
            for j in range(1,q+1): 
                PsiWish_alphasum += scipy.special.psi(0.5 * (self.Sigma[k]['shape'] + 1 - j))
            PsiWish_alphasum = 0.5 * PsiWish_alphasum
            (s, logdet) = np.linalg.slogdet(self.Sigma[k]['rate'])
            ldetWishB = - 0.5 * s * logdet
            C = self.Sigma[k]['shape'] * self.Sigma[k]['irate']

        # distance
        dist = np.zeros((T,))
        d = np.copy(Y)
        if self.mean is not None: d -= np.expand_dims(self.mean[k_mean]['Mu'],axis=0)
        if self.beta is not None: d -= (X @ self.beta[k_beta]['Mu'])
        if diagonal_covmat: Cd = d * C
        else: Cd = d @ C
        for j in range(q): dist -= 0.5 * d[:,j] * Cd[:,j]

        # cov trace for beta
        norm_wish_trace_W = np.zeros((T,))
        if self.beta is not None:
            if diagonal_covmat:
                jj = np.arange(p)
                for j in range(q):
                    if self.hyperparameters["connectivity"] is not None:
                        jj = np.where(self.hyperparameters["connectivity"][:,j]==1)[0]
                    Cb = self.beta[k_beta]['Sigma'][jj,jj[:,np.newaxis],j]
                    norm_wish_trace_W -= 0.5 * C[j] * np.sum(((X[:,jj] @ Cb)) * X[:,jj], axis=1)
            else:
                ind = np.arange(p) * q
                for j1 in range(q):
                    ind1 = ind + j1
                    tmp = X @ self.beta[k_beta]['Sigma'][ind1,:]
                    for j2 in range(q):
                        ind2 = ind + j2
                        norm_wish_trace_W -= 0.5 * C[j1,j2] * np.sum(tmp[:,ind2] * X, axis=1)

        # cov trace for mean
        norm_wish_trace_mean = np.zeros(T)
        if self.mean is not None:
            if diagonal_covmat:
                for j in range(q):
                    norm_wish_trace_mean -= 0.5 * C[j] * self.mean[k_mean]['Sigma'][j]
            else:
                norm_wish_trace_mean = - 0.5 * np.trace(self.mean[k_mean]['Sigma'] @ C)

        L[:,k] = constant + dist + norm_wish_trace_W + norm_wish_trace_mean + ldetWishB + PsiWish_alphasum


    @staticmethod
    def __check_options(options):

        if options is None: options = {}
        if not "cyc" in options: options["cyc"] = 100
        if not "cyc_to_go_under_th" in options: options["cyc_to_go_under_th"] = 10
        if not "initcyc" in options: options["initcyc"] = 10
        if not "initrep" in options: options["initrep"] = 5 
        if not "tol" in options: options["tol"] = 1e-4
        if not "threshold_active" in options: options["threshold_active"] = 20
        if not "deactivate_states" in options: options["deactivate_states"] = True
        if not "stochastic" in options: options["stochastic"] = False
        if not "updateGamma" in options: options["updateGamma"] = True
        if not "updateDyn" in options: options["updateDyn"] = True
        if not "updateObs" in options: options["updateObs"] = True
        if not "verbose" in options: options["verbose"] = True
        return options


    @staticmethod
    def __check_options_stochastic(options,files):

        if options is None: options = {}
        if not "Nbatch" in options: options["Nbatch"] = int(min(len(files)/2,10))
        if not "initNbatch" in options: options["initNbatch"] = options["Nbatch"]
        if not "cyc" in options: options["cyc"] = 100
        if not "initcyc" in options: options["initcyc"] = 25
        if not "forget_rate" in options: options["forget_rate"] = 0.75
        if not "base_weights" in options: options["base_weights"] = 0.25
        if not "min_cyc" in options: options["min_cyc"] = 10
        if ("updateGamma" in options) and (not options["updateGamma"]): 
            options["updateGamma"] = True
            warnings.warn('updateGamma has to be True for stochastic learning')
        if ("updateDyn" in options) and (not options["updateDyn"]): 
            options["updateDyn"] = True
            warnings.warn('updateDyn has to be True for stochastic learning')
        options = glhmm.__check_options(options)
        return options


    @staticmethod
    def __check_Gamma(Gamma):
        K = Gamma.shape[1]
        if np.any(np.isnan(Gamma)):
            raise Exception("NaN were generated in the state time courses, probably due to an artifacts") 
        status = np.all(np.std(Gamma,axis=0)<0.001)
        #status = (np.max(Gamma)<0.6) and (np.min(Gamma)>(1/K/2))
        return status

        
    def __init_Gamma(self,X,Y,indices,options):

        verbose = options["verbose"]
        options["verbose"] = False
        if options["initrep"] == 0:
            self.__init_prior_P_Pi() # init P,Pi priors
            self.__update_dynamics() # make P,Pi based on priors
            Gamma = self.sample_Gamma(indices)
            return Gamma

        fe = np.zeros(options["initrep"])
        for r in range(options["initrep"]):
            hmm_r = copy.deepcopy(self)
            options_r = copy.deepcopy(options)
            options_r["cyc"] = options_r["initcyc"]
            options_r["stochastic"] = False
            options_r["initrep"] = 0
            Gamma_r,_,fe_r = hmm_r.train(X,Y,indices,options=options_r)

            fe[r] = fe_r[-1]
            if (r == 0) or (fe[r] < np.min(fe[0:r])):
                Gamma = np.copy(Gamma_r)
                best = r
            if verbose: 
                print("Init repetition " + str(r+1) + " free energy = " + str(fe[r]))

        if verbose: 
            print("Best repetition: " + str(best+1))

        options["verbose"] = verbose
        return Gamma
                
                
    def __update_Pi(self):
        K = self.hyperparameters["K"]
        self.Pi = np.zeros((K,))
        PsiSum0 = scipy.special.psi(sum(self.Dir_alpha))
        for k in range(K):
            if self.Dir_alpha[k] == 0: continue
            self.Pi[k] = math.exp(scipy.special.psi(self.Dir_alpha[k])-PsiSum0)
        self.Pi = self.Pi / np.sum(self.Pi)


    def __update_P(self):
        K = self.hyperparameters["K"]
        self.P = np.zeros((K,K))
        for j in range(K):
            PsiSum = scipy.special.psi(sum(self.Dir2d_alpha[j,:]))
            for k in range(K):    
                if self.Dir2d_alpha[j,k] == 0: continue
                self.P[j,k] = math.exp(scipy.special.psi(self.Dir2d_alpha[j,k])-PsiSum)
            self.P[j,:] = self.P[j,:] / np.sum(self.P[j,:])


    def __Gamma_loglikelihood(self,Gamma,Xi,indices):
        K = self.hyperparameters["K"]
        minreal = sys.float_info.min
        Gamma_0 = Gamma[indices[:,0]]
        Gamma_0[Gamma_0 < minreal] = minreal
        PsiDir_alphasum = scipy.special.psi(sum(self.Dir_alpha))
        L = 0
        for k in range(K):
            L += np.sum(Gamma_0[:,k]) * (scipy.special.psi(self.Dir_alpha[k]) - PsiDir_alphasum)
        PsiDir2d_alphasum = np.zeros(K)
        for l in range(K): PsiDir2d_alphasum[l] = scipy.special.psi(sum(self.Dir2d_alpha[l,:]))
        for k in range(K):
            for l in range(K):
                L += np.sum(Xi[:,l,k]) * (scipy.special.psi(self.Dir2d_alpha[l,k]) - PsiDir2d_alphasum[l])
        return L


    def __update_priors(self):

        K = self.hyperparameters["K"]
        diagonal_covmat = (self.hyperparameters["covtype"] == 'shareddiag') or \
                        (self.hyperparameters["covtype"] == 'diag')      
        shared_beta = self.hyperparameters["model_beta"] == 'shared'
        shared_mean = self.hyperparameters["model_mean"] == 'shared'
        K_mean,K_beta = K,K
        if shared_mean: K_mean = 1
        if shared_beta: K_beta = 1

        if self.hyperparameters["model_mean"] != 'no':
            for k in range(K_mean):
                if diagonal_covmat:
                    self.alpha_mean[k]["rate"] = self.priors["alpha_mean"]["rate"] \
                        + 0.5 * self.mean[k]["Sigma"] + self.mean[k]["Mu"] ** 2
                else:
                    self.alpha_mean[k]["rate"] = self.priors["alpha_mean"]["rate"] \
                        + 0.5 * np.diag(self.mean[k]["Sigma"]) + self.mean[k]["Mu"] ** 2
                self.alpha_mean[k]["shape"] = self.priors["alpha_mean"]["shape"] + 0.5

        if self.hyperparameters["model_beta"] != 'no':
            p,q = self.beta[0]["Mu"].shape
            jj = np.arange(p)
            for k in range(K_beta):
                self.alpha_beta[k]["rate"] = self.priors["alpha_beta"]["rate"] + 0.5 * self.beta[k]["Mu"] ** 2
                if diagonal_covmat:
                    for j in range(q):
                        if self.hyperparameters["connectivity"] is not None:
                            jj = np.where(self.hyperparameters["connectivity"][:,j]==1)[0]
                        sjj = self.beta[k]["Sigma"][jj,jj[:,np.newaxis],j]
                        if (np.squeeze(sjj).shape != ()): sjj = np.squeeze(sjj)
                        self.alpha_beta[k]["rate"][jj,j] += 0.5 * np.diag(sjj)
                else:
                    self.alpha_beta[k]["rate"] += 0.5 * np.reshape(np.diag(self.beta[k]["Sigma"]),(p,q))
                self.alpha_beta[k]["shape"] = self.priors["alpha_beta"]["shape"] + 0.5

       
    def __init_priors(self,X=None,Y=None,files=None):

        if Y is None: X,Y,_,_ = io.load_files(files,0)
        p = X.shape[1] if X is not None else None
        q = Y.shape[1]

        if files is None:
            prior_shape,prior_rate = self.__compute_prior_covmat(X,Y)
        else:
            prior_shape,prior_rate = self.__compute_prior_covmat(files=files)       

        self.__init_priors_sub(prior_rate,prior_shape,p,q)
        

    def __init_priors_sub(self,prior_rate,prior_shape,p,q):

        K = self.hyperparameters["K"]
        shared_beta = self.hyperparameters["model_beta"] == 'shared'
        shared_mean = self.hyperparameters["model_mean"] == 'shared'
        diagonal_covmat = (self.hyperparameters["covtype"] == 'shareddiag') or \
                        (self.hyperparameters["covtype"] == 'diag')  
        K_mean,K_beta = K,K
        if shared_mean: K_mean = 1
        if shared_beta: K_beta = 1
        
        # priors for dynamics
        self.__init_prior_P_Pi()

        # Covariance matrix, use the range of the global error to set the prior
        self.priors["Sigma"] = {}
        self.priors["Sigma"]["shape"] = prior_shape
        self.priors["Sigma"]["rate"] = prior_rate
        if diagonal_covmat: 
            self.priors["Sigma"]["irate"] = 1 / self.priors["Sigma"]["rate"]
        else:
            self.priors["Sigma"]["irate"] = np.linalg.inv(self.priors["Sigma"]["rate"])

        # alpha (state betas and mean priors)
        if self.hyperparameters["model_mean"] != 'no':
            self.alpha_mean = []
            for k in range(K_mean):
                self.alpha_mean.append({})
                self.alpha_mean[k] = {}
        if self.hyperparameters["model_beta"] != 'no':
            self.alpha_beta = []
            for k in range(K_beta):
                self.alpha_beta.append({})
                self.alpha_beta[k] = {}

        if self.hyperparameters["model_mean"] != 'no':
                self.priors["alpha_mean"] = {}
                self.priors["alpha_mean"]["rate"] = 0.1 * np.ones(q)
                self.priors["alpha_mean"]["shape"] = 0.1
        if self.hyperparameters["model_beta"] != 'no':
                self.priors["alpha_beta"] = {}
                self.priors["alpha_beta"]["rate"] = 0.1 * np.ones((p,q))
                self.priors["alpha_beta"]["shape"] = 0.1        


    def __init_prior_P_Pi(self):

        K = self.hyperparameters["K"]
        # priors for dynamics
        self.priors = {}
        self.priors["Dir_alpha"] = np.ones(K)
        #self.priors["Dir_alpha"][self.Pistructure] = 1
        self.priors["Dir2d_alpha"] = np.ones((K,K))
        for k in range(K):
            #self.priors["Dir2d_alpha"][self.Pstructure[k,],:] = 1
            self.priors["Dir2d_alpha"][k,k] = self.hyperparameters["dirichlet_diag"]


    def __compute_prior_covmat(self,X=None,Y=None,files=None):

        diagonal_covmat = (self.hyperparameters["covtype"] == 'shareddiag') or \
                        (self.hyperparameters["covtype"] == 'diag')  

        if not files is None: # iterative calculation
            N = len(files)
            if self.hyperparameters["model_mean"] != 'no':
                for j in range(N):
                    _,Yj,_,_ = io.load_files(files,j)
                    if j == 0: 
                        m = np.sum(Yj,axis=0)
                        nt = Yj.shape[0]
                    else:
                        m += np.sum(Yj,axis=0)
                        nt += Yj.shape[0]
                m /= nt
            if self.hyperparameters["model_beta"] != 'no':
                for j in range(N):
                    Xj,Yj,_,_ = io.load_files(files,j)
                    if j == 0: 
                        XX = Xj.T @ Xj
                        XY = Xj.T @ Yj
                    else:
                        XX += Xj.T @ Xj
                        XY += Xj.T @ Yj    
                beta = np.linalg.inv(XX + 0.1 * Xj.shape[1]) @ XY   
            for j in range(N):
                Xj,Yj,_,_ = io.load_files(files,j)
                if j == 0: q = Yj.shape[1]
                if self.hyperparameters["model_mean"] != 'no':
                    Yj -= np.expand_dims(m,axis=0)
                if self.hyperparameters["model_beta"] != 'no':
                    Yj -= Xj @ beta
                rj = np.max(Yj,axis=0) - np.min(Yj,axis=0)                
                if j == 0: r = np.copy(rj)
                else: r = np.maximum(r,rj)

        else:
            T,q = Y.shape
            if self.hyperparameters["model_mean"] != 'no': 
                Yr = Y - np.expand_dims(np.mean(Y,axis=0),axis=0)
            else: 
                Yr = np.copy(Y)
            if self.hyperparameters["model_beta"] != 'no':
                p = X.shape[1]
                beta = np.linalg.inv(X.T @ X + 0.1 * np.eye(p)) @ (X.T @ Yr)
                Yr -= X @ beta
            r = np.max(Yr,axis=0) - np.min(Yr,axis=0)

        if diagonal_covmat:
            shape = 0.5 * (q+0.1-1)
            #shape = 0.5 * T
            rate = 0.5 * r
        else:
            shape = (q+0.1-1)
            #shape = T
            rate = np.diag(r)
        return shape,rate

  
    def __update_dynamics(self,Gamma=None,Xi=None,indices=None,
            Dir_alpha=None,Dir2d_alpha=None,rho=1,init=False):
        """
        Update transition prob matrix and initial probabilities
        """

        Pistructure = self.hyperparameters["Pistructure"]
        Pstructure = self.hyperparameters["Pstructure"]

        # Transition probability matrix
        if (Xi is None) and (Gamma is None) and (Dir2d_alpha is None):
            self.Dir2d_alpha = self.priors["Dir2d_alpha"]
        else:
            if Dir2d_alpha is None:
                if Xi is None:
                    Xi = auxiliary.approximate_Xi(Gamma,indices)
                Dir2d_alpha = np.sum(Xi,axis=0)
            if init:
                self.Dir2d_alpha = Dir2d_alpha + self.priors["Dir2d_alpha"]
            else:
                self.Dir2d_alpha = rho * (Dir2d_alpha + self.priors["Dir2d_alpha"]) \
                    + (1-rho) * np.copy(self.Dir2d_alpha)
        self.Dir2d_alpha[~Pstructure] = 0 
        self.__update_P()

        # Initial probabilities
        if (Gamma is None) and (Dir_alpha is None):
            self.Dir_alpha = self.priors["Dir_alpha"]
        else:
            if Dir_alpha is None:
                Dir_alpha = np.sum(Gamma[indices[:,0]],axis=0)
            if init:
                self.Dir_alpha = Dir_alpha + self.priors["Dir_alpha"]
            else:
                self.Dir_alpha = rho * (Dir_alpha + self.priors["Dir_alpha"]) \
                    + (1-rho) * np.copy(self.Dir_alpha)                
        self.Dir_alpha[~Pistructure] = 0
        self.__update_Pi()


    def __init_dynamics(self,Gamma=None,indices=None):
        """
        Initialise transition prob matrix and initial probabilities
        """

        self.__update_dynamics(Gamma,None,indices,init=True)


    def __update_obsdist(self,X,Y,Gamma,Nfactor=1,rho=1):
        """
        Update state distributions
        """        
        
        K = self.hyperparameters["K"]
        T,q = Y.shape
        if self.hyperparameters["model_beta"] != 'no': p = X.shape[1]
        shared_covmat = (self.hyperparameters["covtype"] == 'shareddiag') or \
                        (self.hyperparameters["covtype"] == 'sharedfull')
        diagonal_covmat = (self.hyperparameters["covtype"] == 'shareddiag') or \
                        (self.hyperparameters["covtype"] == 'diag')  
        shared_beta = self.hyperparameters["model_beta"] == 'shared'
        shared_mean = self.hyperparameters["model_mean"] == 'shared'
        K_mean,K_beta = K,K
        if shared_mean: K_mean = 1
        if shared_beta: K_beta = 1
        if self.hyperparameters["model_beta"] != 'no':
            XGX = np.zeros((p,p,K))
            for k in range(K): XGX[:,:,k] = (X * np.expand_dims(Gamma[:,k],axis=1)).T @ X
            XGXb = np.expand_dims(np.sum(XGX,axis=2),axis=2) if shared_beta else XGX
        Gb = np.ones((T,1)) if shared_beta else Gamma
        Gm = np.ones((T,1)) if shared_mean else Gamma

        # Mean
        if self.hyperparameters["model_mean"] != 'no':

            if self.hyperparameters["model_beta"] != 'no':
                Yr = np.copy(Y)
                for k in range(K_beta): 
                    Yr -= (X @ self.beta[k]["Mu"]) * np.expand_dims(Gb[:,k], axis=1)                    
            else: Yr = Y

            for k in range(K_mean):

                if (not shared_mean) and (not self.active_states[k]):
                    continue

                k_sigma = 0 if shared_covmat else k 
                GY = np.expand_dims(Gm[:,k],axis=1).T @ Yr 
                Nk = np.sum(Gm[:,k])

                if diagonal_covmat:
                    alpha = self.alpha_mean[k]["shape"] / self.alpha_mean[k]["rate"]
                    isigma = self.Sigma[k_sigma]["shape"] / self.Sigma[k_sigma]["rate"]
                    iS = Nfactor * isigma * Nk + alpha
                    S = 1 / iS
                    mu = np.squeeze(Nfactor * isigma * S * GY)
                    self.mean[k]["Sigma"] = rho * S + (1-rho) * np.copy(self.mean[k]["Sigma"])
                    self.mean[k]["Mu"] = rho * mu + (1-rho) * np.copy(self.mean[k]["Mu"])

                else:
                    alpha = np.diag(self.alpha_mean[k]["shape"] / self.alpha_mean[k]["rate"])
                    isigma = (self.Sigma[k_sigma]["shape"] * self.Sigma[k_sigma]["irate"]) 
                    gram = isigma * Nk
                    maxlik_mean = (GY / Nk).T
                    iS = Nfactor * gram + alpha
                    iS = (iS + iS.T) / 2
                    S = np.linalg.inv(iS)
                    mu = np.squeeze(Nfactor * S @ gram @ maxlik_mean)
                    self.mean[k]["Sigma"] = rho * S + (1-rho) * np.copy(self.mean[k]["Sigma"])
                    self.mean[k]["Mu"] = rho * mu + (1-rho) * np.copy(self.mean[k]["Mu"])

        # betas 
        if self.hyperparameters["model_beta"] != 'no':

            if self.hyperparameters["model_mean"] != 'no':
                Yr = np.copy(Y)
                for k in range(K_mean): 
                    Yr -= np.expand_dims(self.mean[k]["Mu"], axis=0) * np.expand_dims(Gm[:,k], axis=1)                  
            else: Yr = Y

            for k in range(K_beta):

                if (not shared_beta) and (not self.active_states[k]):
                    continue

                k_sigma = 0 if shared_covmat else k
                XGY = (X * np.expand_dims(Gb[:,k],axis=1)).T @ Yr

                if diagonal_covmat:
                    jj = np.arange(p)
                    for j in range(q):
                        if self.hyperparameters["connectivity"] is not None:
                            jj = np.where(self.hyperparameters["connectivity"][:,j]==1)[0]
                        alpha = np.diag(self.alpha_beta[k]["shape"] / self.alpha_beta[k]["rate"][jj,j])
                        isigma = self.Sigma[k_sigma]["shape"] / self.Sigma[k_sigma]["rate"][j]
                        iS = Nfactor * isigma * XGXb[jj,jj[:,np.newaxis],k] + alpha
                        iS = (iS + iS.T) / 2
                        S = np.linalg.inv(iS) 
                        mu = np.squeeze(S @ np.expand_dims(Nfactor * isigma * XGY[jj,j],axis=1))
                        S_old = np.copy(self.beta[k]["Sigma"][jj,jj[:,np.newaxis],j])
                        mu_old = np.copy(self.beta[k]["Mu"][jj,j])
                        self.beta[k]["Sigma"][jj,jj[:,np.newaxis],j] = rho * S + (1-rho) * S_old
                        self.beta[k]["Mu"][jj,j] = rho * mu + (1-rho) * mu_old

                else:
                    alpha = np.diag(self.alpha_beta[k]["shape"] \
                        / np.reshape(self.alpha_beta[k]["rate"],p*q))
                    isigma = self.Sigma[k_sigma]["shape"] * self.Sigma[k_sigma]["irate"]
                    gram = np.kron(XGXb[:,:,k],isigma)
                    maxlik_beta = np.reshape(np.linalg.lstsq(XGXb[:,:,k],XGY,rcond=None)[0],(p*q,1))
                    iS = Nfactor * gram + alpha
                    iS = (iS + iS.T) / 2
                    S = np.linalg.inv(iS)
                    mu = Nfactor * S @ gram @ maxlik_beta
                    mu = np.reshape(mu,(p,q)) 
                    self.beta[k]["Sigma"] = rho * S + (1-rho) * np.copy(self.beta[k]["Sigma"])
                    self.beta[k]["Mu"] = rho * mu + (1-rho) * np.copy(self.beta[k]["Mu"])
           
        # Sigma
        if shared_covmat:
            if diagonal_covmat:
                rate = np.copy(self.priors["Sigma"]["rate"])
                shape = self.priors["Sigma"]["shape"] + 0.5 * Nfactor * T
            else:
                rate = np.copy(self.priors["Sigma"]["rate"])
                shape = self.priors["Sigma"]["shape"] + Nfactor * T

        for k in range(K):

            d = np.copy(Y) 

            sm = np.zeros(q) if diagonal_covmat else np.zeros((q,q))
            if self.hyperparameters["model_mean"] != 'no': 
                kk = 0 if shared_mean else k
                d -= np.expand_dims(self.mean[kk]["Mu"], axis=0)
                sm = self.mean[kk]["Sigma"] * np.sum(Gamma[:,k])

            sb = np.zeros(q) if diagonal_covmat else np.zeros((q,q))
            if self.hyperparameters["model_beta"] != 'no': 
                kk = 0 if shared_beta else k
                d -= (X @ self.beta[kk]["Mu"])
                if diagonal_covmat:
                    sb = np.zeros((T,q))
                    jj = np.arange(p)
                    for j in range(q):
                        if self.hyperparameters["connectivity"] is not None:
                            jj = np.where(self.hyperparameters["connectivity"][:,j]==1)[0]
                        sb[:,j] += np.sum((X[:,jj] @ \
                            self.beta[kk]["Sigma"][jj,jj[:,np.newaxis],j]) *  X[:,jj],axis=1)
                    sb = np.sum(sb * np.expand_dims(Gamma[:,k], axis=1), axis=0)

                else:
                    for j1 in range(q):
                        ind1 = np.arange(p) * q + j1
                        for j2 in range(j1,q):
                            ind2 = np.arange(p) * q + j2
                            sb[j1,j2] = np.sum(self.beta[kk]["Sigma"][ind1,ind2[:,np.newaxis]] * XGX[:,:,k])
                            sb[j2,j1] = sb[j1,j2]

            if shared_covmat: # self.Sigma[0]["rate"] 
                if diagonal_covmat:
                    rate += 0.5 * Nfactor * \
                        ( (np.sum((d ** 2) * np.expand_dims(Gamma[:,k], axis=1),axis=0)) \
                        + sm + sb
                        )
                else:
                    rate += Nfactor * \
                        ( ((d * np.expand_dims(Gamma[:,k], axis=1)).T @ d) 
                        + sm + sb)
                    
            else:
                if diagonal_covmat:
                    rate = self.priors["Sigma"]["rate"] \
                            + 0.5 * Nfactor * \
                            ( np.sum((d ** 2) * np.expand_dims(Gamma[:,k], axis=1),axis=0) \
                            + sm + sb
                            ) 
                    shape = self.priors["Sigma"]["shape"] + \
                            0.5 * Nfactor * np.sum(Gamma[:,k])
                    self.Sigma[k]["rate"] = rho * rate + (1-rho) * np.copy(self.Sigma[k]["rate"])
                    self.Sigma[k]["shape"] = rho * shape + (1-rho) * np.copy(self.Sigma[k]["shape"]) 
                    self.Sigma[k]["irate"] = 1 / self.Sigma[k]["rate"]
                    
                else:
                    rate =  self.priors["Sigma"]["rate"] + Nfactor * \
                        ( ((d * np.expand_dims(Gamma[:,k], axis=1)).T @ d) \
                        + sm + sb
                        )
                    shape = self.priors["Sigma"]["shape"] + \
                        Nfactor * np.sum(Gamma[:,k])
                    self.Sigma[k]["rate"] = rho * rate + (1-rho) * np.copy(self.Sigma[k]["rate"])
                    self.Sigma[k]["shape"] = rho * shape + (1-rho) * np.copy(self.Sigma[k]["shape"]) 
                    self.Sigma[k]["irate"] = np.linalg.inv(self.Sigma[k]["rate"])   

        if shared_covmat:
            self.Sigma[0]["rate"] = rho * rate + (1-rho) * np.copy(self.Sigma[0]["rate"])
            self.Sigma[0]["shape"] = rho * shape + (1-rho) * np.copy(self.Sigma[0]["shape"]) 
            if diagonal_covmat:
                self.Sigma[0]["irate"] = 1 / self.Sigma[0]["irate"]
            else:
                self.Sigma[0]["irate"] = np.linalg.inv(self.Sigma[0]["rate"])    

        self.__update_priors()


    def __update_obsdist_stochastic(self,files,I,Gamma,rho):

        Nfactor = len(files) / len(I)
        X,Y,_,_ = io.load_files(files,I)
        self.__update_obsdist(X,Y,Gamma,Nfactor,rho)
        

    def __init_obsdist(self,X,Y,Gamma):
        
        K = self.hyperparameters["K"]
        q = Y.shape[1]
        if self.hyperparameters["model_beta"] != 'no': p = X.shape[1]
        shared_covmat = (self.hyperparameters["covtype"] == 'shareddiag') or \
                        (self.hyperparameters["covtype"] == 'sharedfull')
        diagonal_covmat = (self.hyperparameters["covtype"] == 'shareddiag') or \
                        (self.hyperparameters["covtype"] == 'diag') 
        shared_beta = self.hyperparameters["model_beta"] == 'shared'
        shared_mean = self.hyperparameters["model_mean"] == 'shared'
        K_mean,K_beta = K,K
        if shared_mean: K_mean = 1
        if shared_beta: K_beta = 1

        # alpha (keep it unregularised)
        if self.hyperparameters["model_mean"] != 'no':
            self.alpha_mean = []
            for k in range(K_mean):
                self.alpha_mean.append({})
                self.alpha_mean[k]["rate"] = 0.1 * np.ones(q)
                self.alpha_mean[k]["shape"] = 0.0001 # mild-regularised start

        if self.hyperparameters["model_beta"] != 'no':
            self.alpha_beta = []
            for k in range(K_beta):
                self.alpha_beta.append({})
                self.alpha_beta[k]["rate"] = 0.1 * np.ones((p,q))
                self.alpha_beta[k]["shape"] = 0.0001 # mild-regularised start  

        # Sigma (set to priors)
        self.Sigma = []
        if diagonal_covmat and shared_covmat:
            self.Sigma.append({})
            self.Sigma[0]["rate"] = np.copy(self.priors["Sigma"]["rate"])
            self.Sigma[0]["irate"] = 1 / self.Sigma[0]["rate"]
            self.Sigma[0]["shape"] = self.priors["Sigma"]["shape"]
        elif diagonal_covmat and not shared_covmat:
            for k in range(K):
                self.Sigma.append({})
                self.Sigma[k]["rate"] = np.copy(self.priors["Sigma"]["rate"])
                self.Sigma[k]["irate"] = 1 / self.Sigma[k]["rate"]
                self.Sigma[k]["shape"] = self.priors["Sigma"]["shape"]
        elif not diagonal_covmat and shared_covmat:
            self.Sigma.append({})
            self.Sigma[0]["rate"] = np.copy(self.priors["Sigma"]["rate"])
            self.Sigma[0]["irate"] = np.linalg.inv(self.Sigma[0]["rate"])
            self.Sigma[0]["shape"] = self.priors["Sigma"]["shape"] 
        else: # not diagonal_covmat and not shared_covmat
            for k in range(K):
                self.Sigma.append({})
                self.Sigma[k]["rate"] = np.copy(self.priors["Sigma"]["rate"])
                self.Sigma[k]["irate"] = np.linalg.inv(self.Sigma[k]["rate"])
                self.Sigma[k]["shape"] = self.priors["Sigma"]["shape"]

        # create initial values for mean and beta
        if self.hyperparameters["model_beta"] != 'no':
            self.beta = []
            for k in range(K_beta):
                self.beta.append({})
                self.beta[k]["Mu"] = np.zeros((p,q))
                if diagonal_covmat: 
                    self.beta[k]["Sigma"] = np.zeros((p,p,q))
                    for j in range(q): self.beta[k]["Sigma"][:,:,j] = 0.01 * np.eye(p)
                else: self.beta[k]["Sigma"] = 0.01 * np.eye(p*q)
        if self.hyperparameters["model_mean"] != 'no':
            self.mean = []
            for k in range(K_mean):
                self.mean.append({})  
                self.mean[k]["Mu"] = np.zeros(q)
                if diagonal_covmat: self.mean[k]["Sigma"] = 0.01 * np.ones(q)
                else: self.mean[k]["Sigma"] = 0.01 * np.eye(q)

        # do beta and mean conventionally, and redo alpha and Sigma
        self.__update_obsdist(X,Y,Gamma)


    def __init_obsdist_stochastic(self,files,I,Gamma):

        X,Y,_,_ = io.load_files(files,I)
        self.__init_obsdist(X,Y,Gamma)


    def __init_stochastic(self,files,options):

        N = len(files)
        I = np.random.choice(np.arange(N), size=options["initNbatch"], replace=False)
        X,Y,indices,_ = io.load_files(files,I)
        Gamma = self.__init_Gamma(X,Y,indices,options)
        self.__init_priors(files=files)
        self.__init_dynamics(Gamma,indices=indices)
        self.__init_obsdist_stochastic(files,I,Gamma)
        self.__update_priors()


    def __train_stochastic(self,files,Gamma,options):
        
        options = self.__check_options_stochastic(options,files)
        
        N = len(files)
        K = self.hyperparameters["K"]

        if options["verbose"]: start = time.time()

        # init model with a subset of subjects
        if not self.trained:
            if Gamma is None: 
                self.__init_stochastic(files,options)
            else:
                I = np.random.choice(np.arange(N), size=options["initNbatch"], replace=False)
                X,Y,indices,_ = io.load_files(files,I)
                _,_,indices_all,_ = io.load_files(files,do_only_indices=True)
                Gamma_subset = auxiliary.slice_matrix(Gamma,indices)
                self.__init_priors(files=files)
                self.__init_dynamics(Gamma,indices=indices_all)
                self.__update_obsdist_stochastic(files,I,Gamma_subset,1)
                self.__update_priors()
            self.trained = True        

        fe = np.empty(0)
        loglik_entropy = np.zeros((N,3)) # data likelihood and Gamma likelihood & entropy
        n_used = np.zeros(N)
        sampling_prob = np.ones(N) / N
        ever_used = np.zeros(N).astype(bool)

        sum_Gamma = np.zeros((K,N))
        Dir_alpha_each = np.zeros((K,N))
        Dir2d_alpha_each = np.zeros((K,K,N))
        cyc_to_go =  options["cyc_to_go_under_th"]

        # collect subject specific free energy terms
        for j in range(N):
            X,Y,indices,indices_individual = io.load_files(files,j)
            Gamma,Xi,_ = self.decode(X,Y,indices)
            # data likelihood
            todo = (False,True,False,False,False)
            if X is None:
                loglik_entropy[j,0] = np.sum(self.get_fe(None,Y,Gamma,Xi,None,indices_individual[0],todo))
            else:
                loglik_entropy[j,0] = np.sum(self.get_fe(X,Y,Gamma,Xi,None,indices_individual[0],todo))
            # Gamma likelihood and entropy
            todo = (True,False,False,False,False)
            loglik_entropy[j,1] = np.sum(self.get_fe(None,Y,Gamma,Xi,None,indices_individual[0],todo))
            todo = (False,False,True,False,False)
            loglik_entropy[j,2] = np.sum(self.get_fe(None,Y,Gamma,Xi,None,indices_individual[0],todo))

        # do the actual training
        for it in range(options["cyc"]):

            rho = (it + 2)**(-options["forget_rate"])
            I = np.random.choice(np.arange(N), size=options["Nbatch"], replace=False, p=sampling_prob)
            n_used[I] += 1
            n_used = n_used - np.min(n_used) + 1
            ever_used[I] = True

            sampling_prob = options["base_weights"] ** n_used
            sampling_prob = sampling_prob / np.sum(sampling_prob)
            Nfactor = N / np.sum(ever_used)

            X,Y,indices,indices_individual = io.load_files(files,I)
            indices_Xi = auxiliary.Gamma_indices_to_Xi_indices(indices)

            # E-step
            Gamma,Xi,_ = self.decode(X,Y,indices)
            sum_Gamma[:,I] = utils.get_FO(Gamma,indices,True).T
            
            # which states are active? 
            if options["deactivate_states"]:
                for k in range(K):
                    FO = np.sum(sum_Gamma[k,:])
                    active_state = self.active_states[k]
                    self.active_states[k] = FO > options["threshold_active"]
                    if options["verbose"]:
                        if (not active_state) and self.active_states[k]:
                            print("State " + str(k) + " is reactivated")
                        if active_state and (not self.active_states[k]):
                            print("State " + str(k) + " is deactivated")

            # M-step            
            if options["updateDyn"]:
                for j in range(options["Nbatch"]):
                    Dir_alpha_each[:,I[j]] = Gamma[indices[j,0]]
                    tt_j = range(indices_Xi[j,0],indices_Xi[j,1])
                    Dir2d_alpha_each[:,:,I[j]] = np.sum(Xi[tt_j,:,:],axis=0)
                Dir_alpha = Nfactor * np.sum(Dir_alpha_each[:,ever_used],axis=1)
                Dir2d_alpha = Nfactor * np.sum(Dir2d_alpha_each[:,:,ever_used],axis=2)
                self.__update_dynamics(Dir_alpha=Dir_alpha,Dir2d_alpha=Dir2d_alpha,rho=rho)

            if options["updateObs"]:
                self.__update_obsdist(X,Y,Gamma,Nfactor,rho)

            # collect subject specific free energy terms
            for j in range(options["Nbatch"]):
                tt_j = range(indices[j,0],indices[j,1])
                tt_j_xi = range(indices_Xi[j,0],indices_Xi[j,1])
                # data likelihood
                todo = (False,True,False,False,False)
                if X is None:
                    loglik_entropy[I[j],0] = np.sum(self.get_fe(None,Y[tt_j,:], \
                        Gamma[tt_j,:],Xi[tt_j_xi,:,:],None,indices_individual[j],todo))
                else:
                    loglik_entropy[I[j],0] = np.sum(self.get_fe(X[tt_j,:],Y[tt_j,:], \
                        Gamma[tt_j,:],Xi[tt_j_xi,:,:],None,indices_individual[j],todo))
                # Gamma likelihood and entropy
                todo = (True,False,False,False,False)
                loglik_entropy[I[j],1] = np.sum(self.get_fe(None,Y[tt_j,:], \
                        Gamma[tt_j,:],Xi[tt_j_xi,:,:],None,indices_individual[j],todo))
                todo = (False,False,True,False,False)
                loglik_entropy[I[j],2] = np.sum(self.get_fe(None,Y[tt_j,:], \
                        Gamma[tt_j,:],Xi[tt_j_xi,:,:],None,indices_individual[j],todo))
                                              
            # KL divergences
            todo = (False,False,False,True,True)
            kl = np.sum(self.get_fe(None,None,None,None,None,None,todo))
            fe_it = np.sum(kl) + np.sum(loglik_entropy)
            fe = np.append(fe, fe_it) 

            if len(fe) > 1:
                chgFrEn = abs((fe[-1]-fe[-2]) / (fe[-1]-fe[0]))
                if it > 10:
                    if np.abs(chgFrEn) < options["tol"]: cyc_to_go -= 1
                    else: cyc_to_go =  options["cyc_to_go_under_th"]
                if options["verbose"]: 
                    print("Cycle " + str(it+1) + ", free energy = " + str(fe_it) + \
                        ", relative change = " + str(chgFrEn) + ", rho = " + str(rho))
                if cyc_to_go == 0: 
                    if options["verbose"]: print("Reached early convergence")
                    break
            else:
                if options["verbose"]: print("Cycle " + str(it+1) + " free energy = " + str(fe_it))
            

        K_active = np.sum(self.active_states)
        if options["verbose"]:
            end = time.time()
            elapsed = end - start
            print("Finished stochastic training in " + str(round(elapsed,2)) +  \
                    "s : active states = " + str(K_active))

        return fe

    ### Public methods

    def loglikelihood(self,X,Y):
        """Computes the likelihood of the model per state and time point given the data X and Y.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_parcels)
            The timeseries of set of variables 1.
        Y : array-like of shape (n_samples, n_parcels)
            The timeseries of set of variables 2.

        Returns:
        --------
        L : array of shape (n_samples, n_states)
            The likelihood of the model per state and time point given the data X and Y.
            
        Raises:
        --------
        Exception
            If the model has not been trained.
        """

        if not self.trained: 
            raise Exception("The model has not yet been trained") 

        K = self.hyperparameters["K"]
        T = Y.shape[0]

        L = np.zeros((T,K))
        cache = {}
        for k in range(K):
            self.__loglikelihood_k(X,Y,L,k,cache)

        return L


    def decode(self,X,Y,indices=None,files=None,viterbi=False,set=None):
        """Calculates state time courses for all the data using either parallel or sequential processing.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_parcels)
            The timeseries of set of variables 1.
        Y : array-like of shape (n_samples, n_parcels)
            The timeseries of set of variables 2.
        indices : array-like of shape (n_sessions, 2), optional, default=None
            The start and end indices of each trial/session in the input data.
        files : list of str, optional, default=None
            List of filenames corresponding to the indices.
        viterbi : bool, optional, default=False
            Whether or not the Viterbi algorithm should be used.
        set : int, optional, default=None
            Index of the sessions set to decode.

        Returns:
        --------
        If viterbi=True:
            vpath : array of shape (n_samples,)
                The most likely state sequence.
        If viterbi=False:
            Gamma : array of shape (n_samples, n_states)
                The state probability timeseries.
            Xi : array of shape (n_samples - n_sessions, n_states, n_states)
                The joint probabilities of past and future states conditioned on data.
            scale : array-like of shape (n_samples,)
                The scaling factors from the inference, used to compute the free energy.
                In normal use, we would do
                    Gamma,Xi,_ = hmm.decode(X,Y,indices)
                
        Raises:
        -------
        Exception
            If the model has not been trained.
            If both 'files' and 'Y' arguments are provided.
        """


        if (files is not None) and (Y is not None):
            raise Exception("Argument 'files' cannot be used if the data (Y) is also provided")

        if not self.trained: 
            raise Exception("The model has not yet been trained") 

        if files is not None:
            X,Y,indices,_ = io.load_files(files)

        if indices is None: 
            indices = np.zeros((1,2)).astype(int)
            indices[0,0] = 0
            indices[0,1] = Y.shape[0]
        if len(indices.shape) == 1: 
            indices = np.expand_dims(indices,axis=0)

        X_sliced = X
        Y_sliced = Y
        if set is not None:
            if self.hyperparameters["model_beta"] != 'no':
                X_sliced = auxiliary.slice_matrix(X,indices[set,:])
            Y_sliced = auxiliary.slice_matrix(Y,indices[set,:])
            indices_sliced = indices[set,:]
        else:
            indices_sliced = indices

        L = np.exp(self.loglikelihood(X_sliced,Y_sliced))

        minreal = sys.float_info.min
        maxreal = sys.float_info.max
        L[L < minreal] = minreal
        L[L > maxreal] = maxreal

        N = indices.shape[0]

        if viterbi: 
            vpath = self.__forward_backward_vp(L,indices_sliced)
            return vpath
        else:
            Gamma,Xi,scale = self.__forward_backward(L,indices_sliced)
            return Gamma,Xi,scale


    def sample_Gamma(self,size):
        """Generates Gamma, for timeseries of lengths specified in variable size.

        Parameters:
        -----------
        size : array
            Array of shape (n_sessions,) or (n_sessions, 2). If `size` is 1-dimensional,
            each element represents the length of a session. If `size` is 2-dimensional,
            each row of `size` represents the start and end indices of a session in a timeseries.

        Returns:
        --------
        Gamma : array of shape (n_samples, n_states)
            The state probability timeseries.    

        """ 

        #if not self.trained: 
        #    raise Exception("The model has not yet been trained") 

        K = self.hyperparameters["K"]
        if len(size.shape)==1: # T
            T = size
            indices = auxiliary.make_indices_from_T(T)
        else: # indices
            indices = size
            if len(indices.shape) == 1: 
                indices = np.expand_dims(indices,axis=0)
            T = size[:,1] - size[:,0]

        Gamma = np.zeros((np.sum(T),K))
        N = indices.shape[0]
        rng = np.random.default_rng()

        for j in range(N):
            tt = np.arange(indices[j,0],indices[j,1])
            gamma = np.zeros((T[j],K))
            gamma[0,:] = rng.multinomial(1,self.Pi)
            for t in range(1,T[j]):
                k = np.where(gamma[t-1,:])[0][0]
                gamma[t,:] = rng.multinomial(1,self.P[k,:])
            Gamma[tt,:] = gamma

        return Gamma


    def sample(self,size,X=None,Gamma=None):
        """Generates Gamma and Y for timeseries of lengths specified in variable size.

        Parameters:
        -----------
        size : array of shape (n_sessions,) or (n_sessions, 2)
            If `size` is 1-dimensional, each element represents the length of a session. If `size` is 2-dimensional,
            each row of `size` represents the start and end indices of a session in a timeseries.
        X : array of shape (n_samples, n_parcels), default=None
            The timeseries of set of variables 1. 
        Gamma : array of shape (n_samples, n_states), default=None
            The state probability timeseries.

        Returns:
        --------
        Gamma : array of shape (n_samples, n_states)
            The state probability timeseries.
        Y: array of shape (n_samples,n_parcels)
            The timeseries of set of variables 2.
        If X=None:
            X : array of shape (n_samples, n_parcels)
               The timeseries of set of variables 1.

        """


        if not self.trained: 
            raise Exception("The model has not yet been trained") 

        K = self.hyperparameters["K"]
        shared_covmat = (self.hyperparameters["covtype"] == 'shareddiag') or \
                        (self.hyperparameters["covtype"] == 'sharedfull')
        diagonal_covmat = (self.hyperparameters["covtype"] == 'shareddiag') or \
                        (self.hyperparameters["covtype"] == 'diag')  

        if len(np.zeros(100).shape)==1: # T
            T = size
            indices = auxiliary.make_indices_from_T(T)
        else: # indices
            indices = size
            if len(indices.shape) == 1: 
                indices = np.expand_dims(indices,axis=0)
            T = size[:,1] - size[:,0]

        N = indices.shape[0]
        q = self.Sigma[0]["rate"].shape[0]
        
        if Gamma is None:
            Gamma = self.sample_Gamma(size)

        rng = np.random.default_rng()

        if (self.hyperparameters["model_beta"] != 'no') and (X is None):
            p = self.beta[0]["Mu"].shape[1]
            X = np.random.normal(size=(np.sum(T),p))

        # Y, mean
        Y = np.zeros((np.sum(T),q))
        if self.hyperparameters["model_mean"] == 'shared':
            Y += np.expand_dims(self.mean[0]['Mu'],axis=0)
        if self.hyperparameters["model_beta"] == 'shared':
            Y += X @ self.beta[0]["Mu"]
            
        for k in range(K):
            if self.hyperparameters["model_mean"] == 'state': 
                Y += np.expand_dims(self.mean[k]["Mu"],axis=0) * np.expand_dims(Gamma[:,k],axis=1)
            if self.hyperparameters["model_beta"] == 'state':
                Y += (X @ self.beta[k]["Mu"]) * np.expand_dims(Gamma[:,k],axis=1)

        # Y, covariance
        if shared_covmat:
            C = self.Sigma[0]["rate"] / self.Sigma[0]["shape"]
            if diagonal_covmat:
                Y += rng.normal(loc=np.zeros(q),scale=C,size=Y.shape)
            else:
                Y += rng.multivariate_normal(loc=np.zeros(q),cov=C,size=Y.shape)
        else:
            for k in range(K):
                if diagonal_covmat:
                    Y += rng.normal(loc=np.zeros(q),scale=C,size=Y.shape)  \
                        * np.expand_dims(Gamma[:,k],axis=1)
                else:
                    Y += rng.multivariate_normal(loc=np.zeros(q),cov=C,size=Y.shape) \
                        * np.expand_dims(Gamma[:,k],axis=1)

        return X,Y,Gamma


    def get_active_K(self):
        """Returns the number of active states

        Returns:
        --------
        K_active : int
            Number of active states.
        """

        K_active = np.sum(self.active_states)
        return K_active
    

    def get_r2(self,X,Y,Gamma,indices=None):
        """Computes the explained variance per session/trial and per column of Y

        Parameters:
        -----------
        X : array of shape (n_samples, n_variables_1)
            The timeseries of set of variables 1.
        Y : array of shape (n_samples, n_variables_2)
            The timeseries of set of variables 2.
        Gamma : array of shape (n_samples, n_states), default=None
            The state timeseries probabilities.
        indices : array-like of shape (n_sessions, 2), optional, default=None
            The start and end indices of each trial/session in the input data.
                
        Returns:
        --------
        r2 : array of shape (n_sessions, n_variables_2)
            The R-squared (proportion of the variance explained) for each session and each variable in Y.
            
        Raises:
        --------
        Exception
            If the model has not been trained, or if it does not have neither mean or beta

        Notes:
        -------
        This function does not take the covariance matrix into account
        
        """

        if not self.trained: 
            raise Exception("The model has not yet been trained") 

        K = self.hyperparameters["K"]
        q = Y.shape[1]
        N = indices.shape[0]

        r2 = np.zeros((N,q))
        m = np.mean(Y,axis=0)

        for j in range(N):

            tt_j = range(indices[j,0],indices[j,1])

            if X is not None:
                Xj = np.copy(X[tt_j,:])

            d = np.copy(Y[tt_j,:])
            if self.hyperparameters["model_mean"] == 'shared':
                d -= np.expand_dims(self.mean[0]['Mu'],axis=0)
            if self.hyperparameters["model_beta"] == 'shared':
                d -= (Xj @ self.beta[0]['Mu'])
            for k in range(K):
                if self.hyperparameters["model_mean"] == 'state': 
                    d -= np.expand_dims(self.mean[k]['Mu'],axis=0) * np.expand_dims(Gamma[:,k],axis=1)
                if self.hyperparameters["model_beta"] == 'state':
                    d -= (Xj @ self.beta[k]['Mu']) * np.expand_dims(Gamma[:,k],axis=1)
            d = np.sum(d**2,axis=0)

            d0 = np.copy(Y[tt_j,:])
            if self.hyperparameters["model_mean"] != 'no':
                d0 -= np.expand_dims(m,axis=0)
            d0 = np.sum(d0**2,axis=0)

            r2[j,:] = 1 - (d / d0)

        return r2

            
    def get_fe(self,X,Y,Gamma,Xi,scale=None,indices=None,todo=None,non_informative_prior_P=False):
        """Computes the Free Energy of an HMM depending on observation model.
        
        Parameters:
        -----------
        X : array of shape (n_samples, n_parcels)
            The timeseries of set of variables 1.
        Y : array of shape (n_samples, n_parcels)
            The timeseries of set of variables 2.
        Gamma : array of shape (n_samples, n_states), default=None
            The state timeseries probabilities.
        Xi : array-like of shape (n_samples - n_sessions, n_states, n_states)
            The joint probabilities of past and future states conditioned on data.
        scale : array-like of shape (n_samples,), default=None
            The scaling factors used to compute the free energy of the
            dataset. If None, scaling is automatically computed.
        indices : array-like of shape (n_sessions, 2), optional, default=None
            The start and end indices of each trial/session in the input data.
        todo:  bool of shape (n_terms,) or None, default=None
            Whether or not each of the 5 elements (see `fe_terms`) should be computed.
            Only for internal use.
        non_informative_prior_P: array-like of shape (n_states, n_states), optional, default=False
            Prior of transition probability matrix
            Only for internal use. 
                
        Returns:
        --------
        fe_terms : array of shape (n_terms,)
            The variational free energy, separated into different terms:
            - element 1: Gamma Entropy
            - element 2: Data negative log-likelihood
            - element 3: Gamma negative log-likelihood
            - element 4: KL divergence for initial and transition probabilities
            - element 5: KL divergence for the state parameters
            
        Raises:
        --------
        Exception
            If the model has not been trained.
            
        Notes:
        -------
        This function computes the variational free energy using a specific algorithm. For more information on the algorithm, see [^1].
        
        References:
        ------------
        [^1] Smith, J. et al. "A variational approach to Bayesian learning of switching dynamics in dynamical systems." Journal of Machine Learning Research, vol. 18, no. 4, 2017.
        """

        if not self.trained: 
            raise Exception("The model has not yet been trained") 

        if todo is None: # Gamma_entropy, data loglik, Gamma loglik, P/Pi KL, state KL 
            todo = (True,True,True,True,True)

        K = self.hyperparameters["K"]
        shared_covmat = (self.hyperparameters["covtype"] == 'shareddiag') or \
                (self.hyperparameters["covtype"] == 'sharedfull')
        diagonal_covmat = (self.hyperparameters["covtype"] == 'shareddiag') or \
                        (self.hyperparameters["covtype"] == 'diag')  
        shared_beta = self.hyperparameters["model_beta"] == 'shared'
        shared_mean = self.hyperparameters["model_mean"] == 'shared'
        K_mean,K_beta = K,K
        if shared_mean: K_mean = 1
        if shared_beta: K_beta = 1

        if todo[0] or todo[2]:
            if indices is None: 
                indices = np.zeros((1,2)).astype(int)
                indices[0,0] = 0
                indices[0,1] = Y.shape[0]
            elif len(indices.shape) == 1:
                indices = np.expand_dims(np.copy(indices),axis=0)

        if (scale is None) or (sum(todo)<5): # standard way
            use_scale = False
            fe_some_terms = np.zeros(3)
            if todo[0]:
                fe_some_terms[0] = -auxiliary.Gamma_entropy(Gamma,Xi,indices)
            if todo[1]:
                fe_some_terms[1] = -np.sum(self.loglikelihood(X,Y) * Gamma)     
            if todo[2]:
                fe_some_terms[2] = -self.__Gamma_loglikelihood(Gamma,Xi,indices)     
            
        else: # short way if we have the scale variables from the forward-backward algorithm 
            use_scale = True
            fe_some_terms = -np.log(scale) # (only valid just after)

        kldyn = []
        if todo[3]:
            if non_informative_prior_P: P_prior = np.ones((K,K))
            else: P_prior = self.priors["Dir2d_alpha"]
            kldyn.append(auxiliary.dirichlet_kl(self.Dir_alpha,self.priors["Dir_alpha"]))
            for k in range(K):
                kldyn.append(auxiliary.dirichlet_kl(self.Dir2d_alpha[k,:],P_prior[k,:]))

        klobs = []
        if todo[4]:
            q = self.Sigma[0]["rate"].shape[0]
            if self.hyperparameters["model_mean"] != 'no':
                for k in range(K_mean):
                    if diagonal_covmat:
                        for j in range(q):
                            klobs.append(auxiliary.gauss1d_kl( \
                                self.mean[k]["Mu"][j], self.mean[k]["Sigma"][j], \
                                0,self.alpha_mean[k]["rate"][j] / self.alpha_mean[k]["shape"] \
                            ))
                            klobs.append(auxiliary.gamma_kl(
                                self.alpha_mean[k]["shape"],self.alpha_mean[k]["rate"][j], \
                                self.priors["alpha_mean"]["shape"],self.priors["alpha_mean"]["rate"][j] \
                            ))
                    else:
                        klobs.append(auxiliary.gauss_kl( \
                            self.mean[k]["Mu"],self.mean[k]["Sigma"], \
                            np.zeros(q),np.diag(self.alpha_mean[k]["rate"] / self.alpha_mean[k]["shape"]) \
                        ))
                        klobs.append(np.sum(auxiliary.gamma_kl( \
                            self.alpha_mean[k]["shape"],self.alpha_mean[k]["rate"],\
                            self.priors["alpha_mean"]["shape"],self.priors["alpha_mean"]["rate"] \
                        )))   

            if self.hyperparameters["model_beta"] != 'no':
                p = self.beta[0]["Mu"].shape[0]
                jj = np.arange(p)
                for k in range(K_beta):
                    if diagonal_covmat:
                        for j in range(q):
                            if self.hyperparameters["connectivity"] is not None:
                                jj = np.where(self.hyperparameters["connectivity"][:,j]==1)[0]
                            pj = len(jj)
                            klobs.append(auxiliary.gauss_kl( \
                                self.beta[k]["Mu"][jj,j], self.beta[k]["Sigma"][jj,jj[:,np.newaxis],j], \
                                np.zeros((pj,)), np.diag(self.alpha_beta[k]["rate"][jj,j] / self.alpha_beta[k]["shape"]) \
                            ))
                            klobs.append(np.sum(auxiliary.gamma_kl( \
                                self.alpha_beta[k]["shape"],self.alpha_beta[k]["rate"][jj,j], \
                                self.priors["alpha_beta"]["shape"],self.priors["alpha_beta"]["rate"][jj,j] \
                            )))
                    else:
                        klobs.append(auxiliary.gauss_kl(
                            np.reshape(self.beta[k]["Mu"],(p*q,)),\
                            self.beta[k]["Sigma"],\
                            np.zeros(p*q),
                            np.diag(np.reshape(self.alpha_beta[k]["rate"],(p*q,)) / self.alpha_beta[k]["shape"]) \
                        ))
                        klobs.append(np.sum(auxiliary.gamma_kl( \
                                self.alpha_beta[k]["shape"],\
                                np.reshape(self.alpha_beta[k]["rate"],(p*q,)),\
                                self.priors["alpha_beta"]["shape"],\
                                np.reshape(self.priors["alpha_beta"]["rate"],(p*q,)) \
                        )))      

            if shared_covmat and (not diagonal_covmat):
                klobs.append(auxiliary.wishart_kl(self.Sigma[0]["shape"],self.Sigma[0]["rate"],\
                    self.priors["Sigma"]["shape"],self.priors["Sigma"]["rate"]))
            elif (not shared_covmat) and (not diagonal_covmat):
                for k in range(K):
                    klobs.append(auxiliary.wishart_kl(self.Sigma[k]["shape"],self.Sigma[k]["rate"],\
                        self.priors["Sigma"]["shape"],self.priors["Sigma"]["rate"]))
            elif shared_covmat and diagonal_covmat:
                klobs.append(np.sum(auxiliary.gamma_kl(self.Sigma[0]["shape"],self.Sigma[0]["rate"],\
                    self.priors["Sigma"]["shape"],self.priors["Sigma"]["rate"])))
            elif (not shared_covmat) and diagonal_covmat:
                for k in range(K):
                    klobs.append(np.sum(auxiliary.gamma_kl(self.Sigma[k]["shape"],self.Sigma[k]["rate"],\
                        self.priors["Sigma"]["shape"],self.priors["Sigma"]["rate"])))

        if use_scale:
            fe_terms = np.zeros(3)
            fe_terms[0] = np.sum(fe_some_terms)
            fe_terms[1] = sum(kldyn)
            fe_terms[2] = sum(klobs)
        else:
            fe_terms = np.zeros(5)
            for j in range(3): fe_terms[j] = fe_some_terms[j]
            fe_terms[3] = sum(kldyn)
            fe_terms[4] = sum(klobs)

        return fe_terms
        

    def get_covariance_matrix(self,k=0):
        """Returns the covariance matrix for the specified state.

        Parameters:
        -----------
        k : int, optional
            The index of the state. Default=0.

        Returns:
        --------
        array of shape (n_parcels, n_parcels)
            The covariance matrix for the specified state.

        Raises:
        -------
        Exception
            If the model has not been trained.

        """
        if not self.trained: 
            raise Exception("The model has not yet been trained") 

        return self.Sigma[k]["rate"] / self.Sigma[k]["shape"]


    def get_inverse_covariance_matrix(self,k=0):
        """Returns the inverse covariance matrix for the specified state.
        
        Parameters:
        -----------
        k : int, optional
            The index of the state. Default=0.
        
        Returns:
        --------
        array of shape (n_parcels, n_parcels)
            The inverse covariance matrix for the specified state.
        
        Raises:
        -------
        Exception
            If the model has not been trained.
        
        """
        if not self.trained: 
            raise Exception("The model has not yet been trained") 

        return self.Sigma[k]["irate"] * self.Sigma[k]["shape"]


    def get_beta(self,k=0):
        """Returns the regression coefficients (beta) for the specified state.

        Parameters:
        -----------
        k : int, optional, default=0
            The index of the state for which to retrieve the beta value.

        Returns:
        --------
        beta: ndarray of shape (n_variables_1 x n_variables_2)
            The regression coefficients of each variable in X on each variable in Y for the specified state.
            
        Raises:
        -------
        Exception
            If the model has not yet been trained.
            If the model has no beta.
        """

        if not self.trained: 
            raise Exception("The model has not yet been trained") 

        if self.hyperparameters["model_beta"] == 'no':
            raise Exception("The model has no beta")

        return self.beta[k]["Mu"]
    


    def get_betas(self):
        """Returns the regression coefficients (beta) for all states.

        Returns:
        --------
        betas: ndarray of shape (n_variables_1 x n_variables_2 x n_states)
            The regression coefficients of each variable in X on each variable in Y for all states.

        Raises:
        -------
        Exception
            If the model has not yet been trained.
            If the model has no beta.
        """

        if not self.trained: 
            raise Exception("The model has not yet been trained") 

        if self.hyperparameters["model_beta"] == 'no':
            raise Exception("The model has no beta")

        (p,q) = self.beta[0]["Mu"].shape
        K = self.hyperparameters["K"]
        betas = np.zeros((p,q,K))
        for k in range(K): betas[:,:,k] = self.beta[k]["Mu"]
        return betas


    def get_mean(self,k=0):

        """Returns the mean for the specified state.

        Parameters:
        -----------
        k : int, optional, default=0
            The index of the state for which to retrieve the mean.

        Returns:
        --------
        mean: ndarray of shape (n_variables_2,)
            The mean value of each variable in Y for the specified state.

        Raises:
        -------
        Exception
            If the model has not yet been trained.
            If the model has no mean.
        """

        if not self.trained: 
            raise Exception("The model has not yet been trained") 

        if self.hyperparameters["model_mean"] == 'no':
            raise Exception("The model has no mean")

        return self.mean[k]["Mu"]
    

    def get_means(self):

        """Returns the means for all states.

        Returns:
        --------
        means: ndarray of shape (n_variables_2, n_states)
            The mean value of each variable in Y for all states.

        Raises:
        -------
        Exception
            If the model has not yet been trained.
            If the model has no mean.
        """

        if not self.trained: 
            raise Exception("The model has not yet been trained") 

        if self.hyperparameters["model_mean"] == 'no':
            raise Exception("The model has no mean")

        if self.hyperparameters["model_beta"] != 'no':
            q = self.beta[0]["Mu"].shape[1]
        else:
            q = self.Sigma[0]["rate"].shape[0]

        K = self.hyperparameters["K"]
        means = np.zeros((q,K))
        for k in range(K): means[:,k] = self.mean[k]["Mu"]
        return means    


    def dual_estimate(self,X,Y,indices=None,Gamma=None,Xi=None,for_kernel=False):
        """Dual estimation of HMM parameters.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_variables_1)
            The timeseries of set of variables 1.
        Y : array-like of shape (n_samples, n_variables_2)
            The timeseries of set of variables 2.
        indices : array-like of shape (n_sessions, 2), optional
            The start and end indices of each trial/session in the input data. If None, a single segment spanning the entire sequence is used.
        Gamma : array-like of shape (n_samples, n_states), optional
            The state probabilities. If None, it is computed from the input observations.
        Xi : array-like of shape (n_samples - n_sessions, n_states, n_states), optional
            The joint probabilities of past and future states conditioned on data. If None, it is computed from the input observations.
        for_kernel : bool, optional 
            Whether purpose of dual estimation is kernel (gradient) computation, or not
            If True, function will also return Gamma and Xi (default False)

        Returns:
        ---------
        hmm_dual : object
            A copy of the HMM object with updated dynamics and observation distributions.
        """
        
        if not self.trained: 
            raise Exception("The model has not yet been trained") 

        if indices is None: # one big chunk with no cuts
            indices = np.zeros((1,2)).astype(int)
            indices[0,0] = 0
            indices[0,1] = Y.shape[0]
        if len(indices.shape) == 1: 
            indices = np.expand_dims(indices,axis=0)

        N = indices.shape[0]
        hmm_dual = []

        if Gamma is None:
            Gamma,Xi,_ = self.decode(X,Y,indices)

        hmm_dual = copy.deepcopy(self)
        hmm_dual.__update_dynamics(Gamma,Xi,indices)
        hmm_dual.__update_obsdist(X,Y,Gamma)

        # for j in range(N):
        #     tt = np.arange(indices[j,0],indices[j,1])
        #     tt_xi = tt[0:-1] - j
        #     indices_j = np.zeros((1,2)).astype(int)
        #     indices_j[0,1] = indices[j,1] - indices[j,0]
        #     hmm_dual.append = copy.deepcopy(self)
        #     hmm_dual[j].update_dynamics(Gamma[tt,:],Xi[tt_xi,:,:],indices_j)
        #     hmm_dual[j].update_obsdist(X[tt,:],Y[tt,:],Gamma[tt,:])
        if for_kernel:
            return hmm_dual,Gamma,Xi
        else:
            return hmm_dual


    def train(self,X=None,Y=None,indices=None,files=None,Gamma=None,Xi=None,scale=None,options=None):
        """
        Train the GLHMM on input data X and Y, which most general formulation is
        Y = mu_k + X beta_k + noise
        where noise is Gaussian with mean zero and standard deviation Sigma_k

        It supports both standard and stochastic variational learning; 
        for the latter, data must be supplied in files format

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_variables_1)
            The timeseries of set of variables 1.
        Y : array-like of shape (n_samples, n_variables_2)
            The timeseries of set of variables 2.
        indices : array-like of shape (n_sessions, 2), optional
            The start and end indices of each trial/session in the input data. If None, one big segment with no cuts is assumed.
        files : str or list of str, optional
            The filename(s) containing the data to load. If not None, X, Y, and indices are ignored.
        Gamma : array-like of shape (n_samples, n_states), optional
            The initial values of the state probabilities.
        Xi : array-like of shape (n_samples - n_sessions, n_states, n_states), optional
            The joint probabilities of past and future states conditioned on data.
        scale : array-like of shape (n_samples,), optional
            The scaling factors used to compute the free energy of the
            dataset. If None, scaling is automatically computed.
        options : dict, optional
            A dictionary with options to control the training process.

        Returns:
        --------
        Gamma : array-like of shape (n_samples, n_states)
            The state probabilities.
            To avoid unnecessary use of memory, Gamma is only returned if learning is non-stochastic;
            otherwise it is returned as an empty numpy array. 
            To get Gamma after stochastic learning, use the decode method. 
        Xi : array-like of shape (n_samples - n_sessions, n_states, n_states)
            The joint probabilities of past and future states conditioned on data.
            To avoid unnecessary use of memory, Xi is only returned if learning is non-stochastic;
            otherwise it is returned as an empty numpy array. 
            To get Xi after stochastic learning, use the decode method. 
        fe : array-like
            The free energy computed at each iteration of the training process.
            
        Raises:
        -------
        Exception
	        If `files` and `Y` are both provided or if neither are provided.
	        If `X` is not provided and the hyperparameter 'model_beta' is True.
            If 'files' is not provided and stochastic learning is called upon
        """

        stochastic = (options is not None) and ("stochastic" in options) and (options["stochastic"])
        
        if (files is not None) and (Y is not None) and (not stochastic):
            warnings.warn("Argument 'files' cannot be used if the data (Y) is also provided")

        if (files is None) and (Y is None):
            raise Exception("Training needs data")
        
        if (X is None) and (self.hyperparameters["model_beta"] != 'no'):
            raise Exception("If you want to model beta, X is needed as an argument")

        if stochastic:
            if files is None: 
                raise Exception("For stochastic learning, argument 'files' must be provided")
            if (X is not None) or (Y is not None):
                warnings.warn("X and Y are not used in stochastic learning")
            fe = self.__train_stochastic(files,Gamma,options)
            return np.empty(0),np.empty(0),fe

        options = self.__check_options(options)
        K = self.hyperparameters["K"]

        if files is not None:
            X,Y,indices,_ = io.load_files(files)

        if indices is None: # one big chunk with no cuts
            indices = np.zeros((1,2)).astype(int)
            indices[0,1] = Y.shape[0]
        if len(indices.shape) == 1: 
            indices = np.expand_dims(indices,axis=0)

        if options["verbose"]: start = time.time()

        if not self.trained:
            if Gamma is None: 
                Gamma = self.__init_Gamma(X,Y,indices,options)
            elif Gamma.shape != (Y.shape[0],K): 
                raise Exception('Supplied initial Gamma has not the correct dimensions')
            self.__init_priors(X,Y)
            self.__init_dynamics(Gamma,indices=indices)
            self.__init_obsdist(X,Y,Gamma)
            self.__update_priors()
            self.trained = True

        fe = np.empty(0)
        cyc_to_go = options["cyc_to_go_under_th"]

        for it in range(options["cyc"]):

            if options["updateGamma"]:

                # E-step
                Gamma,Xi,scale = self.decode(X,Y,indices)
                status = self.__check_Gamma(Gamma)
                if status:
                    warnings.warn('Gamma has almost zero variance: stuck in a weird solution')
                
                # which states are active? 
                if options["deactivate_states"]:
                    FO = np.sum(Gamma,axis=0)
                    for k in range(K):
                        active_state = self.active_states[k]
                        self.active_states[k] = FO[k] > options["threshold_active"]
                        if options["verbose"]:
                            if (not active_state) and self.active_states[k]:
                                print("State " + str(k) + " is reactivated")
                            if active_state and (not self.active_states[k]):
                                print("State " + str(k) + " is deactivated")

                # epsilon = 1
                # while status:
                #     self.perturb(epsilon)
                #     Gamma,Xi,scale = self.decode(X,Y,indices)
                #     status = self.__check_Gamma(Gamma)
                #     epsilon *= 2

            # if we use the scale to compute the FE, it's only valid after the E-step
            fe_it = np.sum(self.get_fe(X,Y,Gamma,Xi,scale,indices))
            fe = np.append(fe, fe_it) 

            if it > 1:
                chgFrEn = abs((fe[-1]-fe[-2]) / (fe[-1]-fe[0]))
                if np.abs(chgFrEn) < options["tol"]: cyc_to_go -= 1
                else: cyc_to_go = options["cyc_to_go_under_th"]
                if options["verbose"]: 
                    print("Cycle " + str(it+1) + ", free energy = " + str(fe_it) + \
                        ", relative change = " + str(chgFrEn))
                if cyc_to_go == 0: 
                    if options["verbose"]: print("Reached early convergence")
                    break
            else:
                if options["verbose"]: print("Cycle " + str(it+1) + " free energy = " + str(fe_it))

            # M-step
            if options["updateDyn"]:
                self.__update_dynamics(Gamma,Xi,indices)

            if options["updateObs"]:
                self.__update_obsdist(X,Y,Gamma)
                
        K_active = np.sum(self.active_states)
        if options["verbose"]:
            end = time.time()
            elapsed = end - start
            print("Finished training in " + str(round(elapsed,2)) +  \
                    "s : active states = " + str(K_active))

        return Gamma,Xi,fe

