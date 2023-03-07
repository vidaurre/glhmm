#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions - Gaussian Linear Hidden Markov Model
@author: Diego Vidaurre 2022
"""

import numpy as np
import sys
import scipy.special
import scipy.io
import math

from numba import njit


def slice_matrix(M,indices):
    """Slices rows of input matrix M based on indices array along axis 0.

    Parameters:
    -----------
    M : array-like of shape (n_samples, n_parcels)
        The input matrix.
    indices : array-like of shape (n_sessions, 2)
        The indices that define the sections (i.e., trials/sessions) of the data to be processed.

    Returns:
    --------
    M_sliced : array-like of shape (n_total_samples, n_parcels)
        The sliced matrix.

    """
    N = indices.shape[0]
    T = indices[:,1] - indices[:,0]
    M_sliced = np.empty((np.sum(T),M.shape[1]))
    acc = 0
    for j in range(N):
        ind_1 = np.arange(indices[j,0],indices[j,1])
        ind_2 = np.arange(0,T[j]) + acc
        acc += T[j]
        M_sliced[ind_2,:] = M[ind_1,:]
    return M_sliced


def make_indices_from_T(T):
    """Creates indices array from trials/sessions lengths.
    
    Parameters:
    -----------
    T : array-like of shape (n_sessions,)
        Contains the lengths of each trial/session.

    Returns:
    --------
    indices : array-like of shape (n_sessions, 2)
        The indices that define the sections of the data to be processed.

    """

    
    N = T.shape[0]
    indices = np.zeros((N,2),dtype=int)
    acc = 0
    for j in range(N):
        indices[j,0] = acc
        indices[j,1] = acc + T[j]
        acc += T[j]
    return indices


def Gamma_indices_to_Xi_indices(indices):
    """Converts indices from Gamma array to Xi array format.

    Note Xi has 1 sample less than Gamma per trial/session (i.e., n_samples - n_sessions).

    Parameters:
    -----------
    indices : array-like of shape (n_sessions, 2)
        The indices that define the segments of the data to be processed.

    Returns:
    --------
    indices_Xi : array-like of shape (n_sessions, 2)
        The converted indices in Xi array format.

    """

    indices_Xi = np.copy(indices)
    for j in range(indices.shape[0]):
        indices_Xi[j,0] -= j
        indices_Xi[j,1] -= (j+1)
    return indices_Xi


def jls_extract_def():
    return 


def approximate_Xi(Gamma,indices):
    """Approximates Xi array based on Gamma and indices.

    Parameters:
    -----------
    Gamma : array-like of shape (n_samples, n_states)
            The state probability time series.
    indices : array-like of shape (n_sessions, 2)
        The indices that define the segments of the data to be processed.

    Returns:
    --------
    Xi : array-like of shape (n_samples - n_sessions, n_states, n_states)
        The joint probabilities of past and future states conditioned on data.

    """
    jls_extract_def()

    K = Gamma.shape[1]
    N = indices.shape[0]
    T = indices[:,1] - indices[:,0]
    Xi = np.empty((np.sum(T)-N,K,K))
    acc = 0
    for j in range(N):
        ind_1_1 = np.arange(indices[j,0],indices[j,1]-1)
        ind_1_2 = np.arange(indices[j,0]+1,indices[j,1])
        ind_2 = np.arange(0,T[j]-1) + acc
        acc += (T[j]-1)
        xi = Gamma[ind_1_1,:].T @ Gamma[ind_1_2,:]
        xi = xi / np.sum(xi)
        Xi[ind_2,:,:] = xi
    return Xi    


@njit
def compute_alpha_beta(L,Pi,P):
    """Computes alpha and beta values and scaling factors.

    Parameters:
    -----------
    L : array-like of shape (n_samples, n_states)
        The L matrix.
    Pi : array-like with shape (n_states,)
        The initial state probabilities.
    P : array-like of shape (n_states, n_states)
        The transition probabilities across states.

    Returns:
    --------
    a : array-like of shape (n_samples, n_states)
        The alpha values.
    b : array-like of shape (n_samples, n_states)
        The beta values.
    sc : array-like of shape (n_samples,)
        The scaling factors.

    """
    T,K = L.shape

    #minreal = sys.float_info.min
    #maxreal = sys.float_info.max

    a = np.zeros((T,K))
    b = np.zeros((T,K))
    sc = np.zeros(T)

    a[0,:] = Pi * L[0,:]
    sc[0] = np.sum(a[0,:])
    a[0,:] = a[0,:] / sc[0]

    for t in range(1,T):
        a[t,:] = (a[t-1,:] @ P) * L[t,:]
        sc[t] = np.sum(a[t,:])
        #if sc[t]<minreal: sc[t] = minreal
        a[t,:] = a[t,:] / sc[t]

    b[T-1,:] = np.ones((1,K)) / sc[T-1]
    for t in range(T-2,-1,-1):
        b[t,:] = ( (b[t+1,:] * L[t+1,:]) @ P.T ) / sc[t]
        #bad = b[t,:]>maxreal
        #if bad.sum()>0: b[t,bad] = maxreal

    return a,b,sc


def compute_qstar(L,Pi,P):
    """Compute the most probable state sequence.

    Parameters:
    -----------
    L : array-like of shape (n_samples, n_states)
        The L matrix.
    Pi : array-like with shape (n_states,)
        The initial state probabilities.
    P : array-like of shape (n_states, n_states)
        The transition probabilities across states.

    Returns:
    --------
    qstar : array-like of shape (n_samples, n_states)
        The most probable state sequence.

    """
    T,K = L.shape

    delta = np.zeros((T,K))
    psi = np.zeros((T,K)).astype(int)
    qstar = np.zeros((T,K))

    delta[0,:] = Pi * L[0,:]
    delta[0,:] = delta[0,:] / np.sum(delta[0,:])

    for t in range(1,T):
        for k in range(K):
            v = delta[t-1,:] * P[:,k]
            mv = np.amax(v)
            delta[t,k] = mv * L[t,k]
            psi[t,k] = np.where(mv==v)[0][0]
        delta[t,:] = delta[t,:] / np.sum(delta[t,:])

    id = np.where(delta[-1,:]==np.amax(delta[-1,:]))[0][0]
    qstar[T-1,id] = 1

    for t in range(T-2,-1,-1):
        id0 = np.where(qstar[t+1,:]==1)[0][0]
        id = psi[t+1,id0]
        qstar[t,id] = 1

    return qstar

## Mathematical functions related to the free energy

def gauss1d_kl(mu_q, sigma_q, mu_p, sigma_p):

    """Computes the KL divergence between two univariate Gaussian distributions.

    Parameters:
    -----------
    mu_q : float of shape (n_parcels,)
        The mean of the first Gaussian distribution.
    sigma_q : float of shape (n_parcels, n_parcels)
        The variance of the first Gaussian distribution.
    mu_p : float of shape (n_parcels,)
        The mean of the second Gaussian distribution.
    sigma_p : float of shape (n_parcels, n_parcels)
        The variance of the second Gaussian distribution.

    Returns:
    --------
    D : float
        The KL divergence between the two Gaussian distributions.

    """
    D = 0.5 * math.log(sigma_p/sigma_q) + \
        0.5 * ((mu_q - mu_p)**2) / sigma_p + \
        0.5 * sigma_q / sigma_p
    return D


def gauss_kl(mu_q, sigma_q, mu_p, sigma_p):
    """Computes the KL divergence between two multivariate Gaussian distributions.

    Parameters:
    -----------
    mu_q : float of shape (n_parcels,)
        The mean of the first Gaussian distribution.
    sigma_q : float of shape (n_parcels, n_parcels)
        The variance of the first Gaussian distribution.
    mu_p : float of shape (n_parcels,)
        The mean of the second Gaussian distribution.
    sigma_p : float of shape (n_parcels, n_parcels)
        The variance of the second Gaussian distribution.

    Returns:
    --------
    D : float
        The KL divergence between the two Gaussian distributions.

    """

    if len(mu_q) == 1:
        D = gauss1d_kl(mu_q[0], sigma_q[0,0], mu_p[0], sigma_p[0,0])
    else:
        N = mu_q.shape[0]
        isigma_p = np.linalg.inv(sigma_p)
        (sign, logdet) = np.linalg.slogdet(sigma_p)
        t1 = sign * logdet
        (sign, logdet) = np.linalg.slogdet(sigma_q)
        t2 = sign * logdet        
        d = mu_q - mu_p
        D = 0.5 * (t1 - t2 - N + np.trace(isigma_p @ sigma_q) + ((d.T @ isigma_p) @ d))
    return D


def gamma_kl(shape_q,rate_q,shape_p,rate_p):
    """Computes the Kullback-Leibler divergence between two Gamma distributions with shape and rate parameters.

    The Kullback-Leibler divergence is a measure of how different two probability distributions are.

    This implementation follows the formula presented here (https://statproofbook.github.io/P/gam-kl) from the book "KL-Divergences of Normal, Gamma, Dirichlet and Wishart densities" by Penny, William D. in 2001.

    Parameters
    ----------
    shape_q : float or numpy.ndarray
        The shape parameter of the first Gamma distribution.
    rate_q : float or numpy.ndarray
        The rate parameter of the first Gamma distribution.
    shape_p : float or numpy.ndarray
        The shape parameter of the second Gamma distribution.
    rate_p : float or numpy.ndarray
        The rate parameter of the second Gamma distribution.

    Returns
    -------
    D : float or numpy.ndarray
        The Kullback-Leibler divergence between the two Gamma distributions.
    """

    D = shape_p * np.log(rate_q / rate_p) \
        + scipy.special.gammaln(shape_p) - scipy.special.gammaln(shape_q) \
        + (shape_q - shape_p) * scipy.special.psi(shape_q) \
        - (rate_q - rate_p) * shape_q / rate_q

    return D


def wishart_kl(shape_q,C_q,shape_p,C_p):
    """Computes the Kullback-Leibler (KL) divergence between two Wishart distributions.

    Parameters:
    -----------
    shape_q : float
        Shape parameter of the first Wishart distribution.
    C_q : ndarray of shape (n_parcels, n_parcels)
        Scale parameter of the first Wishart distribution.
    shape_p : float
        Shape parameter of the second Wishart distribution.
    C_p : ndarray of shape (n_parcels, n_parcels)
        Scale parameter of the second Wishart distribution.

    Returns:
    --------
    D : float
        KL divergence from the first to the second Wishart distribution.

    """ 

    def L(shape,C):
        N = C.shape[0]
        PsiWish_alphasum = 0
        for j in range(1,N+1): 
            PsiWish_alphasum += scipy.special.psi(0.5 * (shape + 1 - j))
        (s, logdet) = np.linalg.slogdet(C)
        ldetWishB = s * logdet
        return (PsiWish_alphasum - ldetWishB + N * math.log(2))

    def logZ(shape,C):
        N = C.shape[0]
        t1 = shape * N / 2 * math.log(2)
        (s, logdet) = np.linalg.slogdet(C)
        t2 = - (shape/2) * s * logdet
        t3 = (N*(N-1)/4) * math.log(math.pi)
        for j in range(1,N+1):  
            t3 += scipy.special.gammaln(0.5 * (shape + 1 - j))
        return (t1 + t2 + t3)
        
    N = C_q.shape[0]
    iC_q = np.linalg.inv(C_q)

    t1 = ((shape_q - N - 1)/2) * L(shape_q,C_q) - \
            ((shape_p - N - 1)/2) * L(shape_p,C_p)
    t2 = - N * shape_q / 2
    t3 = shape_q * np.trace(C_p @ iC_q) / 2
    t4 = logZ(shape_p,C_p) - logZ(shape_q,C_q)

    return (t1 + t2 + t3 + t4)


def dirichlet_kl(alpha_q,alpha_p):
    """Computes the Kullback-Leibler divergence between two Dirichlet distributions with parameters alpha_q and alpha_p.

    Parameters:
    -----------
        alpha_q : Array of shape (n_states,)
            The concentration parameters of the first Dirichlet distribution.
        alpha_p : Array of shape (n_states,)
            The concentration parameters of the second Dirichlet distribution.

    Returns:
    --------
        float: The Kullback-Leibler divergence between the two Dirichlet distributions.
    """
    sum_alpha_q = np.sum(alpha_q)
    sum_alpha_p = np.sum(alpha_p)
    t1 = + scipy.special.gammaln(sum_alpha_q) - scipy.special.gammaln(sum_alpha_p) \
        + np.sum(scipy.special.gammaln(alpha_p)) - np.sum(scipy.special.gammaln(alpha_q))
    t2 = np.sum( (alpha_q - alpha_p) * (scipy.special.psi(alpha_q) - scipy.special.psi(sum_alpha_q)) )
    return ( t1 + t2 )


def Gamma_entropy(Gamma,Xi,indices):
    """Computes the entropy of a Gamma distribution and a sequence of transition probabilities Xi.

    Parameters:
    -----------
        Gamma : Array-like of shape (n_samples, n_states)
            The posterior probabilities of a hidden variable.
        Xi : Array-like of shape (n_samples - n_sessions, n_states, n_states)
            The joint probability of past and future states conditioned on data.
        indices : Array-like of shape (n_sessions, 2)
            The indices of the segments.

    Returns:
    --------
        float: The entropy of the Gamma distribution and the sequence of transition probabilities.

    """
    
    minreal = sys.float_info.min
    # initial point
    Gamma_0 = Gamma[indices[:,0]]
    Gamma_0[Gamma_0 < minreal] = minreal
    Entr = -np.sum(Gamma_0 * np.log(Gamma_0))
    # transitions
    Xi[Xi < minreal] = minreal
    Xi_norm = np.zeros(Xi.shape)
    for k in range(Gamma.shape[1]):
        Xi_norm[:,k,:] = Xi[:,k,:] / np.expand_dims(np.sum(Xi[:,k,:],axis=1),axis=1)
    Entr -= np.sum(Xi * np.log(Xi_norm))
    return Entr
    
