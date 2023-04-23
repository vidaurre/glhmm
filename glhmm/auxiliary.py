#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary functions - Gaussian Linear Hidden Markov Model
@author: Diego Vidaurre 2022
"""

import sys

from numba import njit
import numpy as np


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
        The start and end indices of each trial/session in the input data.

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
        The start and end indices of each trial/session in the input data.

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
        The start and end indices of each trial/session in the input data.

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


@njit
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
    psi = np.zeros((T,K)).astype(np.int64)
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


def Gamma_entropy(Gamma,Xi,indices):
    """Computes the entropy of a Gamma distribution and a sequence of transition probabilities Xi.

    Parameters:
    -----------
        Gamma : Array-like of shape (n_samples, n_states)
            The posterior probabilities of a hidden variable.
        Xi : Array-like of shape (n_samples - n_sessions, n_states, n_states)
            The joint probability of past and future states conditioned on data.
        indices : Array-like of shape (n_sessions, 2)
            The start and end indices of each trial/session in the input data.

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
    
