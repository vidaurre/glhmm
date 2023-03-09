#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some public useful functions - Gaussian Linear Hidden Markov Model
@author: Diego Vidaurre 2023
"""

import numpy as np
import statistics
import math

def get_FO(Gamma,indices,summation=False):
    """Calculates the fractional occupancy of each state.
    
    Parameters:
    -----------
    Gamma : array-like, shape (n_samples, n_states)
        The state probability time series.
    indices : array-like, shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.
    summation : bool, optional, default=False
        If True, the sum of each row is not normalized, otherwise it is.

    Returns:
    --------
    FO : array-like, shape (n_sessions, n_states)
        The fractional occupancy of each state per session.

    """

    N = indices.shape[0]
    K = Gamma.shape[1]
    FO = np.zeros((N,K))
    for j in range(N):
        ind = np.arange(indices[j,0],indices[j,1])
        FO[j,:] = np.sum(Gamma[ind,:],axis=0)
        if not summation:
            FO[j,:] /= np.sum(FO[j,:])
    return FO
    

def get_maxFO(Gamma,indices):
    """Calculates the maximum fractional occupancy per session.

    The first argument can also be a viterbi path (vpath).

    Parameters:
    -----------
    Gamma : array-like of shape (n_samples, n_states); or a vpath, array of shape (n_samples,)
        The Gamma represents the state probability timeseries and the vpath represents the most likely state sequence.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.

    Returns:
    --------
    maxFO: array-like of shape (n_sessions,)
        The maximum fractional occupancy across states for each trial/session

    Notes:
    ------
    The maxFO is useful to assess the amount of `state mixing`. For more information, see [^1].

    References:
    -----------
    [^1]: Ahrends, R., et al. (2022). Data and model considerations for estimating time-varying functional connectivity in fMRI. NeuroImage 252, 119026.
           https://pubmed.ncbi.nlm.nih.gov/35217207/)

    """
    FO = get_FO(Gamma,indices)
    return np.max(FO,axis=1)


def get_state_evoked_response(Gamma,indices):
    """Calculates the state evoked response 

    The first argument can also be a viterbi path (vpath).

    Parameters:
    ---------------
    Gamma : array-like of shape (n_samples, n_states), or a vpath array of shape (n_samples,)
        The Gamma represents the state probability timeseries and the vpath represents the most likely state sequence.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.

    Returns:
    ------------
    ser : array-like of shape (n_samples, n_states)
        The state evoked response matrix.

    Raises:
    -------
    Exception
        If the input data violates any of the following conditions:
        - There is only one trial/session
        - Not all trials/sessions have the same length.
    """

    N = indices.shape[0]
    if N == 1: 
        raise Exception("There is only one segment / trial")
    T = indices[:,1] - indices[:,0]
    if not(np.all(T[0]==T)):
        raise Exception("All segments / trials must have the same length")
    K = Gamma.shape[1]
    T = T[0]

    ser = np.mean(np.reshape(Gamma,(T,N,K),order='F'),axis=1)
    return ser


def get_switching_rate(Gamma,indices):
    """Calculates the switching rate.

    The first argument can also be a viterbi path (vpath).

    Parameters:
    ---------------
    Gamma : array-like of shape (n_samples, n_states), or a vpath array of shape (n_samples,)
        The Gamma represents the state probability timeseries and the vpath represents the most likely state sequence.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.

    Returns:
    ------------
    SR : array-like of shape (n_sessions, n_states)
        The switching rate matrix.

    """
    N = indices.shape[0]
    K = Gamma.shape[1]
    SR = np.zeros((N,K))
    for j in range(N):
        ind = np.arange(indices[j,0],indices[j,1])
        SR[j,:] = np.mean(np.abs(Gamma[ind[1:],:]-Gamma[ind[0:-1],:]),axis=0)
    return SR


def get_visits(vpath,k,threshold=0):
    """Computes a list of visits for state k, given a viterbi path (vpath).

    Parameters:
    ---------------
    vpath : array-like of shape (n_samples,)
        The viterbi path represents the most likely state sequence.
    k : int
        The state for which to compute the visits.
    threshold : int, optional, default=0
        A threshold value used to exclude visits with a duration below this value.

    Returns:
    ------------
    lengths : list of floats
        A list of visit durations for state k, where each duration is greater than the threshold.
    onsets : list of ints
        A list of onset time points for each visit.

    Notes:
    ------
    A visit to state k is defined as a contiguous sequence of time points in which state k is active.

    """

    lengths = []
    onsets = []
    T = vpath.shape[0]
    vpath_k = vpath[:,k]
    t = 0 
    while t < T: 
        t += np.where(vpath_k[t:]==1)[0]
        if len(t)==0: 
            break
        t = t[0]
        onsets.append(t)
        tend = np.where(vpath_k[t:]==0)[0]
        if len(tend)==0: 
            length_visit = len(vpath_k)-t
            if length_visit > threshold: lengths.append(float(length_visit))
            break
        tend = tend[0]
        length_visit = tend
        if length_visit > threshold: lengths.append(float(length_visit))
        t += tend
    return lengths,onsets


def get_life_times(vpath,indices,threshold=0):
    """Calculates the average, median and maximum life times for each state.

    Parameters:
    -----------
    vpath : array-like of shape (n_samples,)
        The viterbi path represents the most likely state sequence.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.
    threshold : int, optional, default=0
        A threshold value used to exclude visits with a duration below this value.

    Returns:
    --------
    meanLF : array-like of shape (n_sessions, n_states)
        The average visit duration for each state in each trial/session.
    medianLF : array-like of shape (n_sessions, n_states)
        The median visit duration for each state in each trial/session.
    maxLF : array-like of shape (n_sessions, n_states)
        The maximum visit duration for each state in each trial/session.

    Notes:
    ------
    A visit to a state is defined as a contiguous sequence of time points in which the state is active.
    The duration of a visit is the number of time points in the sequence.
    This function uses the `get_visits` function to compute the visits and exclude those below the threshold.

    """
    N = indices.shape[0]
    K = vpath.shape[1]    
    meanLF = np.zeros((N,K)) 
    medianLF = np.zeros((N,K)) 
    maxLF = np.zeros((N,K)) 
    for j in range(N):
        ind = np.arange(indices[j,0],indices[j,1]).astype(int)
        for k in range(K):
            visits,_ = get_visits(vpath[ind,:],k,threshold=0)
            if len(visits) > 0:
                meanLF[j,k] = statistics.mean(visits)
                medianLF[j,k] = statistics.median(visits)
                maxLF[j,k] = max(visits)
    return meanLF, medianLF, maxLF


def get_state_onsets(vpath,indices,threshold=0):
    """Calculates the state onsets, i.e., the time points when each state activates.

    Parameters:
    ---------------
    vpath : array-like of shape (n_samples, n_states)
        The viterbi path represents the most likely state sequence.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.
    threshold : int, optional, default=0
        A threshold value used to exclude visits with a duration below this value.

    Returns:
    --------
    onsets : list of lists of ints
        A list of the time points when each state activates for each trial/session.

    Notes:
    ------
    A visit to a state is defined as a contiguous sequence of time points in which the state is active.
    This function uses the `get_visits` function to compute the visits and exclude those below the threshold.

    """

    N = indices.shape[0]
    K = vpath.shape[1]    
    onsets = []
    for j in range(N):
        onsets_j = []
        ind = np.arange(indices[j,0],indices[j,1]).astype(int)
        for k in range(K):
            _,onsets_k = get_visits(vpath[ind,:],k,threshold=0)
            onsets_j.append(onsets_k)
        onsets.append(onsets_j)
    return onsets


def get_FO_entropy(Gamma,indices):
    """Calculates the entropy of each session, if we understand fractional occupancies as probabilities.

    Parameters:
    --------------
    Gamma : array-like of shape (n_samples, n_states)
        The Gamma represents the state probability timeseries.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.

    Returns:
    --------
    entropy : array-like of shape (n_sessions,)
        The entropy of each session.

    """  
    fo = get_FO(Gamma,indices)
    N,K = fo.shape
    entropy = np.zeros(N)
    for j in range(N):
        for k in range(K):
            if fo[j,k] == 0: continue
            entropy[j] -= math.log(fo[j,k]) * fo[j,k]
    return entropy
    

def get_state_evoked_response_entropy(Gamma,indices):
    """Calculates the entropy of each time point, if we understand state evoked responses as probabilities.

    Parameters:
    ---------------
    Gamma: array-like of shape (n_samples, n_states)
        The Gamma represents the state probability timeseries.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.

    Returns:
    ------------
    entropy: array-like of shape (n_samples,)
        The entropy of each time point.

    """  
    ser = get_state_evoked_response(Gamma,indices)
    T,K = ser.shape
    entropy = np.zeros(T)
    for t in range(T):
        for k in range(K):
            if ser[t,k] == 0: continue
            entropy[t] -= math.log(ser[t,k]) * ser[t,k]
    return entropy



