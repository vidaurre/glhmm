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
    """
    Calculates the fractional occupancy.
    The first argument can also be a viterbi path
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
    """
    Calculates the max fractional occupancy per subject,
    useful to assess the amount of "state mixing".
    The first argument can also be a viterbi path
    """
    FO = get_FO(Gamma,indices)
    return np.max(FO,axis=1)


def get_state_evoked_response(Gamma,indices):
    """
    Calculates the state evoked response 
    (only defined if all segments have the same length).
    The first argument can also be a viterbi path
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
    """
    Calculates the switching rate.
    The first argument can also be a viterbi path
    """
    N = indices.shape[0]
    K = Gamma.shape[1]
    SR = np.zeros((N,K))
    for j in range(N):
        ind = np.arange(indices[j,0],indices[j,1])
        SR[j,:] = np.mean(np.abs(Gamma[ind[1:],:]-Gamma[ind[0:-1],:]),axis=0)
    return SR


def get_visits(vpath,k,threshold=0):
    """
    Computes a list of visits for state k, given viterbi path vpath
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
    """
    Calculates the average, median and max life times.
    The first argument must be a viterbi path
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
    """
    Calculates the state onsets, ie when each state activates.
    The first argument must be a viterbi path
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
    """
    Calculates the entropy of each session, 
    if we understand fractional occupancies as probabilities.
    The first argument must be a viterbi path
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
    """
    Calculates the entropy of each time point, 
    if we understand state evoked responses as probabilities.
    The first argument must be a viterbi path
    """  
    ser = get_state_evoked_response(Gamma,indices)
    T,K = ser.shape
    entropy = np.zeros(T)
    for t in range(T):
        for k in range(K):
            if ser[t,k] == 0: continue
            entropy[t] -= math.log(ser[t,k]) * ser[t,k]
    return entropy



