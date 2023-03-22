#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic graphics - Gaussian Linear Hidden Markov Model
@author: Diego Vidaurre 2023
"""
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

from . import utils

def show_trans_prob_mat(self,t=None,show_diag=True,show_colorbar=True):
    """Displays a heatmap of the transition probability matrix.

    Parameters:
    -----------
    t : 1D array or None, default=None
        The time index for the transition probability matrix to be displayed. 
        If None (default), displays the entire matrix.
    show_diag : bool
        Whether to show the diagonal elements of the matrix. If False, sets the 
        diagonal elements to zero and normalizes the rows of the matrix.
    show_colorbar : bool, default=True
        Whether to display the colorbar.
    """
    P = np.copy(self.P)
    if only_active_states:
        P = P[hmm.active_states,hmm.active_states]
        
    K = P.shape[0]

    if not show_diag:
        for k in range(P.shape[0]):
            P[k,k] = 0
            P[k,:] = P[k,:] / np.sum(P[k,:])

    _,ax = plt.subplots()
    g = sb.heatmap(ax=ax,data=P,\
        cmap='bwr',xticklabels=np.arange(K), yticklabels=np.arange(K),
        square=True,cbar=show_colorbar)
    for k in range(K):
        g.plot([0, K],[k, k], '-k')
        g.plot([k, k],[0, K], '-k')

    ax.axhline(y=0, color='k',linewidth=4)
    ax.axhline(y=K, color='k',linewidth=4)
    ax.axvline(x=0, color='k',linewidth=4)
    ax.axvline(x=K, color='k',linewidth=4)


def show_Gamma(Gamma,tlim=None,Hz=1,palette='Oranges'):
    """Displays the activity of the hidden states as a function of time.

    Parameters:
    -----------
    Gamma : array of shape (n_samples, n_states)
        The state timeseries probabilities.
    tlim : 2 x 1 array or None, default=None
        The time interval to be displayed. If None (default), displays the 
        entire sequence.
    Hz : int, default=1
        The frequency of the signal, in Hz.
    palette : str, default = 'Oranges'
        The name of the color palette to use.
    """
    Hz = 100

    T,K = Gamma.shape

    x = np.round(np.linspace(0.0, 256-1, K)).astype(int)
    cmap = plt.get_cmap('plasma').colors
    cols = np.zeros((K,3))
    for k in range(K):
        cols[k,:] = cmap[x[k]]

    if tlim is None:
        df = pd.DataFrame(Gamma, index=np.arange(T)/Hz)
    else:
        T = tlim[1] - tlim[0]
        df = pd.DataFrame(Gamma[tlim[0]:tlim[1],:], index=np.arange(T)/Hz)
    df = df.divide(df.sum(axis=1), axis=0)
    ax = df.plot(kind='area', stacked=True,ylim=(0,1),legend=False,color=cols)

    ax.set_ylabel('Percent (%)')
    ax.margins(0,0)

    plt.show()

def show_temporal_statistic(s):
    """Displays box plots of a given temporal aspect of the states
    (e.g., fractional occupancies, switching rates, life times) `s`.

    Params:
    -----------
    s : array-like
        The temporal statistic to plot. Can be one of the following:
        - `utils.get_FO(Gamma, indices)`
        - `utils.get_switching_rate(Gamma, indices)`
        - `utils.get_life_times(vpath, indices)[0]`
        - `utils.get_FO_entropy(Gamma, indices)`
        
    Examples:
    ---------
    >>> indices = [0, 1, 2]
    >>> Gamma = np.random.rand(100, 3)
    >>> s = utils.get_FO(Gamma, indices)
    >>> show_temporal_statistic(s)
    """
    s = eval("utils.get_" + statistic)(Gamma,indices)
    if statistic not in ["FO","switching_rate","life_times","entropy"]:
        raise Exception("statistic has to be 'FO','switching_rate','life_times' or 'entropy'") 
    N,K = s.shape

    sb.set(style='whitegrid')
    if type_plot=='boxplot':
        if N < 10:
            raise Exception("Too few sessions for a boxplot; use barplot") 
        sb.boxplot(data=s,palette='plasma')
    elif type_plot=='barplot':
        sb.barplot(data=np.concatenate((s,s)),palette='plasma', errorbar=None)
    elif type_plot=='matrix':
        if N < 2:
            raise Exception("There is only one session; use barplot") 
        fig,ax = plt.subplots()
        labels_x = np.round(np.linspace(0,N,5)).astype(int)
        pos_x = np.linspace(0,N,5)
        if K > 10: labels_y = np.linspace(0,K-1,5)
        else: labels_y = np.arange(K)
        im = plt.imshow(s.T,aspect='auto')
        plt.xticks(pos_x, labels_x)
        plt.yticks(labels_y, labels_y)
        ax.set_xlabel('Sessions')
        ax.set_ylabel('States')
        fig.tight_layout()


def show_beta(hmm,only_active_states=False,X=None,Y=None,show_average=None):
    """Displays the beta coefficients of a given HMM.

    Parameters:
    -----------
    hmm: HMM object
        An instance of the HMM class containing the beta coefficients to be visualized.
    only_active_states: bool, optional, default=False
        If True, only the beta coefficients of active states are shown.
    X: numpy.ndarray, optional, default=None
        The timeseries of set of variables 1.
    Y: numpy.ndarray, optional, default=None
        The timeseries of set of variables 2.
    show_average: bool, optional, default=None
        If True, an additional row of the average beta coefficients is shown.
    """
    if show_average is None:
        show_average = not ((X is None) or (Y is None))
    
    K = hmm.hyperparameters["K"]
    beta = hmm.get_betas()

    if only_active_states:
        idx,_ = np.where(hmm.active_states)
        beta = beta[:,:,idx]
    else:
        idx = np.arange(K)

    (p,q,K) = beta.shape

    if show_average:
        Yr = np.copy(Y)
        Yr -= np.expand_dims(np.mean(Y,axis=0), axis=0)   
        b0 = np.linalg.inv(X.T @ X) @ (X.T @ Yr)
        K += 1
        B = np.zeros((p,q,K))
        B[:,:,0:K-1] = beta
        B[:,:,-1] = b0 
    else:
        B = beta

    Bstar1 = np.zeros((p,q,K,K))
    for k in range(K): Bstar1[:,:,k,:] = B
    Bstar2 = np.zeros((p,q,K,K))
    for k in range(K): Bstar2[:,:,:,k] = B   

    I1 = np.zeros((p,q,K,K),dtype=object)
    for j in range(q): I1[:,j,:,:] = str(j)
    I2 = np.zeros((p,q,K,K),dtype=object)
    for k in range(K): 
        if show_average and (k==(K-1)): 
            I2[:,:,k,:] = 'Average'
        else:
            I2[:,:,k,:] = 'State ' + str(k)
    I3 = np.zeros((p,q,K,K),dtype=object)
    for k in range(K): 
        if show_average and (k==(K-1)): 
            I3[:,:,:,k] = 'Average'
        else:
            I3[:,:,:,k] = 'State ' + str(k)

    Bstar1 = np.expand_dims(np.reshape(Bstar1,p*q*K*K,order='F'),axis=0)
    Bstar2 = np.expand_dims(np.reshape(Bstar2,p*q*K*K,order='F'),axis=0)
    I1 = np.expand_dims(np.reshape(I1,p*q*K*K,order='F'),axis=0)
    I2 = np.expand_dims(np.reshape(I2,p*q*K*K,order='F'),axis=0)
    I3 = np.expand_dims(np.reshape(I3,p*q*K*K,order='F'),axis=0)

    B = np.concatenate((Bstar1,Bstar2,I1,I2,I3),axis=0).T
    df = pd.DataFrame(B,columns=('x','y','Variable','beta x','beta y'))

    sb.relplot(x='x', 
        y='y', 
        hue='Variable', 
        col="beta x", row="beta y",
        data=df,
        palette='cool', 
        height=8)
    