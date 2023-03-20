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
    t : float or None, default=None
        The time index for the transition probability matrix to be displayed. 
        If None (default), displays the entire matrix.
    show_diag : bool
        Whether to show the diagonal elements of the matrix. If False, sets the 
        diagonal elements to zero and normalizes the rows of the matrix.
    show_colorbar : bool, default=True
        Whether to display the colorbar.
    """
    P = np.copy(self.P)
    K = P.shape[0]

    if not show_diag:
        for k in range(P.shape[0]):
            P[k,k] = 0
            P[k,:] = P[k,:] / np.sum(P[k,:])

    fig = plt.figure()
    gs = fig.add_gridspec(1,2)
    ax = fig.add_subplot(gs[0, 0])
    g = sb.heatmap(ax=ax,data=P,\
        cmap='bwr',xticklabels=False, yticklabels=False,square=True,cbar=show_colorbar)
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
    tlim : tuple or None, default=None
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
        T = tlim[1] - tlim[0] - 1
        df = pd.DataFrame(Gamma[tlim[0]:tlim[1],:], index=np.arange(T)/Hz)
    df = df.divide(df.sum(axis=1), axis=0)
    ax = df.plot(kind='area', stacked=True,ylim=(0,1),legend=False,color=cols)

    ax.set_ylabel('Percent (%)')
    ax.margins(0,0)

    plt.show()


def show_temporal_statistic(s):
    """Displays box plots of a given temporal aspect of the states
    (e.g., fractional occupancies, switching rates, life times) `s`.

    Parameters
    ----------
    s : array-like
        The temporal statistic to plot. Can be one of the following:
        - `utils.get_FO(Gamma, indices)`
        - `utils.get_switching_rate(Gamma, indices)`
        - `utils.get_life_times(vpath, indices)[0]`
        - `utils.get_FO_entropy(Gamma, indices)`
        
    Examples
    --------
    >>> indices = [0, 1, 2]
    >>> Gamma = np.random.rand(100, 3)
    >>> s = utils.get_FO(Gamma, indices)
    >>> show_temporal_statistic(s)
    """

    sb.set(style='whitegrid')
    sb.boxplot(data=s,palette='plasma')






