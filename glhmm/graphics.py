#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic graphics - Gaussian Linear Hidden Markov Model
@author: Diego Vidaurre & Nick Yao Larsen 2025
"""
import os
import math
import logging
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import trange

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import cm, ticker, gridspec
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap, to_rgba_array
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter,MaxNLocator

import seaborn as sb

from nilearn import plotting, surface, image
from nilearn.surface import vol_to_surf
from nilearn.image import resample_to_img
from nilearn._utils.niimg_conversions import check_niimg_3d

from . import utils
from glhmm.io import *





def show_trans_prob_mat(hmm,only_active_states=False,show_diag=True,show_colorbar=True):
    """Displays the transition probability matrix of a given HMM.

    Parameters:
    -----------
    hmm: HMM object
        An instance of the HMM class containing the transition probability matrix to be visualized.
    only_active_states (bool), optional, default=False
        Whether to display only active states or all states in the matrix.
    show_diag (bool), optional, defatult=True
        Whether to display the diagonal elements of the matrix or not.
    show_colorbar (bool), optional, default=True
        Whether to display the colorbar next to the matrix or not.
    """
    
    P = np.copy(hmm.P)
    if only_active_states:
        P = P[hmm.active_states,hmm.active_states]
        
    K = P.shape[0]

    if not show_diag:
        for k in range(P.shape[0]):
            P[k,k] = 0
            P[k,:] = P[k,:] / np.sum(P[k,:])

    _,axes = plt.subplots()
    g = sb.heatmap(ax=axes,data=P,\
        cmap='bwr',xticklabels=np.arange(K), yticklabels=np.arange(K),
        square=True,cbar=show_colorbar)
    for k in range(K):
        g.plot([0, K],[k, k], '-k')
        g.plot([k, k],[0, K], '-k')

    axes.axhline(y=0, color='k',linewidth=4)
    axes.axhline(y=K, color='k',linewidth=4)
    axes.axvline(x=0, color='k',linewidth=4)
    axes.axvline(x=K, color='k',linewidth=4)


def show_Gamma(Gamma, line_overlay=None, tlim=None, Hz=1, palette='viridis'):
    """Displays the activity of the hidden states as a function of time.
    
    Parameters:
    -----------
    Gamma : array of shape (n_samples, n_states)
        The state timeseries probabilities.
    line_overlay : array of shape (n_samples, 1)
        A secondary related data type to overlay as a line.
    tlim : 2x1 array or None, default=None
        The time interval to be displayed. If None (default), displays the 
        entire sequence.
    Hz : int, default=1
        The frequency of the signal, in Hz.
    palette (str), default = 'Oranges'
        The name of the color palette to use.
    """

    T,K = Gamma.shape

    # Setup colors
    x = np.round(np.linspace(0.0, 256-1, K)).astype(int)
    # cmap = plt.get_cmap('plasma').colors
    cmap = plt.get_cmap(palette)
    cmap = cmap(np.arange(0, cmap.N))[:, :3]
    
    colors = np.zeros((K,3))
    for k in range(K):
        colors[k,:] = cmap[x[k]]

    # Setup data according to given limits
    if tlim is not None:
        T = tlim[1] - tlim[0]
        data = Gamma[tlim[0] : tlim[1], :]
        if line_overlay is not None:
            line = line_overlay[tlim[0] : tlim[1]].copy()
    else: 
        data = Gamma
    
    df = pd.DataFrame(data, index=np.arange(T)/Hz)
    df = df.divide(df.sum(axis=1), axis=0)
    
    # Plot Gamma area
    ax = df.plot(
        kind='area',
        stacked=True,
        ylim=(0,1),
        legend=False,
        color=colors
    )
    
    # Overlay line if given
    if line_overlay is not None:
        df2 = pd.DataFrame(line, index=np.arange(T)/Hz)
        ax2 = ax.twinx()
        df2.plot(ax=ax2, legend=False, color="black")
        ax2.set(ylabel = '')
    
    # Adjust axis specifications
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set(
        title  = "",
        xlabel = 'Time [s]',
        ylabel = 'State probability')
    ax.margins(0,0)
    
    plt.show()


def show_temporal_statistic(Gamma,indices, statistic='FO',type_plot='barplot'):
    """Plots a statistic over time for a set of sessions.

    Parameters:
    -----------
    Gamma : array of shape (n_samples, n_states)
        The state timeseries probabilities.
    indices: numpy.ndarray of shape (n_sessions,)
        The session indices to plot.
    statistic(str), default='FO'
        The statistic to compute and plot. Can be 'FO', 'switching_rate' or 'FO_entropy'.
    type_plot(str), default='barplot'
        The type of plot to generate. Can be 'barplot', 'boxplot' or 'matrix'.

    Raises:
    -------
    Exception
        - Statistic is not one of 'FO', 'switching_rate' or 'FO_entropy'.
        - type_plot is 'boxplot' and there are less than 10 sessions.
        - type_plot is 'matrix' and there is only one session.
    """
    
    s = eval("utils.get_" + statistic)(Gamma,indices)
    if statistic not in ["FO","switching_rate","FO_entropy"]:
        raise Exception("statistic has to be 'FO','switching_rate' or 'FO_entropy'") 
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


def show_beta(hmm,only_active_states=True,recompute_states=False,
              X=None,Y=None,Gamma=None,show_average=None,alpha=1.0):
    """
    Displays the beta coefficients of a given HMM.
    The beta coefficients can be extracted directly from the HMM structure or reestimated from the data;
    for the latter, X, Y and Gamma need to be provided as parameters. 
    This is useful for example if one has run the model on PCA space, 
    but wants to show coefficients in the original space.
    
    Parameters:
    -----------
    hmm: HMM object
        An instance of the HMM class containing the beta coefficients to be visualized.
    only_active_states(bool), optional, default=False
        If True, only the beta coefficients of active states are shown.
    recompute_states(bool), optional, default=False
        If True, the betas will be recomputed from the data and the state time courses
    X: numpy.ndarray, optional, default=None
        The timeseries of set of variables 1.
    Y: numpy.ndarray, optional, default=None
        The timeseries of set of variables 2.
    Gamma: numpy.ndarray, optional, default=None
        The state time courses
    show_average(bool), optional, default=None
        If True, an additional row of the average beta coefficients is shown.
    alpha: float, optional, default=0.1
        The regularisation parameter to be applied if the betas are to be recomputed.

    """
    
    if show_average is None:
        show_average = not ((X is None) or (Y is None))
    
    K = hmm.get_betas().shape[2]
    
    if recompute_states:
        if (Y is None) or (X is None) or (Gamma is None):
            raise Exception("The data (X,Y) and the state time courses (Gamma) need \
                             to be provided if recompute_states is True ")
        (p,q) = (X.shape[1],Y.shape[1])
        beta = np.zeros((p,q,K))
        for k in range(K):
            if hmm.hyperparameters["model_mean"] != 'no':
                m = (np.expand_dims(Gamma[:,k],axis=1).T @ Yr) / np.sum(Gamma[:,k])
                Yr = Y - np.expand_dims(m, axis=0)
            else:
                Yr = Y
            beta[:,:,k] =  np.linalg.inv((X * np.expand_dims(Gamma[:,k],axis=1)).T @ X + alpha * np.eye(p)) @ \
                ((X * np.expand_dims(Gamma[:,k],axis=1)).T @ Yr)
    else:
        beta = hmm.get_betas()
        (p,q,_) = beta.shape

    if only_active_states:
        idx = np.where(hmm.active_states)[0]
        beta = beta[:,:,idx]
        K = beta.shape[2]
    else:
        idx = np.arange(K)

    if show_average:
        Yr = Y - np.expand_dims(np.mean(Y,axis=0), axis=0) 
        b0 = np.linalg.inv(X.T @ X + alpha * np.eye(p)) @ (X.T @ Yr)
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

    g = sb.relplot(x='x', 
        y='y', 
        s=25,
        hue='Variable', 
        col="beta x", row="beta y",
        data=df,
        palette='cool')
    
    for item, ax in g.axes_dict.items():
        ax.grid(False, axis='x')
        ax.set_title('')

    # def show_r2(r2=None,hmm=None,Gamma=None,X=None,Y=None,indices=None,show_average=False):

    #     if r2 is None:
    #         if (Y is None) or (indices is None):
    #             raise Exception("Y and indices (and maybe X) has to be specified if r2 is not provided")
    #         r2 = hmm.get_r2(X,Y,Gamma,indices)

    #     if show_average:
    #         if (Y is None) or (indices is None):
    #             raise Exception("Y and indices (and maybe X) has to be specified if the average is to computed") 

    #         r20 = hmm.get_r2(X,Y,Gamma,indices)

    #         for j in range(N):

    #             tt_j = range(indices[j,0],indices[j,1])

    #             if X is not None:
    #                 Xj = np.copy(X[tt_j,:])

    #             d = np.copy(Y[tt_j,:])
    #             if self.hyperparameters["model_mean"] == 'shared':
    #                 d -= np.expand_dims(self.mean[0]['Mu'],axis=0)
    #             if self.hyperparameters["model_beta"] == 'shared':
    #                 d -= (Xj @ self.beta[0]['Mu'])
    #             for k in range(K):
    #                 if self.hyperparameters["model_mean"] == 'state': 
    #                     d -= np.expand_dims(self.mean[k]['Mu'],axis=0) * np.expand_dims(Gamma[:,k],axis=1)
    #                 if self.hyperparameters["model_beta"] == 'state':
    #                     d -= (Xj @ self.beta[k]['Mu']) * np.expand_dims(Gamma[:,k],axis=1)
    #             d = np.sum(d**2,axis=0)

    #             d0 = np.copy(Y[tt_j,:])
    #             if self.hyperparameters["model_mean"] != 'no':
    #                 d0 -= np.expand_dims(m,axis=0)
    #             d0 = np.sum(d0**2,axis=0)

    #             r2[j,:] = 1 - (d / d0)


def custom_colormap():
    """
    Generate a custom colormap consisting of segments from red to blue.

    Returns:
    --------
    A custom colormap with defined color segments.
    """
    # Retrieve existing colormaps
    coolwarm_cmap = plt.get_cmap('coolwarm').reversed()
    coolwarm_cmap2 = plt.get_cmap('autumn')
    copper_cmap = plt.get_cmap('copper').reversed()
    # Define the colors for the colormap
    copper_color1 = to_rgba_array(copper_cmap(1))[0][:3]
    # Define the colors for the colormap
    red = (1,0,0)
    red2 = (66/255, 13/255, 9/255)
    orange =(1, 0.5, 0)
    # red_color1 = to_rgba_array(coolwarm_cmap(0))[0][:3]
    warm_color2 = to_rgba_array(coolwarm_cmap2(0.8))[0][:3]
    blue_color1 = to_rgba_array(coolwarm_cmap(0.6))[0][:3]
    blue_color2 = to_rgba_array(coolwarm_cmap(1.0))[0][:3] # Extract the blue color from coolwarm

    # Define the color map with three segments: red to white, white, and white to blue
    cmap_segments = [
        (0.0, red2),
        #(0.002, orange),
        (0.005, red),   # Intermediate color
        (0.02, orange),   # Intermediate color
        #(0.045, warm_color1),
        (0.040, warm_color2),  # Intermediate color
        (0.05, copper_color1),
        (0.09,blue_color1),
        (1, blue_color2)
    ]

    # Create the custom colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', cmap_segments)

    return custom_cmap

def red_colormap():
    """
    Generate a custom colormap consisting of red and warm colors.

    Returns:
    --------
    A custom colormap with red and warm color segments.
    """
    # Get the reversed 'coolwarm' colormap
    coolwarm_cmap = plt.get_cmap('coolwarm').reversed()
    # Get the 'autumn' colormap
    autumn_cmap = plt.get_cmap('autumn')

    # Define the colors for the colormap
    red0 = (float(120/255), 0, 0)
    red = (1, 0, 0)
    red2 = (66/255, 13/255, 9/255)
    orange = (1, 0.5, 0)
    red_color1 = to_rgba_array(coolwarm_cmap(0))[0][:3]
    warm_color1 = to_rgba_array(autumn_cmap(0.4))[0][:3]
    warm_color2 = to_rgba_array(autumn_cmap(0.7))[0][:3]
    # Define the color map with three segments: red to white, white, and white to blue
    cmap_segments = [
        (0.0, red2),
        (0.3, red0),
        (0.5, red),
        (0.7, warm_color1),  # Intermediate color
        (1, warm_color2),    # Intermediate color
    ]
    # Create the custom colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', cmap_segments)
    return custom_cmap

def blue_colormap():
    """
    Generate a custom blue colormap.

    Returns:
    --------
    A custom colormap with shades of blue.
    """
    coolwarm_cmap = plt.get_cmap('coolwarm').reversed()
    copper_cmap = plt.get_cmap('copper').reversed()
    # cool_cmap = plt.get_cmap('cool')
    # Define the colors for the colormap
    # white = (1, 1, 1)  # White color
    copper_color1 = to_rgba_array(copper_cmap(1))[0][:3]
    # cool_color1 = to_rgba_array(cool_cmap(0.3))[0][:3]
    # blue_color1 = to_rgba_array(coolwarm_cmap(0.5))[0][:3]
    blue_color2 = to_rgba_array(coolwarm_cmap(0.7))[0][:3]
    blue_color3 = to_rgba_array(coolwarm_cmap(1.0))[0][:3] # Extract the blue color from coolwarm

    # Define the color map with three segments: red to white, white, and white to blue
    cmap_segments = [
        (0, copper_color1),
        #(0.15, cool_color1),
        (0.2,blue_color2),
        #(0.7, cool_color1),
        (1, blue_color3)
    ]
    # Create the custom colormap
    blue_cmap = LinearSegmentedColormap.from_list('custom_colormap', cmap_segments)

    return blue_cmap

def create_cmap_alpha(cmap_list,color_array, alpha):
    """
    Modify the colors in a colormap based on an alpha threshold.

    Parameters:
    -----------
    cmap_list (numpy.ndarray)
        List of colors representing the original colormap.
    color_array (numpy.ndarray)
        Array of color values corresponding to each colormap entry.
    alpha (float)
        Alpha threshold for modifying colors.

    Returns:
    --------
    Modified list of colors representing the colormap with adjusted alpha values.
    """
    cmap_list_alpha =cmap_list.copy()

    _,idx_alpha =np.where(color_array <= alpha)
    coolwarm_cmap = plt.get_cmap('coolwarm').reversed()
    #coolwarm_cmap2 = plt.get_cmap('autumn')
    red = (1,0,0)
    orange =(1, 0.5, 0)
    red_color1 = to_rgba_array(coolwarm_cmap(0))[0][:3]

        
    list_red =  [red,red_color1,orange]
    idx_interval =int(idx_alpha[-1]/(len(list_red)-1))

    # Recolor the first to -idx_interval
    cmap_list_alpha[:idx_alpha[-1],:3]=list_red[0]
    for i in range(len(list_red)-1):
        cmap_list_alpha[idx_interval*(i+1):idx_alpha[-1]+1,:3]=list_red[i+1]
        
    return cmap_list_alpha

def interpolate_colormap(cmap_list):     
    """
    Create a new colormap with the modified color_array.

    Parameters:
    --------------
    cmap_list (numpy.ndarray): 
        Original color array for the colormap.

    Returns:
    ----------  
    modified_cmap (numpy.ndarray): 
        Modified colormap array.
    """
    # Create a new colormap with the modified color_array
    modified_cmap  = np.ones_like(cmap_list)

    for channel_idx in range(3):
        # Extract the channel values from the colormap
        channel_values = cmap_list[:, channel_idx]

        # Get unique values, their indices, and counts
        unique_values, unique_indices, counts = np.unique(channel_values, return_index=True, return_counts=True)

        # Create a copy unique_indices that is will get reduced for every interation
        remaining_indices = unique_indices.copy()
        remaining_counts = counts.copy()
        # Create a list to store the interpolated values
        new_map_list = []

        for _ in range(len(unique_values)-1):
            # Find the minimum value
            min_value = np.min(remaining_indices)
            # Locate the index
            min_idx =np.where(unique_indices==min_value)
            # Remove the minimum value from the array
            remaining_counts = remaining_counts[remaining_indices != min_value]
            remaining_indices = remaining_indices[remaining_indices != min_value]
            
            # Find the location of the next minimum value from remaining_indices
            next_min_value_idx =np.where(unique_indices==np.min(remaining_indices))
            # Calculate interpolation space difference
            space_diff = (unique_values[next_min_value_idx]-unique_values[min_idx])/int(counts[min_idx])
            # Append interpolated values to the list
            new_map_list.append(np.linspace(unique_values[min_idx], unique_values[next_min_value_idx]-space_diff, int(counts[min_idx])))
        last_val =np.where(unique_indices==np.min(remaining_indices))
        for _ in range(int(remaining_counts)):
            # Append the last value to the new_map_list
            new_map_list.append([unique_values[last_val]])
        con_values= np.squeeze(np.concatenate(new_map_list))
        # Insert values into the new color map
        modified_cmap [:,channel_idx]=con_values
    return modified_cmap

def plot_p_value_matrix(pval_in, alpha = 0.05, normalize_vals=True, figsize=(9, 5), 
                        title_text="Heatmap (p-values)", fontsize_labels=12, fontsize_title=14, annot=False, 
                        cmap_type='default', cmap_reverse=True, xlabel="", ylabel="", 
                        xticklabels=None, yticklabels = None,x_tick_min=None, x_tick_max=None, num_x_ticks=None, num_y_ticks=None,  tick_positions = [0.001, 0.01, 0.05, 0.1, 0.3, 1], 
                        none_diagonal = False, num_colors = 256, xlabel_rotation=0, save_path=None, return_fig= False):
    """
    Plot a heatmap of p-values.

    Parameters:
    -----------
    pval (numpy.ndarray)
        The p-values data to be plotted.
    normalize_vals : (bool, optional), default=False:
        If True, the data range will be normalized from 0 to 1.
    figsize tuple, optional, default=(12,7):
        Figure size in inches (width, height).
    steps (int, optional), default=11:
        Number of steps for x and y-axis ticks.
    title_text (str, optional), default= "Heatmap (p-values)"
        Title text for the heatmap.
    fontsize_labels (int, optional), default=12:
        Font size for the x and y-axis labels.
    fontsize_title (int, optional), default=14
        fontsize of title
    annot (bool, optional), default=False: 
        If True, annotate each cell with the numeric value.
    cmap (str, optional), default= "default":
        Colormap to use. Default is a custom colormap based on 'coolwarm'.
    xlabel (str, optional), default=""
        X-axis label. If not provided, default labels based on the method will be used.
    ylabel (str, optional), default=""
        Y-axis label. If not provided, default labels based on the method will be used.
    xticklabels (List[str], optional), default=None:
        If not provided, labels will be numbers equal to shape of pval.shape[1].
        Else you can define your own labels, e.g., xticklabels=['sex', 'age'].
    x_tick_min (float, optional), default=None:
        Minimum value for the x-tick labels.
    x_tick_max (float, optional), default=None:
        Maximum value for the x-tick labels.
    num_x_ticks (int, optional), default=5:
        Number of x-ticks.
    tick_positions (list, optional), default=[0, 0.001, 0.01, 0.05, 0.1, 0.3, 1]
        Positions of ticks on the colorbar.
    none_diagonal (bool, optional), default=False:
        If you want to turn the diagonal into NaN numbers.
    num_colors (numpy.ndarray), default=259:
        Define the number of different shades of color.
    xlabel_rotation (numpy-mdarray), default=0
        The degree of rotation for the labels in the x-axis
    save_path (str, optional), default=None
        If a string is provided, it saves the figure to that specified path
    """
    if pval_in.ndim>2:
        pval = np.squeeze(pval_in)
        if pval.ndim>2:
            raise ValueError(f"The p-value is {pval.ndim} dimensional\n"
                    "Adjust your p-values so it becomes 2-dimensional")

    else:
        pval = pval_in.copy()
    if pval.ndim==0:
        pval = np.reshape(pval, (1, 1))
    if xlabel_rotation==45:
        ha ="right"
    else:
        ha = "center" 
    if pval.ndim==2:   
        num_x_ticks = num_x_ticks if num_x_ticks is not None else pval.shape[1] if pval.shape[1]<20 else 5
        num_y_ticks = num_y_ticks if num_y_ticks is not None else pval.shape[0] if pval.shape[0]<20 else 5
    else:
        num_x_ticks = num_x_ticks if num_x_ticks is not None else pval.shape[0] if pval.shape[0]<20 else 5
        #num_y_ticks = num_x_ticks if num_x_ticks is not None else pval.shape[0] if pval.shape[0]<20 else 5

    # Ensure p-values are within the log range
    pval_min = -3
    pval[pval != 0] = np.clip(pval[pval != 0], 10**pval_min, 1)

    # Convert to log scale
    color_array = np.logspace(pval_min, 0, num_colors).reshape(1, -1)
    
    fig, axes = plt.subplots(figsize=figsize)
    if len(pval.shape)==1:
        pval =np.expand_dims(pval,axis=0)
    if cmap_type=='default':

        if alpha == None and normalize_vals==False:
            cmap = cm.coolwarm.reversed()
        elif alpha == None and normalize_vals==True:
            # Create custom colormap
            coolwarm_cmap = custom_colormap()
            # Create a new colormap with the modified color_array
            cmap_list = coolwarm_cmap(color_array)[0]
            modified_cmap=interpolate_colormap(cmap_list)
            # Create a LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list('custom_colormap', modified_cmap)
        else:

            # Make a jump in color after alpha
            # Get blue colormap
            cmap_blue = blue_colormap()
            # Create a new colormap with 
            cmap_list = cmap_blue(color_array)[0]
            red_cmap = red_colormap()
            blue_cmap = blue_colormap()
            # Specify the number of elements you want (e.g., 50)
            num_elements_red = np.sum(color_array <= alpha)
            num_elements_blue = np.sum(color_array > alpha)

            # Generate equally spaced values between 0 and 1
            colormap_val_red = np.linspace(0, 1, num_elements_red)
            colormap_val_blue = np.linspace(0, 1, num_elements_blue)

            # Apply the colormap to the generated values
            cmap_red = red_cmap(colormap_val_red)
            cmap_blue = blue_cmap(colormap_val_blue)
            # overwrite the values below alpha
            cmap_list[:num_elements_red,:]=cmap_red
            cmap_list[num_elements_red:,:]=cmap_blue
            cmap = LinearSegmentedColormap.from_list('custom_colormap', cmap_list)

    else:
        # Get the colormap dynamically based on the input string
        cmap = getattr(cm, cmap_type, None)
        if cmap_reverse:
            cmap =cmap.reversed()

        # Set the value of 0 to white in the colormap
    if none_diagonal:
        # Create a copy of the pval matrix
        pval_with_nan_diagonal = np.copy(pval)

        # Set the diagonal elements to NaN in the copied matrix
        np.fill_diagonal(pval_with_nan_diagonal, np.nan)
        pval = pval_with_nan_diagonal.copy()

    if normalize_vals:
        norm = LogNorm(vmin=10**pval_min, vmax=1)

        heatmap = sb.heatmap(pval, ax=axes, cmap=cmap, annot=annot, fmt=".3f", cbar=False, norm=norm)
    else:
        heatmap = sb.heatmap(pval, ax=axes, cmap=cmap, annot=annot, fmt=".3f", cbar=False)

    # Add labels and title
    axes.set_xlabel(xlabel, fontsize=fontsize_labels)
    axes.set_ylabel(ylabel, fontsize=fontsize_labels)
    axes.set_title(title_text, fontsize=fontsize_title)
    # Number of x-tick steps
    steps=len(pval)
    
    # define x_ticks
    x_tick_positions = np.linspace(0, pval.shape[1]-1, num_x_ticks).astype(int)

    # Generate x-tick labels based on user input or default to time points
    if x_tick_min is not None and x_tick_max is not None:
        x_tick_labels = np.linspace(x_tick_min, x_tick_max, num_x_ticks).round(2)
        if np.all(x_tick_labels == x_tick_labels.astype(int)):
            x_tick_labels = x_tick_labels.astype(int)
    elif x_tick_min is not None:
        x_tick_labels = np.linspace(x_tick_min, pval.shape[1], num_x_ticks).round(2)
        if np.all(x_tick_labels == x_tick_labels.astype(int)):
            x_tick_labels = x_tick_labels.astype(int)
    elif x_tick_max is not None:
        x_tick_labels = np.linspace(0, x_tick_max, num_x_ticks).round(2)
        if np.all(x_tick_labels == x_tick_labels.astype(int)):
            x_tick_labels = x_tick_labels.astype(int) 
    else:
        x_tick_labels = x_tick_positions

    # Set the x-axis ticks and labels
    if xticklabels is not None:
        if isinstance(xticklabels, str):
            # Generate labels like "Hello 1", "Hello 2", ...
            xticklabels = [f"{xticklabels} {i + 1}" for i in range(len(x_tick_labels))]
        elif not isinstance(xticklabels, list) or len(xticklabels) != len(x_tick_labels):
            warnings.warn(f"xticklabels must be a list matching x_tick_labels, or a string. Using default numeric labels instead.")
            xticklabels = [f"Feature {i + 1}" for i in range(len(x_tick_labels))]

        axes.set_xticks(x_tick_positions + 0.5)
        axes.set_xticklabels(xticklabels, rotation=xlabel_rotation, fontsize=10, ha=ha)

    elif pval.shape[1] > 1:
        axes.set_xticks(x_tick_positions + 0.5)
        axes.set_xticklabels(x_tick_labels+1, rotation=0, fontsize=10, ha=ha)

    else:
        axes.set_xticklabels([])

    # Define y_ticks
    y_tick_positions = np.linspace(0, pval.shape[0]-1, num_y_ticks).astype(int)
    if pval.shape[0]>1:
        # Set y-axis tick labels
        if yticklabels is not None:
            if isinstance(yticklabels, str):
                # Generate labels like "Label 1", "Label 2", ...
                yticklabels = [f"{yticklabels} {i + 1}" for i in range(len(y_tick_positions))]
            elif not isinstance(yticklabels, list) or len(yticklabels) != len(y_tick_positions):
                warnings.warn(f"yticklabels must be a list matching y_tick_positions, or a string. Using default numeric labels instead.")
                yticklabels = [f"{i + 1}" for i in range(len(y_tick_positions))]
                

            axes.set_yticks(y_tick_positions + 0.5)
            axes.set_yticklabels(yticklabels, fontsize=10, rotation=0)
        else:
            # Fallback: use index numbers
            axes.set_yticks(y_tick_positions + 0.5)
            axes.set_yticklabels(y_tick_positions+1, fontsize=10, rotation=0)
    else:
        axes.set_yticklabels([])
        
    # Create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    
    if normalize_vals:   
        # Define tick positions and labels
        tick_positions = np.array(tick_positions)
        # Add colorbar
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        colorbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax, ticks=tick_positions, format="%.3g"
        )
    
    else:
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="3.5%", pad=0.05)
        # Create a custom colorbar
        colorbar = plt.colorbar(heatmap.get_children()[0], cax=cax)
        # Set the ticks to range from the bottom to the top of the colorbar
        # Get the minimum and maximum values from your data
        min_value = np.nanmin(pval)
        max_value = np.nanmax(pval)

        # Set ticks with at least 5 values evenly spaced between min and max
        colorbar.set_ticks(np.linspace(min_value, max_value, 5).round(2))
        #colorbar.set_ticks([0, 0.25, 0.5, 1])  # Adjust ticks as needed
        
    plt.tight_layout()
    # Save the figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    if return_fig:
        return fig
    else:
        plt.show()
    
  
def plot_permutation_distribution(base_statistics_perms, title_text="Permutation Distribution",xlabel="Test Statistic Values",ylabel="Density", save_path=None, return_fig=False):
    """
    Plot the histogram of the permutation with the observed statistic marked.

    Parameters:
    -----------
    base_statistics_perms (numpy.ndarray)
        An array containing the permutation values.
    title_text (str, optional), default="Permutation Distribution":
        Title text of the plot.
    xlabel (str, optional), default="Test Statistic Values"
        Text of the xlabel.
    ylabel (str, optional), default="Density"
        Text of the ylabel.
    save_path (str, optional), default=None
        If a string is provided, it saves the figure to that specified path
    """
    fig =plt.figure()
    sb.histplot(base_statistics_perms, kde=True)
    plt.axvline(x=base_statistics_perms[0], color='red', linestyle='--', label='Observed Statistic')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title_text, fontsize=14)
        
    plt.legend()

    # Save the figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight') 
    if return_fig:
        return fig
    else:
        plt.show()

def plot_scatter_with_labels(p_values, alpha=0.05, title_text="", xlabel=None, ylabel=None, xlim_start=0.9, ylim_start=0, save_path=None, return_fig=False):
    """
    Create a scatter plot to visualize p-values with labels indicating significant points.

    Parameters:
    -----------
    p_values (numpy.ndarray)
        An array of p-values. Can be a 1D array or a 2D array with shape (1, 5).
    alpha (float, optional), default=0.05:
        Threshold for significance.
    title_text (str, optional), default="":
        The title text for the plot.
    xlabel (str, optional), default=None:
        The label for the x-axis.
    ylabel (str, optional), default=None:
        The label for the y-axis.
    xlim_start (float, optional), default=-5
        Start position of x-axis limits.
    ylim_start (float, optional), default=-0.1
        Start position of y-axis limits.
    save_path (str, optional), default=None
        If a string is provided, it saves the figure to that specified path

    Notes:
    ------
    Points with p-values less than alpha are considered significant and marked with red text.
    """

    # If p_values is a 2D array with shape (1, 5), flatten it to 1D
    if len(p_values.shape) == 2 and p_values.shape[0] == 1 and p_values.shape[1] == 5:
        p_values = p_values.flatten()

    # Create a binary mask based on condition (values below alpha)
    mask = p_values < alpha

    # Create a hue p_values based on the mask (True/False values)
    hue = mask.astype(int)

    # Set the color palette and marker style
    markers = ["o", "s"]

    # Create a scatter plot with hue and error bars
    fig, ax = plt.subplots(figsize=(8, 6))
    sb.scatterplot(x=np.arange(0, len(p_values)) + 1, y=-np.log(p_values), hue=hue, style=hue,
                    markers=markers, s=40, edgecolor='k', linewidth=1, ax=ax)

    # Add labels and title to the plot
    if not title_text:
        ax.set_title(f'Scatter Plot of P-values, alpha={alpha}', fontsize=14)
    else:
        ax.set_title(title_text, fontsize=14)

    if xlabel is None:
        ax.set_xlabel('Index', fontsize=12)
    else:
        ax.set_xlabel(xlabel, fontsize=12)

    if ylabel is None:
        ax.set_ylabel('-log(p-values)', fontsize=12)
    else:
        ax.set_ylabel(ylabel, fontsize=12)

    # Add text labels for indices where the mask is True
    for i, m in enumerate(mask):
        if m:
            ax.text(i + 1, -np.log(p_values[i]), str(i+1), ha='center', va='bottom', color='red', fontsize=10)

    # Adjust legend position and font size
    ax.legend(title="Significance", loc="upper right", fontsize=10, bbox_to_anchor=(1.25, 1))

    # Set axis limits to focus on the relevant data range
    ax.set_xlim(xlim_start, len(p_values) + 1)
    ax.set_ylim(ylim_start, np.max(-np.log(p_values)) * 1.2)

    # # Customize plot background and grid style
    # sb.set_style("white")
    # ax.grid(color='lightgray', linestyle='--')

    # Show the plot
    plt.tight_layout()
    # Save the figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight') 

    if return_fig:
        return fig
    else:
        plt.show()

def plot_vpath(viterbi_path, signal=None, idx_data=None, figsize=(7, 4), fontsize_labels=13, fontsize_title=16, 
               yticks=None, time_conversion_rate=None, xlabel="Timepoints", ylabel="", title="Viterbi Path", cmap=None,
               signal_label="Signal", show_legend=True, vertical_linewidth=1.5, save_path=None, return_fig=False):
    """
    Plot Viterbi path with optional signal overlay.

    Parameters:
    -----------
    viterbi_path 
        The Viterbi path data matrix.
    signal (numpy.ndarray), optional 
        Signal data to overlay on the plot. Default is None.
    idx_data (numpy.ndarray), optional  
        Array representing time intervals. Default is None.
    figsize (tuple), optional 
        Figure size. Default is (7, 4).
    fontsize_labels (int), optional
        Font size for axis labels. Default is 13.
    fontsize_title (int), optional 
        Font size for plot title. Default is 16.
    yticks (bool), optional 
        Whether to show y-axis ticks. Default is None.
    time_conversion_rate (float), optional
        Conversion rate from time steps to seconds. Default is None.
    xlabel (str), optional
        Label for the x-axis. Default is "Timepoints".
    ylabel (str), optional
        Label for the y-axis. Default is "".
    title (str), optional 
        Title for the plot. Default is "Viterbi Path".
    cmap (str, optional): 
        Name of a colormap to use for state line colors (default is 'Set3').
    signal_label (str, optional
        Label for the signal plot. Default is "Signal".
    show_legend (bool), optional
        Whether to show the legend. Default is True.
    vertical_linewidth (float), optional
        Line width for vertical gray lines. Default is 1.5.
    save_path (str, optional), default=None
        If a string is provided, it saves the figure to that specified path
    """
    num_states = viterbi_path.shape[1]
    colors = sb.color_palette("Set3", n_colors=num_states)

    # Assign distinct colors for each state
    if cmap is not None:
        # Assign distinct colors for each component
        valid_cmaps = plt.colormaps()
        if isinstance(cmap, str) and cmap in valid_cmaps:
            if num_states <= 10:
                cmap = plt.get_cmap(cmap)
                colors = [cmap(i) for i in range(num_states)]
            else: 
                colors = get_distinct_colors(num_states, cmap)
        else:
            warnings.warn(f"Invalid colormap '{cmap}'. Falling back to 'Set3'. "
                          f"Use one of: {', '.join(valid_cmaps[:5])}... etc.")
            cmap = plt.get_cmap('Set3')
            colors = [cmap(i) for i in range(num_states)]
    elif num_states <= 10:
        cmap = plt.get_cmap('Set3')
        colors = [cmap(i) for i in range(num_states)]
    else:
        colors = get_distinct_colors(num_states, cmap)

    fig, axes = plt.subplots(figsize=figsize)

    
    # Plot Viterbi path
    if time_conversion_rate is not None:
        time_seconds = np.arange(viterbi_path.shape[0]) / time_conversion_rate
        axes.stackplot(time_seconds, viterbi_path.T, colors=colors, labels=[f'State {i + 1}' for i in range(num_states)])
        if xlabel == "Timepoints":
            xlabel = "Time (seconds)"
        axes.set_xlabel(xlabel, fontsize=fontsize_labels)
    else:
        axes.stackplot(np.arange(viterbi_path.shape[0]), viterbi_path.T, colors=colors, labels=[f'State {i + 1}' for i in range(num_states)])
        axes.set_xlabel(xlabel, fontsize=fontsize_labels)

    axes.set_ylabel(ylabel, fontsize=fontsize_labels)
    axes.set_title(title, fontsize=fontsize_title)

    # Plot signal overlay
    if signal is not None:
        # Normalize the sig_data to the range [0, 1]
        min_value = np.min(signal)
        max_value = np.max(signal)
        normalized_sig_data = ((signal - min_value) / (max_value - min_value))
        if time_conversion_rate is not None:
            time_seconds = np.arange(len(normalized_sig_data)) / time_conversion_rate
            axes.plot(time_seconds, normalized_sig_data, color='black', label=signal_label)
            axes.set_xlabel(xlabel, fontsize=fontsize_labels)
        else:
            axes.plot(normalized_sig_data, color='black', label=signal_label)

    # Draw vertical gray lines for T_t intervals
    if idx_data is not None:
        for idx in idx_data[:-1, 1]:
            axes.axvline(x=idx, color='gray', linestyle='--', linewidth=vertical_linewidth)

    # Show legend
    if show_legend:
        axes.legend(title='States', loc='upper left', bbox_to_anchor=(1, 1))

    if yticks and signal is not None:
        scaled_values = [int(val * len(np.unique(normalized_sig_data))) for val in np.unique(normalized_sig_data)]
        # Set y-ticks with formatted integers
        axes.set_yticks(np.unique(normalized_sig_data), scaled_values)
    else:
        # Remove x-axis tick labels
        axes.set_yticks([])
        
    # Remove the frame around the plot
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    # Adjust tick label font size
    axes.tick_params(axis='both', labelsize=fontsize_labels)

    plt.tight_layout() 
    # Save the figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight') 
    if return_fig:
        return fig
    else:
        plt.show()
    
def plot_average_probability(Gamma_data, title='Average probability for each state', fontsize=16, figsize=(7, 5), 
                             vertical_lines=None, line_colors=None, highlight_boxes=False, save_path=None, return_fig=False):

    """
    Plots the average probability for each state over time.

    Parameters:
    -----------
    Gamma_data (numpy.ndarray)
        Can be a 2D or 3D array representing gamma values.
        Shape: (num_timepoints, num_states) or (num_timepoints, num_trials, num_states)
    title (str, optional), default='Average probability for each state':
        Title for the plot.
    fontsize (int, optional), default=16:
        Font size for labels and title.
    figsize (tuple, optional), default=(8,6):
        Figure size (width, height) in inches).
    vertical_lines (list of tuples, optional), default=None:
        List of pairs specifying indices for vertical lines.
    line_colors (list of str or bool, optional), default=None:
        List of colors for each pair of vertical lines. If True, generates random colors (unless a list is provided).
    highlight_boxes (bool, optional), default=False:
        Whether to include highlighted boxes for each pair of vertical lines.
    save_path (str, optional), default=None
        If a string is provided, it saves the figure to that specified path
    """

    # Initialize an array for average gamma values
    Gamma_avg = np.zeros((Gamma_data.shape[0], Gamma_data.shape[-1]))

    if Gamma_data.ndim==3:
        # Calculate and store average gamma values
        for i in range(Gamma_data.shape[0]):
            filtered_values = Gamma_data[i, :, :]
            Gamma_avg[i, :] = np.mean(filtered_values, axis=0).round(3)
    else:
        Gamma_avg = Gamma_data.copy()

    # Set figure size
    fig, axes = plt.subplots(1, figsize=figsize)

    # Plot each line with a label
    for state in range(Gamma_data.shape[-1]):
        plt.plot(Gamma_avg[:, state], label=f'State {state + 1}')
        
    # Add vertical lines, line colors, and highlight boxes
    if vertical_lines:
        for idx, pair in enumerate(vertical_lines):
            color = line_colors[idx] if line_colors and len(line_colors) > idx else 'gray'
            axes.axvline(x=pair[0], color=color, linestyle='--', linewidth=1)
            axes.axvline(x=pair[1], color=color, linestyle='--', linewidth=1)

            if highlight_boxes:
                rect = plt.Rectangle((pair[0], axes.get_ylim()[0]), pair[1] - pair[0], axes.get_ylim()[1] - axes.get_ylim()[0], linewidth=0, edgecolor='none', facecolor=color, alpha=0.2)
                axes.add_patch(rect)

    # Add labels and legend
    plt.xlabel('Timepoints', fontsize=fontsize)
    plt.ylabel('Average probability', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)

    # Add legend for the highlighted boxes
    if highlight_boxes:
        legend_rect = plt.Rectangle((0, 0), 1, 1, linewidth=0, edgecolor='none', facecolor='gray', alpha=0.2, label='Interval with significant difference')
        plt.legend(handles=[legend_rect], loc='upper right')

    # Place legend for the lines to the right of the figure
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Save the figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight') 
    if return_fig:
        return fig
    else:
        plt.show()


def plot_FO(FO, figsize=(8, 4), fontsize_ticks=12, fontsize_labels=14, fontsize_title=16, width=0.8, xlabel='Subject',
            ylabel='Fractional occupancy', title='State Fractional Occupancies', cmap=None,
            show_legend=True, num_x_ticks=11, num_y_ticks=5, pad_y_spine=None, save_path=None, return_fig=False):
    """
    Plot fractional occupancies for different states.

    Parameters:
    -----------
    FO (numpy.ndarray):
        Fractional occupancy data matrix.
    figsize (tuple, optional), default=(8,4):
        Figure size.
    fontsize_ticks (int, optional), default=12:
        Font size for tick labels.
    fontsize_labels (int, optional), default=14:
        Font size for axes labels.
    fontsize_title (int, optional), default=16:
        Font size for plot title.
    width (float, optional), default=0.5:
        Width of the bars.
    xlabel (str, optional), default='Subject':
        Label for the x-axis.
    ylabel (str, optional), default='Fractional occupancy':
        Label for the y-axis.
    title (str, optional), default='State Fractional Occupancies':
        Title for the plot.
    cmap (str, optional): 
        Name of a colormap to use for state line colors (default is 'Set3').
    show_legend (bool, optional), default=True:
        Whether to show the legend.
    num_x_ticks (int, optional), default=11:
        Number of ticks for the x-axis.
    num_y_ticks (int, optional), default=5:
        Number of ticks for the y-axis.
    pad_y_spine (float, optional), default=None:
        Shifting the positin of the spine for the y-axis.
    save_path (str, optional), default=None:
        If a string is provided, it saves the figure to that specified path.
    """
    fig, axes = plt.subplots(figsize=figsize)
    bottom = np.zeros(FO.shape[0])
    sessions = np.arange(1, FO.shape[0] + 1)
    num_states = FO.shape[1]
    
    # Assign distinct colors for each component
    if cmap is not None:
        # Assign distinct colors for each component
        valid_cmaps = plt.colormaps()
        if isinstance(cmap, str) and cmap in valid_cmaps:
            if num_states <= 10:
                cmap = plt.get_cmap(cmap)
                colors = [cmap(i) for i in range(num_states)]
            else: 
                colors = get_distinct_colors(num_states, cmap)
        else:
            warnings.warn(f"Invalid colormap '{cmap}'. Falling back to 'Set3'. "
                          f"Use one of: {', '.join(valid_cmaps[:5])}... etc.")
            cmap = plt.get_cmap('Set3')
            colors = [cmap(i) for i in range(num_states)]
    elif num_states <= 10:
        cmap = plt.get_cmap('Set3')
        colors = [cmap(i) for i in range(num_states)]
    else:
        colors = get_distinct_colors(num_states, cmap)
        
    for k in range(num_states):
        axes.bar(sessions, FO[:, k], bottom=bottom, color=colors[k], width=width)
        bottom += FO[:, k]
    
    axes.set_xlabel(xlabel, fontsize=fontsize_labels)
    axes.set_ylabel(ylabel, fontsize=fontsize_labels)
    axes.set_title(title, fontsize=fontsize_title)
    
    ticks = np.linspace(1, FO.shape[0], FO.shape[0]).astype(int)
    # If there are more than 11 states then reduce the number of ticks
    if len(ticks) > 11:
        n_ticks = num_x_ticks
    else:
        n_ticks = len(ticks)
    axes.set_xticks(np.linspace(1, FO.shape[0], n_ticks).astype(int))
    axes.set_yticks(np.linspace(0, 1, num_y_ticks))

    # Adjust tick label font size
    axes.tick_params(axis='x', labelsize=fontsize_ticks)
    axes.tick_params(axis='y', labelsize=fontsize_ticks)

    # Remove the frame around the plot
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    if pad_y_spine is None:
        pad_y_spine = -figsize[0]*2.2
    axes.spines['left'].set_position(('outward', pad_y_spine))  # Adjust the outward position of the left spine

    # Add a legend if needed
    if show_legend:
        axes.legend([f'State {i + 1}' for i in range(FO.shape[1])], fontsize=fontsize_ticks, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    # Save the figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if return_fig:
        return fig
    else:
        plt.show()

def plot_switching_rates(SR, figsize=(8, 4), fontsize_ticks=12,fontsize_labels=14,fontsize_title=16,width=0.18,group_gap=None,
                        xlabel='Subject', ylabel='Switching Rate',title='State Switching Rates',cmap=None,
                        show_legend=True,num_x_ticks=11,num_y_ticks=5,pad_y_spine=None,save_path=None, return_fig=False):
    """
    Plot grouped bar charts of switching rates for different states across sessions.

    Parameters
    ----------
    SR : np.ndarray
        Array of shape (n_sessions, n_states) containing switching rates.
    figsize : tuple of float, default=(8, 4)
        Size of the figure in inches (width, height).
    fontsize_ticks : int, default=12
        Font size for axis tick labels.
    fontsize_labels : int, default=14
        Font size for x-axis and y-axis labels.
    fontsize_title : int, default=16
        Font size for the title.
    width : float, default=0.18
        Width of each individual bar representing a state.
    group_gap : float or None, default=None
        Horizontal spacing between each group (session). If None, uses width * 0.5.
    xlabel : str, default='Subject'
        Label for the x-axis.
    ylabel : str, default='Switching Rate'
        Label for the y-axis.
    title : str, default='State Switching Rates'
        Title of the plot.
    cmap : str or None, default=None
        Name of a matplotlib colormap. If None, a default discrete colormap is used.
    show_legend : bool, default=True
        Whether to display the legend for states.
    num_x_ticks : int, default=11
        Number of ticks to show on the x-axis (session axis).
    num_y_ticks : int, default=5
        Number of ticks to show on the y-axis.
    pad_y_spine : float or None, default=None
        Padding to shift the left spine outward. If None, it's computed from the figure width.
    save_path : str or None, default=None
        If provided, saves the figure to this file path.
    """

    fig, axes = plt.subplots(figsize=figsize, constrained_layout=True)

    num_sessions = SR.shape[0]
    num_states = SR.shape[1]
    total_width = num_states * width

    if group_gap is None:
        group_gap = width * 0.5  # default spacing between groups

    # Calculate x positions with gap between groups
    group_centers = np.arange(num_sessions) * (total_width + group_gap)

    # Assign distinct colors for each component
    if cmap is not None:
        valid_cmaps = plt.colormaps()
        if isinstance(cmap, str) and cmap in valid_cmaps:
            if num_states <= 10:
                cmap = plt.get_cmap(cmap)
                colors = [cmap(i) for i in range(num_states)]
            else:
                colors = get_distinct_colors(num_states, cmap)
        else:
            warnings.warn(
                f"Invalid colormap '{cmap}'. Falling back to 'Set3'. "
                f"Use one of: {', '.join(valid_cmaps[:5])}... etc."
            )
            cmap = plt.get_cmap('Set3')
            colors = [cmap(i) for i in range(num_states)]
    elif num_states <= 10:
        cmap = plt.get_cmap('Set3')
        colors = [cmap(i) for i in range(num_states)]
    else:
        colors = get_distinct_colors(num_states, cmap)

    # Plot bars
    for k in range(num_states):
        offset = width * k
        axes.bar(group_centers + offset, SR[:, k], width, color=colors[k])

    # Labeling
    axes.set_xlabel(xlabel, fontsize=fontsize_labels)
    axes.set_ylabel(ylabel, fontsize=fontsize_labels)
    axes.set_title(title, fontsize=fontsize_title)

    # Set x-ticks centered on groups
    xticks_pos = group_centers + total_width / 2 - width / 2
    session_labels = np.arange(1, num_sessions + 1)
    if num_sessions > num_x_ticks:
        xtick_idx = np.linspace(0, num_sessions - 1, num_x_ticks).astype(int)
        axes.set_xticks(xticks_pos[xtick_idx])
        axes.set_xticklabels(session_labels[xtick_idx])
    else:
        axes.set_xticks(xticks_pos)
        axes.set_xticklabels(session_labels)

    # Y-ticks
    axes.yaxis.set_major_locator(MaxNLocator(nbins=num_y_ticks, prune=None, integer=False)) 
    axes.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
    axes.tick_params(axis='both', labelsize=fontsize_ticks)

    # Remove spines
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    if pad_y_spine is None:
        pad_y_spine = -figsize[0] * 2.2
    axes.spines['left'].set_position(('outward', pad_y_spine))

    if show_legend:
        axes.legend(
            ['State {}'.format(i + 1) for i in range(num_states)],
            fontsize=fontsize_labels,
            loc='upper left',
            bbox_to_anchor=(1, 1)
        )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if return_fig:
        return fig
    else:
        plt.show()


def plot_state_lifetimes(LT, figsize=(8, 4), fontsize_ticks=12, fontsize_labels=14, fontsize_title=16, width=0.18, group_gap = None,
                         xlabel='Subject', ylabel='Lifetime', title='State Lifetimes', cmap= None,
                         show_legend=True, num_x_ticks=11, num_y_ticks=5, pad_y_spine=None, save_path=None, return_fig=False):
    """
    Plot state lifetimes for different states.

    Parameters:
    -----------
    LT (numpy.ndarray): 
        State lifetime (dwell time) data matrix.
    figsize (tuple, optional), default=(8, 4):
        Figure size.
    fontsize_ticks (int, optional), default=12:
        Font size for tick labels.
    fontsize_labels (int, optional), default=14:
        Font size for axeses labels.
    fontsize_title (int, optional), default=16:
        Font size for plot title.
    width (float, optional), default=0.18:
        Width of the bars.
    group_gap : float or None
        Gap between groups of bars (sessions). If None, defaults to width * 0.5.
    xlabel (str, optional), default='Subject':
        Label for the x-axesis.
    ylabel (str, optional), default='Lifetime':
        Label for the y-axesis.
    title (str, optional), default='State Lifetimes':
        Title for the plot.
    cmap (str, optional): 
        Name of a colormap to use for state line colors (default is 'Set3').
    show_legend (bool, optional), default=True:
        Whether to show the legend.
    num_x_ticks (int, optional), default=11:
        Number of ticks for the x-axis.
    num_y_ticks (int, optional), default=5:
        Number of ticks for the y-axis.
    pad_y_spine (float, optional), default=None:
        Shifting the positin of the spine for the y-axis.
    save_path (str, optional), default=None
        If a string is provided, it saves the figure to that specified path
    """

    fig, axes = plt.subplots(figsize=figsize, constrained_layout=True)

    num_sessions = LT.shape[0]
    num_states = LT.shape[1]
    total_width = num_states * width

    if group_gap is None:
        group_gap = width * 0.5

    # Calculate x positions for each group
    group_centers = np.arange(num_sessions) * (total_width + group_gap)

    # Assign distinct colors
    if cmap is not None:
        valid_cmaps = plt.colormaps()
        if isinstance(cmap, str) and cmap in valid_cmaps:
            if num_states <= 10:
                cmap = plt.get_cmap(cmap)
                colors = [cmap(i) for i in range(num_states)]
            else:
                colors = get_distinct_colors(num_states, cmap)
        else:
            warnings.warn(f"Invalid colormap '{cmap}'. Falling back to 'Set3'.")
            cmap = plt.get_cmap('Set3')
            colors = [cmap(i) for i in range(num_states)]
    elif num_states <= 10:
        cmap = plt.get_cmap('Set3')
        colors = [cmap(i) for i in range(num_states)]
    else:
        colors = get_distinct_colors(num_states, cmap)

    # Plot bars
    for k in range(num_states):
        offset = width * k
        axes.bar(group_centers + offset, LT[:, k], width, color=colors[k])

    axes.set_xlabel(xlabel, fontsize=fontsize_labels)
    axes.set_ylabel(ylabel, fontsize=fontsize_labels)
    axes.set_title(title, fontsize=fontsize_title)

    # Center x-ticks under each group
    xticks_pos = group_centers + total_width / 2 - width / 2
    session_labels = np.arange(1, num_sessions + 1)
    if num_sessions > num_x_ticks:
        xtick_idx = np.linspace(0, num_sessions - 1, num_x_ticks).astype(int)
        axes.set_xticks(xticks_pos[xtick_idx])
        axes.set_xticklabels(session_labels[xtick_idx])
    else:
        axes.set_xticks(xticks_pos)
        axes.set_xticklabels(session_labels)

    # Y-ticks
    axes.yaxis.set_major_locator(MaxNLocator(nbins=num_y_ticks, prune=None, integer=False)) 
    axes.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
    axes.tick_params(axis='both', labelsize=fontsize_ticks)

    # Remove spines
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    if pad_y_spine is None:
        pad_y_spine = -figsize[0] * 2.2
    axes.spines['left'].set_position(('outward', pad_y_spine))

    if show_legend:
        axes.legend(
            ['State {}'.format(i + 1) for i in range(num_states)],
            fontsize=fontsize_labels,
            loc='upper left',
            bbox_to_anchor=(1, 1)
        )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    if return_fig:
        return fig
    else:
        plt.show()


def plot_initial_state_probabilities(init_stateP, cmap='coolwarm',
                                     figsize=(2, 4), title_text="Initial State Probabilities",
                                     fontsize_labels=12, fontsize_title=14, tick_size=10,
                                     num_ticks=None, save_path=None, return_fig=False):
    """
    Plot the initial state probabilities of a Hidden Markov Model as a vertical heatmap.

    Parameters
    ----------
    init_probs : np.ndarray
        1D array of shape (n_states,) representing the initial state probabilities.
    cmap : str, default="coolwarm"
        Colormap used for the heatmap.
    figsize : tuple of float, default=(2.5, 4)
        Size of the full figure in inches (width, height).
    title_text : str, default="Initial State Probabilities"
        Title displayed above the plot.
    fontsize_labels : int, default=12
        Font size for axis labels and colorbar label.
    fontsize_title : int, default=14
        Font size for the plot title.
    tick_size : int, default=10
        Font size for tick labels.
    num_ticks : int or None, default=None
        Number of ticks to show on the y-axis and colorbar.
        If None, automatically adjusts based on the number of states.
    save_path : str or None, default=None
        If provided, saves the figure to this file path.
    return_fig : bool, default=False
        If True, returns the figure object instead of displaying it.
    """
    init_stateP = np.atleast_1d(init_stateP)
    if init_stateP.ndim != 1:
        raise ValueError("Initial state probabilities must be 1-dimensional.")
    
    n_states = len(init_stateP)
    if num_ticks is None:
        num_ticks = n_states if n_states <= 20 else 5

    fig, ax = plt.subplots(figsize=figsize)

    # Use seaborn heatmap but disable default colorbar
    heatmap = sb.heatmap(init_stateP.reshape(-1, 1), ax=ax, cmap=cmap,
                          cbar=False, xticklabels=False, yticklabels=False)

    # Axis formatting
    ax.set_title(title_text, fontsize=fontsize_title)
    ax.set_xlabel("")
    ax.set_ylabel("State", fontsize=fontsize_labels)
    ax.tick_params(labelsize=tick_size)

    # Y-ticks
    ax.set_yticks(np.linspace(0.5, n_states - 0.5, num_ticks))
    ax.set_yticklabels(np.linspace(1, n_states, num_ticks, dtype=int))

    # Add colorbar manually
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="40%", pad=0.1)
    cbar = plt.colorbar(ax.collections[0], cax=cax)
    cbar.set_label("Probability", fontsize=fontsize_labels)
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.locator = MaxNLocator(nbins=num_ticks)
    cbar.update_ticks()
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if return_fig:
        return fig
    else:
        plt.show()



def plot_state_means_activations(state_means, cmap_type='coolwarm', cmap_reverse=False,
                                  figsize=(3, 5), title_text="State Mean Activations", xlabel="State", ylabel="Brain region",
                                  fontsize_labels=12, fontsize_title=14, tick_size=10, annot=False, 
                                  xticklabels=None, yticklabels=None, xlabel_rotation=None,
                                  num_x_ticks=None, num_y_ticks=None, save_path=None, return_fig=False):
    """
    Plot a heatmap of state mean activations with optional tick formatting and custom labels.

    Parameters
    ----------
    state_means : np.ndarray
        Array of shape (n_states, n_features) or (n_features, n_states).
        Each column represents the mean activation per state across features (e.g., brain regions).
    cmap_type : str, default='coolwarm'
        Name of the matplotlib colormap to use.
    cmap_reverse : bool, default=False
        If True, reverses the selected colormap.
    figsize : tuple of float, default=(3, 5)
        Size of the figure in inches (width, height).
    title_text : str, default="State Mean Activations"
        Title to display above the heatmap.
    xlabel : str, default="State"
        Label for the x-axis.
    ylabel : str, default="Brain region"
        Label for the y-axis.
    fontsize_labels : int, default=12
        Font size for axis labels and colorbar label.
    fontsize_title : int, default=14
        Font size for the plot title.
    tick_size : int, default=10
        Font size for tick labels.
    annot : bool, default=False
        If True, annotate each cell in the heatmap with its value.
    xticklabels : list, str, or None, default=None
        List of custom x-tick labels, or string prefix to auto-generate labels (e.g., "State").
        If None, numeric labels are used.
    yticklabels : list, str, or None, default=None
        List of custom y-tick labels, or string prefix to auto-generate labels.
        If None, numeric labels are used.
    xlabel_rotation : int or None, default=None
        Rotation angle for x-axis labels. Automatically set to 45 if number of states > 10.
    num_x_ticks : int or None, default=None
        Number of ticks to show on the x-axis. Automatically chosen if None.
    num_y_ticks : int or None, default=None
        Number of ticks to show on the y-axis. Automatically chosen if None.
    save_path : str or None, default=None
        If provided, saves the figure to this path.
    return_fig : bool, default=False
        If True, returns the figure object instead of displaying the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Returns the figure if return_fig=True, otherwise shows the plot directly.
    """
    # Ensure correct orientation
    state_means = np.atleast_2d(state_means)
    if state_means.shape[0] < state_means.shape[1]:
        state_means = state_means.T  # shape: (n_features, n_states)

    n_features, n_states = state_means.shape

    # Set default tick font size and rotation
    if xlabel_rotation is None:
        xlabel_rotation = 45 if n_states > 10 else 0
    ha = "center"

    # Determine number of ticks
    if num_x_ticks is None:
        num_x_ticks = n_states if n_states <= 20 else 5
    if num_y_ticks is None:
        num_y_ticks = n_features if n_features <= 20 else 5

    x_tick_positions = np.linspace(0, n_states - 1, num_x_ticks).astype(int)
    y_tick_positions = np.linspace(0, n_features - 1, num_y_ticks).astype(int)

    # Colormap
    cmap = getattr(plt.cm, cmap_type, plt.cm.coolwarm)
    if cmap_reverse:
        cmap = cmap.reversed()

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    heatmap = sb.heatmap(state_means, ax=ax, cmap=cmap, annot=annot, fmt=".2f",
                          cbar=False, xticklabels=False, yticklabels=False)

    # Labels and title
    ax.set_title(title_text, fontsize=fontsize_title)
    ax.set_xlabel(xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel, fontsize=fontsize_labels)
    ax.tick_params(labelsize=tick_size)

    # X-tick labels
    if xticklabels is not None:
        if isinstance(xticklabels, str):
            xticklabels = [f"{xticklabels} {i + 1}" for i in range(len(x_tick_positions))]
        elif not isinstance(xticklabels, list) or len(xticklabels) != len(x_tick_positions):
            xticklabels = [f"State {i + 1}" for i in range(len(x_tick_positions))]
    else:
        xticklabels = [str(i + 1) for i in x_tick_positions]

    ax.set_xticks(x_tick_positions + 0.5)
    ax.set_xticklabels(xticklabels, rotation=xlabel_rotation, ha=ha, fontsize=tick_size)

    # Y-tick labels
    if yticklabels is not None:
        if isinstance(yticklabels, str):
            yticklabels = [f"{yticklabels} {i + 1}" for i in range(len(y_tick_positions))]
        elif not isinstance(yticklabels, list) or len(yticklabels) != len(y_tick_positions):
            yticklabels = [f"{i + 1}" for i in range(len(y_tick_positions))]
    else:
        yticklabels = [str(i + 1) for i in y_tick_positions]

    ax.set_yticks(y_tick_positions + 0.5)
    ax.set_yticklabels(yticklabels, rotation=0, fontsize=tick_size)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = plt.colorbar(heatmap.get_children()[0], cax=cax)
    cbar.set_label("Activation Level", fontsize=fontsize_labels)
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if return_fig:
        return fig
    else:
        plt.show()


def plot_state_covariances(state_FC, cmap='coolwarm',
                           fontsize_title=12, fontsize_labels=10,
                           tick_size=8, figsize_per_plot=(2.5, 2.8),
                           wspace=None, hspace=None, same_scale=False, num_cols =3,
                           num_ticks=6, save_path=None, return_fig=False):
    """
    Plot state-specific covariance matrices (e.g., functional connectivity) in a grid layout.

    Parameters
    ----------
    state_FC : np.ndarray
        3D array of shape (n_channels, n_channels, n_states).
    cmap : str, default='coolwarm'
        Colormap for plotting.
    fontsize_title : int
        Font size of subplot titles.
    fontsize_labels : int
        Font size of axis labels.
    tick_size : int
        Font size for tick labels.
    figsize_per_plot : tuple
        Size per subplot (width, height).
    wspace, hspace : float or None
        Spacing between subplots.
    same_scale : bool
        If True, uses the same color scale for all subplots.
    num_ticks : int
        Number of tick marks on each axis (shared between x and y).
    save_path : str or None
        If set, saves the plot to this path.
    """
    n_channels, _, K = state_FC.shape

    num_cols_new = min(num_cols, K)
    num_rows = (K + num_cols_new - 1) // num_cols_new
    fig_width = num_cols_new * figsize_per_plot[0]
    fig_height = num_rows * figsize_per_plot[1]

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    axes = np.array(axes).reshape(-1)

    if same_scale:
        vmin = np.min(state_FC)
        vmax = np.max(state_FC)
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    tick_positions = np.linspace(0, n_channels - 1, num_ticks).astype(int)
    tick_labels = [str(i + 1) for i in tick_positions]

    for k in range(K):
        ax = axes[k]
        im = ax.imshow(state_FC[:, :, k], cmap=cmap, interpolation="none", norm=norm)
        ax.set_title(f"State #{k+1}", fontsize=fontsize_title)
        ax.set_xlabel("Brain region", fontsize=fontsize_labels)
        ax.set_ylabel("Brain region", fontsize=fontsize_labels)
        ax.tick_params(labelsize=tick_size)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=tick_size)
        ax.set_yticklabels(tick_labels, fontsize=tick_size)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=tick_size)
        cbar.locator = MaxNLocator(nbins=4)
        cbar.update_ticks()

    # Hide unused axes
    for ax in axes[K:]:
        ax.axis("off")

    if wspace is None:
        wspace = 0.3 if num_cols <= 3 else max(0.60, 0.4 / num_cols)
    if hspace is None:
        hspace = 0.05 if num_rows <= 3 else max(0.15, 0.5 / num_rows)

    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    if return_fig:
        return fig
    else:
        plt.show()



def plot_transition_matrix(TP, with_self_transitions=False, normalize=True, 
                           cmap='coolwarm', figsize=(4, 4), 
                           fontsize_title=14, fontsize_labels=12, tick_size=10,
                           title_text=None, num_ticks=None, save_path=None, return_fig=False):
    """
    Plot a single transition probability matrix (with or without self-transitions).

    Parameters
    ----------
    TP : np.ndarray
        Transition matrix of shape (n_states, n_states).
    with_self_transitions : bool, default=True
        If False, self-transitions will be removed and rows re-normalized.
    normalize : bool, default=True
        Whether to normalize the rows after removing self-transitions.
    cmap : str, default='coolwarm'
        Colormap to use for the heatmap.
    figsize : tuple, default=(4, 4)
        Size of the figure.
    fontsize_title : int, default=14
        Font size of the title.
    fontsize_labels : int, default=12
        Font size for x and y axis labels.
    tick_size : int, default=10
        Font size for axis tick labels.
    title_text : str or None
        Custom title. If None, a default is used based on `with_self_transitions`.
    num_ticks : int or None
        Number of tick labels to show on each axis.
    save_path : str or None
        If set, saves the figure to this path.
    return_fig : bool, default=False
        If True, returns the matplotlib figure object.
    """
    TP = np.atleast_2d(TP)
    n_states = TP.shape[0]

    TP_plot = TP.copy()

    if not with_self_transitions:
        np.fill_diagonal(TP_plot, 0)
        if normalize:
            row_sums = TP_plot.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # prevent divide-by-zero
            TP_plot = TP_plot / row_sums
        np.fill_diagonal(TP_plot, np.nan)  # mask diagonal as NaN for plotting

    if title_text is None:
        title_text = "Transition Probabilities" if with_self_transitions else "Transition Probabilities\nWithout Self-Transitions"

    # Tick logic
    if num_ticks is None:
        num_ticks = n_states if n_states <= 20 else 5
    tick_positions = np.linspace(0, n_states - 1, num_ticks).astype(int)
    tick_labels = [str(i + 1) for i in tick_positions]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(TP_plot, cmap=cmap, interpolation='none')

    ax.set_title(title_text, fontsize=fontsize_title)
    ax.set_xlabel("To State", fontsize=fontsize_labels)
    ax.set_ylabel("From State", fontsize=fontsize_labels)
    ax.tick_params(labelsize=tick_size)

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=tick_size)
    ax.set_yticklabels(tick_labels, fontsize=tick_size)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if return_fig:
        return fig
    else:
        plt.show()

def plot_state_prob_and_covariance(init_stateP, TP, state_means, state_FC, TP_with_self_trans=False, cmap='coolwarm',figsize=(9, 7), num_ticks=5,
                                save_path=None, title_size=None, label_size=None, tick_size=None, return_fig=False):
    """
    Plot HMM parameters: initial state probabilities, transition matrix,
    state means, and state covariance matrices.

    Parameters
    ----------
    init_stateP : np.ndarray
        Array of shape (n_states,) representing the initial state probabilities.
    TP : np.ndarray
        Transition probability matrix of shape (n_states, n_states).
    state_means : np.ndarray
        Array of shape (n_states, n_features) representing the mean activity per state.
    state_FC : np.ndarray
        Array of shape (n_features, n_features, n_states) representing state-specific covariance matrices.
    cmap : str or matplotlib colormap, default='coolwarm'
        Colormap used for all plots.
    figsize : tuple of float, default=(9, 7)
        Size of the full figure in inches (width, height).
    num_ticks : int, default=5
        Number of ticks to show on colorbars and axes.
    save_path : str or None, default=None
        If provided, the figure will be saved to this path.
    title_size : int or None, optional
        Font size for subplot titles. If None, automatically scaled based on figure size.
    label_size : int or None, optional
        Font size for axis labels (currently reserved for future extension).
    tick_size : int or None, optional
        Font size for tick labels. If None, automatically scaled based on figure size.
    """
    num_states = init_stateP.shape[0]
    num_cov_states = state_FC.shape[2]
    num_plots = 3 + num_cov_states
    num_cols = min(3, num_plots)
    num_rows = (num_plots + num_cols - 1) // num_cols  # ceil division

    # Dynamic sizing fallback
    base_scale = (figsize[0] + figsize[1]) / 2
    if title_size is None:
        title_size = int(base_scale * 1.6)
    if label_size is None:
        label_size = int(base_scale * 1.3)
    if tick_size is None:
        tick_size = int(base_scale * 1.2)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_2d(axes)

    # === Initial state probabilities ===
    im0 = axes[0, 0].imshow(init_stateP.reshape(-1, 1), cmap=cmap)
    axes[0, 0].set_title("Initial state\nprobabilities", fontsize=title_size)
    axes[0, 0].set_xticks([])
    axes[0, 0].tick_params(labelsize=tick_size)
    if num_states <= 10:
        yticks = np.arange(num_states)
        axes[0, 0].set_yticks(yticks)
        axes[0, 0].set_yticklabels(yticks + 1)
    else:
        axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=num_ticks, integer=True))
        yticks = axes[0, 0].get_yticks()
        axes[0, 0].set_yticks(yticks)
        axes[0, 0].set_yticklabels([int(t) + 1 for t in yticks if 0 <= t < num_states])
    #axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=num_ticks, integer=True))

    cbar0 = fig.colorbar(im0, ax=axes[0, 0])
    cbar0.locator = MaxNLocator(nbins=num_ticks)
    cbar0.update_ticks()
    cbar0.ax.tick_params(labelsize=tick_size)

    # === Transition probabilities ===
    if TP_with_self_trans== True:
        im1 = axes[0, 1].imshow(TP, cmap=cmap)
        axes[0, 1].set_title("Transition probabilities", fontsize=title_size)
        axes[0, 1].tick_params(labelsize=tick_size)

        if num_states <= 10:
            xyticks = np.arange(num_states)
            axes[0, 1].set_xticks(xyticks)
            axes[0, 1].set_xticklabels(xyticks + 1)
            axes[0, 1].set_yticks(xyticks)
            axes[0, 1].set_yticklabels(xyticks + 1)
        else:

            axes[0, 1].xaxis.set_major_locator(MaxNLocator(nbins=num_ticks, integer=True))
            axes[0, 1].yaxis.set_major_locator(MaxNLocator(nbins=num_ticks, integer=True))
    #    axes[0, 1].set_xticklabels([int(t) + 1 for t in axes[0, 1].get_xticks()])
    #    axes[0, 1].set_yticklabels([int(t) + 1 for t in axes[0, 1].get_yticks()])

        cbar1 = fig.colorbar(im1, ax=axes[0, 1])
        cbar1.locator = MaxNLocator(nbins=num_ticks)
        cbar1.update_ticks()
        cbar1.ax.tick_params(labelsize=tick_size)
    else:
        TP_noself = TP - np.diag(np.diag(TP))  # Remove self-transitions
        TP_noself2 = TP_noself / TP_noself.sum(axis=1, keepdims=True)  # Normalize probabilities

        im1 = axes[0, 1].imshow(TP_noself2, cmap=cmap)
        axes[0, 1].set_title("Transition probabilities\n w.o. self transition", fontsize=title_size)
        axes[0, 1].tick_params(labelsize=tick_size)
        if num_states <= 10:
            xyticks = np.arange(num_states)
            axes[0, 1].set_xticks(xyticks)
            axes[0, 1].set_xticklabels(xyticks + 1)
            axes[0, 1].set_yticks(xyticks)
            axes[0, 1].set_yticklabels(xyticks + 1)
        else:
            axes[0, 1].xaxis.set_major_locator(MaxNLocator(nbins=num_ticks, integer=True))
            axes[0, 1].yaxis.set_major_locator(MaxNLocator(nbins=num_ticks, integer=True))
        #    axes[0, 1].set_xticklabels([int(t) + 1 for t in axes[0, 1].get_xticks()])
        #    axes[0, 1].set_yticklabels([int(t) + 1 for t in axes[0, 1].get_yticks()])

        cbar1 = fig.colorbar(im1, ax=axes[0, 1])
        cbar1.locator = MaxNLocator(nbins=num_ticks)
        cbar1.update_ticks()
        cbar1.ax.tick_params(labelsize=tick_size)

    # === State Means ===
    im2 = axes[0, 2].imshow(state_means, cmap=cmap, aspect='auto')
    axes[0, 2].set_title("State means", fontsize=title_size)
    axes[0, 2].tick_params(labelsize=tick_size)
    num_features = state_means.shape[1]
    if num_states <= 10:
        xyticks = np.arange(num_states)
        axes[0, 2].set_xticks(xyticks)
        axes[0, 2].set_xticklabels(xyticks + 1)
    else:
        axes[0, 2].xaxis.set_major_locator(MaxNLocator(nbins=num_ticks, integer=True))
    axes[0, 2].yaxis.set_major_locator(MaxNLocator(nbins=num_ticks, integer=True))

    #ax.set_xticklabels(cov_ticks + 1, rotation=45 if len(cov_ticks) > 10 else 0)
    
    
#    axes[0, 2].set_xticklabels([int(t) + 1 for t in axes[0, 2].get_xticks()])
#    axes[0, 2].set_yticklabels([int(t) + 1 for t in axes[0, 2].get_yticks()])

    if num_features > 15:
        axes[0, 2].tick_params(axis='x', rotation=45)

    cbar2 = fig.colorbar(im2, ax=axes[0, 2])
    cbar2.locator = MaxNLocator(nbins=num_ticks)
    cbar2.update_ticks()
    cbar2.ax.tick_params(labelsize=tick_size)

    # === Covariances ===
    min_val, max_val = np.min(state_FC), np.max(state_FC)
    cov_ticks = MaxNLocator(nbins=num_ticks).tick_values(0, state_FC.shape[0]  - 1).astype(int)
    cov_ticks = cov_ticks[cov_ticks < state_FC.shape[0]]

    for k in range((num_cols * num_rows) - 3):
        row_idx = (k + 3) // num_cols
        col_idx = (k + 3) % num_cols

        if k < num_cov_states:
            ax = axes[row_idx, col_idx]
            im = ax.imshow(state_FC[:, :, k], cmap=cmap, vmin=min_val, vmax=max_val)
            ax.set_title(f"State covariance\nstate #{k + 1}", fontsize=title_size)
            ax.tick_params(labelsize=tick_size)

            ax.set_xticks(cov_ticks)
            ax.set_yticks(cov_ticks)
            ax.set_xticklabels(cov_ticks , rotation=45 if len(cov_ticks) > 10 else 0)
            ax.set_yticklabels(cov_ticks )

            cbar = fig.colorbar(im, ax=ax)
            cbar.locator = MaxNLocator(nbins=num_ticks)
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=tick_size)
        else:
            axes[row_idx, col_idx].axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if return_fig:
        return fig
    else:
        plt.show()


def plot_condition_difference(
    Gamma_epoch, R_trials, 
    title='Average Probability and Difference', 
    condition_labels=('Condition 1', 'Condition 2'), fontsize_sup_title=16,
    fontsize_title=14, fontsize_labels=12, figsize=(12, 3), vertical_lines=None, line_colors=None, 
    highlight_boxes=False, stimulus_onset=None, x_tick_min=None, 
    x_tick_max=None, num_x_ticks=5, num_y_ticks=5, xlabel='Timepoints', save_path=None, return_fig=False):
    """
    Plots the average probability for each state over time for two conditions and their difference.

    Parameters:
    -----------
    Gamma_epoch (numpy.ndarray)
        3D array representing reconstructed gamma values. Shape: (num_timepoints, num_trials, num_states)
    R_trials (numpy.ndarray)
        1D array representing the condition for each trial.
        Should have the same length as the second dimension of Gamma_epoch.
    title (str, optional), default='Average Probability and Difference':
        Title for the plot.
    condition_labels : tuple of str, optional
        Labels for the two conditions. Default is ('Condition 1', 'Condition 2').
    fontsize_sup_title (int, optional), default=16:
        Font size for sup_title.
    fontsize_title (int, optional), default=14:
        Font size for title.
    fontsize_labels (int, optional), default=12:
        Font size for labels.
    figsize (tuple, optional), default=(9, 2):
        Figure size (width, height).
    vertical_lines (list of tuples, optional), default=None:
        List of pairs specifying indices for vertical lines.
    line_colors (list of str or bool, optional), default=None:
        List of colors for each pair of vertical lines. If True, generates random colors
        (unless a list is provided).
    highlight_boxes (bool, optional), default=False:
        Whether to include highlighted boxes for each pair of vertical lines.
    stimulus_onset (int, optional), default=None:
        Index of the data where the stimulus onset should be positioned.
    x_tick_min (float, optional), default=None:
        Minimum value for the x-tick labels.
    x_tick_max (float, optional), default=None:
        Maximum value for the x-tick labels.
    num_x_ticks (int, optional), default=5:
        Number of x-ticks.
    num_y_ticks (int, optional), default=5:
        Number of y-ticks.
    save_path (str), optional, default=None
        If a string is provided, it saves the figure to that specified path
    Example usage:
    --------------
    plot_condition_difference(Gamma_epoch, R_trials, vertical_lines=[(10, 100)], highlight_boxes=True)
    """

    # Validate inputs
    if stimulus_onset is not None and not isinstance(stimulus_onset, (int, float)):
        raise ValueError("stimulus_onset must be a number.")
    if len(condition_labels) != 2:
        raise ValueError("condition_labels must be a tuple with exactly two labels.")
    
    filt_val = np.zeros((2, Gamma_epoch.shape[0], Gamma_epoch.shape[2]))

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    conditions = np.unique(R_trials)

    # Variables to store global min and max y-values
    global_min = float('inf')
    global_max = float('-inf')

    # Plot for each condition
    for idx, condition in enumerate(conditions):
        for i in range(Gamma_epoch.shape[0]):
            filtered_values = Gamma_epoch[i, (R_trials == condition), :]
            filt_val[idx, i, :] = np.mean(filtered_values, axis=0).round(3)
        # Update global min and max y-values
        current_min = filt_val[idx, :, :].min()
        current_max = filt_val[idx, :, :].max()
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)
        
    # Generate x-tick labels
    num_timepoints = Gamma_epoch.shape[0]
    x_tick_positions = np.linspace(0, num_timepoints - 1, num_x_ticks).astype(int)
    # Generate x-tick labels based on user input or default to time points
    if x_tick_min is not None and x_tick_max is not None:
        x_tick_labels = np.linspace(x_tick_min, x_tick_max, num_x_ticks).round(2)
        if np.all(x_tick_labels == x_tick_labels.astype(int)):
            x_tick_labels = x_tick_labels.astype(int)
    elif x_tick_min is not None:
        x_tick_labels = np.linspace(x_tick_min, pval.shape[1], num_x_ticks).round(2)
        if np.all(x_tick_labels == x_tick_labels.astype(int)):
            x_tick_labels = x_tick_labels.astype(int)
    elif x_tick_max is not None:
        x_tick_labels = np.linspace(0, x_tick_max, num_x_ticks).round(2)
        if np.all(x_tick_labels == x_tick_labels.astype(int)):
            x_tick_labels = x_tick_labels.astype(int) 
    else:
        x_tick_labels = x_tick_positions

    # Plot for each condition with standardized y-axis
    for idx, condition in enumerate(conditions):
        axes[idx].plot(filt_val[idx, :, :])
        axes[idx].set_title(condition_labels[idx], fontsize=fontsize_title)
        axes[idx].set_xticks(x_tick_positions)
        axes[idx].set_xticklabels(x_tick_labels)
        axes[idx].set_yticks(np.linspace(global_min, global_max, num_y_ticks).round(2))
        axes[idx].set_ylim(global_min, global_max)  # Set standardized y-limits # Set standardized y-limits
        axes[idx].set_xlim(x_tick_positions[0], x_tick_positions[-1])
        axes[idx].set_ylabel('Average Probability', fontsize=fontsize_labels)

        
    # Find the element-wise difference
    difference = filt_val[0, :, :] - filt_val[1, :, :]

    # Plot the difference
    axes[2].plot(difference)
    axes[2].set_title("Difference", fontsize=fontsize_title)

    axes[2].set_yticks(np.linspace(axes[2].get_ylim()[0], axes[2].get_ylim()[1], num_y_ticks).round(2))
    axes[2].set_xticks(x_tick_positions)
    axes[2].set_xticklabels(x_tick_labels)
    axes[2].set_xlim(x_tick_positions[0], x_tick_positions[-1])
    axes[2].set_xlabel(xlabel, fontsize=fontsize_labels)


    # Add stimulus onset line and label
    if stimulus_onset is not None:
        for ax in axes:
            ax.axvline(x=stimulus_onset, color='black', linestyle='--', linewidth=2)

    # Add vertical lines, line colors, and highlight boxes
    if vertical_lines:
        for idx, pair in enumerate(vertical_lines):
            color = line_colors[idx] if line_colors and len(line_colors) > idx else 'gray'
            axes[2].axvline(x=pair[0], color=color, linestyle='--', linewidth=1)
            axes[2].axvline(x=pair[1], color=color, linestyle='--', linewidth=1)

            if highlight_boxes:
                rect = plt.Rectangle((pair[0], axes[2].get_ylim()[0]), pair[1] - pair[0], axes[2].get_ylim()[1] - axes[2].get_ylim()[0], linewidth=0, edgecolor='none', facecolor=color, alpha=0.2)
                axes[2].add_patch(rect)

    # Set labels fontsize
    for ax in axes:
        ax.set_xlabel(xlabel, fontsize=fontsize_labels) 

    # Label each state on the right for the last figure (axes[2])
    state_labels = [f"State {state+1}" for state in range(Gamma_epoch.shape[2])]
    axes[2].legend(state_labels, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=fontsize_labels)

    fig.suptitle(title, fontsize=fontsize_sup_title)

    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Save the figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight') 
 
    
    
    if return_fig:
        return fig
    else:
        plt.show()
    
def plot_p_values_over_time(pval_in, figsize=(8, 3), xlabel="Timepoints", ylabel="P-values (Log Scale)",
                            title_text="P-values over time", fontsize_labels=12, fontsize_title=14, 
                            stimulus_onset=None, x_tick_min=None, x_tick_max=None, 
                            num_x_ticks=5, tick_positions=[0.001, 0.01, 0.05, 0.1, 0.3, 1], num_colors=259, 
                            alpha=0.05, plot_style="line", linewidth=2.5, scatter_on=True, save_path=None, return_fig=False):
    """
    Plot a scatter plot of p-values over time with a log-scale y-axis and a colorbar.

    Parameters:
    -----------
    pval (numpy.ndarray):
        The p-values data to be plotted.
    figsize (tuple), optional, default=(8, 4):
        Figure size in inches (width, height).
    total_time_seconds : float, optional, default=None
        Total time duration in seconds. If provided, time points will be scaled accordingly.
    xlabel (str, optional), default="Timepoints":
        Label for the x-axis.
    ylabel (str, optional), default="P-values (Log Scale)":
        Label for the y-axis.
    title_text (str, optional), default="P-values over time":
        Title for the plot.
    fontsize_labels (int, optional), default=12:
        Font size for the x and y-axis labels.
    fontsize_title (int, optional), default=14
        fontsize of title
    stimulus_onset (int, optional), default=None:
        Index of the data where the stimulus onset should be positioned.
    x_tick_min (float, optional), default=None
        Minimum value for x-axis ticks.
    x_tick_max (float, optional), default=None
        Maximum value for x-axis ticks.
    num_x_ticks (int, optional), default=5
        Number of x-axis ticks.
    tick_positions (list, optional), default=[0, 0.001, 0.01, 0.05, 0.1, 0.3, 1]:
        Specific values to mark on the y-axis.
    num_colors (int, optional), default=259:
        Resolution for the color bar.
    alpha (float, optional), default=0.05:
        Alpha value is the threshold we set for the p-values when doing visualization.
    plot_style (str, optional), default="line":
        Style of plot.
    linewidth (float, optional), default=2.5:
        Width of the lines in the plot.
    save_path (str), optional, default=None
        If a string is provided, it saves the figure to that specified path
    """

    # Check if stimulus_onset is a number
    if stimulus_onset is not None and not isinstance(stimulus_onset, (int, float)):
        raise ValueError("stimulus_onset must be a number.")
    
    pval = np.squeeze(pval_in.copy())
    if pval.ndim != 1:
        # Raise an exception and stop function execution
        raise ValueError("To use the function 'plot_p_values_over_time', the variable for p-values must be one-dimensional.")
    
    # Ensure p-values are within the log range
    pval_min = -3
    pval = np.clip(pval, 10**pval_min, 1)
    # Convert to log scale
    color_array = np.logspace(pval_min, 0, num_colors).reshape(1, -1)
    
    time_points = np.arange(len(pval))

    if alpha == None:
        # Create custom colormap
        coolwarm_cmap = custom_colormap()
        # Create a new colormap with the modified color_array
        cmap_list = coolwarm_cmap(color_array)[0]
        cmap_list = interpolate_colormap(cmap_list)
    else:    
        # Make a jump in color after alpha
        # Get blue colormap
        cmap_blue = blue_colormap()
        # Create a new colormap with 
        cmap_list = cmap_blue(color_array)[0]
        red_cmap = red_colormap()
        blue_cmap = blue_colormap()
        # Specify the number of elements you want (e.g., 50)
        num_elements_red = np.sum(color_array <= alpha)
        num_elements_blue = np.sum(color_array > alpha)

        # Generate equally spaced values between 0 and 1
        colormap_val_red = np.linspace(0, 1, num_elements_red)
        colormap_val_blue = np.linspace(0, 1, num_elements_blue)

        # Apply the colormap to the generated values
        cmap_red = red_cmap(colormap_val_red)
        cmap_blue = blue_cmap(colormap_val_blue)
        # overwrite the values below alpha
        cmap_list[:num_elements_red,:]=cmap_red
        cmap_list[num_elements_red:,:]=cmap_blue
    cmap = LinearSegmentedColormap.from_list('custom_colormap', cmap_list)        
    # Create the line plot with varying color based on p-values
    fig, axes = plt.subplots(figsize=figsize)

    # Normalize the data to [0, 1] for the colormap with logarithmic scale
    norm = LogNorm(vmin=10**pval_min , vmax=1)

    if plot_style == "line":
        # Plot the line segments with varying colors
        for i in range(len(time_points)-1):
            # Determine the color for the current segment
            if scatter_on and pval[i + 1] > alpha:
                color = cmap(norm(pval[i + 1]))
            else:
                color = cmap(norm(pval[i]))

            # Plot the line segment
            axes.plot([time_points[i], time_points[i + 1]],[pval[i], pval[i + 1]], color=color, linewidth=linewidth)

            if scatter_on:
                # Handle specific scatter cases
                if pval[i + 1] > alpha and pval[i] < alpha:
                    if i > 0 and pval[i - 1] < alpha:
                        pass  # Explicit no-op for clarity
                    else:
                        axes.scatter([time_points[i]],[pval[i]],c=pval[i],cmap=cmap,norm=norm)
    elif plot_style=="scatter":
        axes.scatter(time_points, pval, c=pval, cmap=cmap, norm=LogNorm(vmin=10**pval_min, vmax=1))
    elif plot_style=="scatter_line":
        axes.scatter(time_points, pval, c=pval, cmap=cmap, norm=LogNorm(vmin=10**pval_min, vmax=1))    
            # Draw lines between points
        axes.plot(time_points, pval, color='black', linestyle='-', linewidth=1)

    # Add labels and title
    axes.set_xlabel(xlabel, fontsize=fontsize_labels)
    axes.set_ylabel(ylabel, fontsize=fontsize_labels)
    axes.set_title(title_text, fontsize=fontsize_title)
    
    # define x_ticks
    x_tick_positions = np.linspace(0, len(pval), num_x_ticks).astype(int)

    # Generate x-tick labels based on user input or default to time points
    if x_tick_min is not None and x_tick_max is not None:
        x_tick_labels = np.linspace(x_tick_min, x_tick_max, num_x_ticks).round(2)
        if np.all(x_tick_labels == x_tick_labels.astype(int)):
            x_tick_labels = x_tick_labels.astype(int)
    elif x_tick_min is not None:
        x_tick_labels = np.linspace(x_tick_min, pval.shape[1], num_x_ticks).round(2)
        if np.all(x_tick_labels == x_tick_labels.astype(int)):
            x_tick_labels = x_tick_labels.astype(int)
    elif x_tick_max is not None:
        x_tick_labels = np.linspace(0, x_tick_max, num_x_ticks).round(2)
        if np.all(x_tick_labels == x_tick_labels.astype(int)):
            x_tick_labels = x_tick_labels.astype(int) 
    else:
        x_tick_labels = x_tick_positions

    # Set axis limits to focus on the relevant data range
    axes.set_xticks(x_tick_positions)
    axes.set_xticklabels(x_tick_labels)
    axes.set_xlim(x_tick_positions[0], x_tick_positions[-1])  # Set x-axis limits without white space
    axes.set_ylim([0.0008, 1.5])
    # Set y-axis to log scale
    axes.set_yscale('log')
    # Mark specific values on the y-axis
    plt.yticks([0.001, 0.01, 0.05, 0.1, 0.3, 1], ['0.001', '0.01', '0.05', '0.1', '0.3', '1'])
    
    # Add colorbar
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="3.5%", pad=0.05)
    colorbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax, ticks=tick_positions, format="%.3g"
    )

    plt.tight_layout()
    # Add stimulus onset line and label
    if stimulus_onset is not None:
        axes.axvline(x=stimulus_onset, color='black', linestyle='--', linewidth=2)
    # Save the figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight') 
        
    if return_fig:
        return fig
    else:
        plt.show()

def plot_p_values_bar(
    pval_in, xticklabels=None, figsize=(9, 4), num_colors=256, xlabel="",
    ylabel="P-values (Log Scale)", title_text="Bar Plot", fontsize_labels =12,fontsize_title=14,
    tick_positions=[0.001, 0.01, 0.05, 0.1, 0.3, 1], top_adjustment=0.8,
    alpha=0.05, pad_title=25, xlabel_rotation=45, pval_text_height_same=False,
    save_path=None, return_fig=False):
    """
    Visualize a bar plot with LogNorm and a colorbar.

    Parameters:
    -----------
    pval_in (numpy.ndarray):
        Array of p-values to be plotted.
    xticklabels (str or list, optional), default=None:
        Either a list of category labels, or a single string.
        - If a list: Must match the length of pval_in.
        - If a string: Auto-generates labels like "<string> 1", "<string> 2", ..., "<string> N"
        where N = len(pval_in).
        - If None or invalid: Default labels will be used ("Var 1", "Var 2", ..., "Var N").
    figsize (tuple, optional), default=(9, 4):
        Figure size in inches (width, height).
    num_colors (int, optional), default=256:
        Number of colors in the colormap.
    xlabel (str, optional), default="":
        X-axis label.
    ylabel (str, optional), default="P-values (Log Scale)":
        Y-axis label.
    title_text (str, optional), default="Bar Plot":
        Title for the plot.
    fontsize_labels (int, optional), default=12:
        Font size for the x and y-axis labels.
    fontsize_title (int, optional), default=14
        fontsize of title
    tick_positions (list, optional), default=[0, 0.001, 0.01, 0.05, 0.1, 0.3, 1]
        Positions of ticks on the colorbar.
    top_adjustment (float, optional), default=0.9:
        Adjustment for extra space between title and plot.
    alpha (float, optional), default=0.05:
        Alpha value is the threshold we set for the p-values when doing visualization.
    pad_title (int, optional), default=20:
        Padding for the plot title.
    save_path (str), optional, default=None
        If a string is provided, it saves the figure to that specified path
    pval_text_height_same (bool), default=False
        Whether the p-values of each bar should be plotted at the same height or adjusted to the height of each individual bar
    """
    # Validate input and flatten p-values
    pval = np.squeeze(pval_in).flatten() if pval_in.shape[0]==1 or pval_in.ndim==2 and np.any(np.array(pval_in.shape) == 1) else pval_in.copy()
    if pval.ndim != 1:
        raise ValueError("The input 'pval_in' must be a one-dimensional array.")


    # Validate xticklabels
    if xticklabels is not None:
        if isinstance(xticklabels, str):
            # Generate labels like "Hello 1", "Hello 2", ..., "Hello N"
            xticklabels = [f"{xticklabels} {i + 1}" for i in range(len(pval))]
        elif not isinstance(xticklabels, list):
            warnings.warn(f"xticklabels must be a list or a string, but got {type(xticklabels)}. Using default labels instead.")
            xticklabels = None
        elif len(xticklabels) != len(pval):
            raise ValueError(f"xticklabels length ({len(xticklabels)}) does not match pval length ({len(pval)}).")

    # Set default labels if needed
    if xticklabels is None or len(xticklabels) == 0:
        xticklabels = [f"Var {i + 1}" for i in range(len(pval))]

    # Ensure p-values are within the log range
    pval_min = -3
    pval = np.clip(pval, 10**pval_min, 1)
    # Convert to log scale
    color_array = np.logspace(pval_min, 0, num_colors).reshape(1, -1)

    if alpha == None:
        # Create custom colormap
        coolwarm_cmap = custom_colormap()
        # Create a new colormap with the modified color_array
        cmap_list = coolwarm_cmap(color_array)[0]
        cmap_list = interpolate_colormap(cmap_list)
    else:    
        # Make a jump in color after alpha
        # Get blue colormap
        cmap_blue = blue_colormap()
        # Create a new colormap with 
        cmap_list = cmap_blue(color_array)[0]
        red_cmap = red_colormap()
        blue_cmap = blue_colormap()
        # Specify the number of elements you want (e.g., 50)
        num_elements_red = np.sum(color_array <= alpha)
        num_elements_blue = np.sum(color_array > alpha)

        # Generate equally spaced values between 0 and 1
        colormap_val_red = np.linspace(0, 1, num_elements_red)
        colormap_val_blue = np.linspace(0, 1, num_elements_blue)

        # Apply the colormap to the generated values
        cmap_red = red_cmap(colormap_val_red)
        cmap_blue = blue_cmap(colormap_val_blue)
        # overwrite the values below alpha
        cmap_list[:num_elements_red,:]=cmap_red
        cmap_list[num_elements_red:,:]=cmap_blue

    
    # Create a LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('custom_colormap', cmap_list)

    # Plot the bar chart
    fig, axes = plt.subplots(figsize=figsize)
    norm = LogNorm(vmin=10**pval_min, vmax=1)
    bar_colors = cmap(norm(pval))
    bars = axes.bar(xticklabels, pval, color=bar_colors)

    # Add data labels above bars
    max_yval = max(pval) if pval_text_height_same else None
    for bar in bars:
        yval = bar.get_height().round(3)
        if yval ==1:
            yval = int(yval)
        text_y = max_yval + 0.5 if max_yval else yval + 0.5
        axes.text(
            bar.get_x() + bar.get_width() / 2, text_y, f"{yval}",
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    # Set logarithmic scale for y-axis
    axes.set_yscale('log')

    # Set y-axis limits explicitly to ensure it stays between 0.001 and 1
    # axes.set_ylim(0.001, 1)

    # Define tick positions and labels
    tick_positions = np.array(tick_positions)
    axes.set_yticks(tick_positions)
    axes.set_yticklabels([f"{pos:.3g}" for pos in tick_positions])

    # Customize plot aesthetics
    axes.set_xlabel(xlabel, fontsize=fontsize_labels)
    axes.set_ylabel(ylabel, fontsize=fontsize_labels)
    axes.set_title(title_text, fontsize=fontsize_title, pad=pad_title)

    # Define the tick positions explicitly
    axes.set_xticks(np.arange(len(xticklabels)))  # Set the tick positions based on xticklabels length
    axes.set_xticklabels(xticklabels, rotation=xlabel_rotation, ha='right' if xlabel_rotation == 45 else 'center')
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)

    # Add colorbar
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="3.5%", pad=0.05)
    colorbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax, ticks=tick_positions, format="%.3g"
    )

    # Add extra space for the title
    plt.subplots_adjust(top=top_adjustment)
    plt.tight_layout()
    # Save the plot if required
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if return_fig:
        return fig
    else:
        plt.show()


def plot_data_grid(data_list, titles=None, figsize_per_plot=(4, 3), 
                   main_title="Data", xlabel="Time (s)", ylabel="Signal", 
                   fontsize_labels=12, fontsize_title=18, line_width=1.8, grid=False, 
                   title_fontsize=14, tick_fontsize=10, standardize_yaxis=False, 
                   y_buffer=0.05, num_y_ticks=None, title_spacing=10, save_path=None, return_fig=False):
    """
    Create a grid of subplots to visualize multiple datasets with a clean layout.

    Parameters:
    -----------
    data_list (list of numpy.ndarray):
        List of data arrays to be plotted, each representing one subplot.
    titles (str, list of str, or None), default=None:
        Title for each plot. If None, default titles "Plot 1", "Plot 2", ... will be used.
        If a single string is provided, it will be numbered for each plot.
    figsize_per_plot (tuple, optional), default=(4, 3):
        Size of each subplot in the grid.
    main_title (str, optional), default="Data Visualization":
        Main title for the entire grid of subplots.
    xlabel (str, optional), default="Time (s)":
        Label for the x-axis of each subplot.
    ylabel (str, optional), default="Signal":
        Label for the y-axis of each subplot.
    fontsize_labels (int, optional), default=12:
        Font size for the x and y-axis labels.
    fontsize_title (int, optional), default=18:
        Font size for the main title.
    line_width (float, optional), default=1.8:
        Line width for the data plots.
    grid (bool, optional), default=False:
        Whether to include a grid in each subplot.
    title_fontsize (int, optional), default=14:
        Font size for subplot titles.
    tick_fontsize (int, optional), default=10:
        Font size for axis tick labels.
    standardize_yaxis (bool, optional), default=False:
        If True, sets the same y-axis limits for all plots based on the global min and max values.
    y_buffer (float, optional), default=0.05:
        Buffer added to the min and max y-axis values when standardizing, as a percentage.
    num_y_ticks (int, optional), default=None:
        Number of y-ticks for the y-axis. If None, matplotlib default is used.
    title_spacing (int, optional), default=10:
        Spacing (padding) between subplot titles and the plots.
    save_path (str, optional), default=None:
        If provided, saves the figure to the specified path.
    """
    # Calculate rows and columns for the grid
    n_plots = len(data_list)
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)

    # Handle titles: default to numbered "Plot X" if None
    if titles is None:
        titles = [f"Plot {i+1}" for i in range(n_plots)]
    elif isinstance(titles, str):
        titles = [f"{titles} {i+1}" for i in range(n_plots)]

    # Determine global y-axis limits if standardize_yaxis is True
    if standardize_yaxis:
        valid_min = min(np.nanmin(data) for data in data_list if np.isfinite(data).any())
        valid_max = max(np.nanmax(data) for data in data_list if np.isfinite(data).any())
        y_range = valid_max - valid_min
        y_min = valid_min - y_buffer * y_range
        y_max = valid_max + y_buffer * y_range

    # Create the grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_per_plot[0], n_rows * figsize_per_plot[1]))
    axes = axes.flatten()  # Flatten axes for easy iteration

    # Plot each dataset
    for i, (ax, data, title) in enumerate(zip(axes, data_list, titles)):
        ax.plot(data, linestyle='-', linewidth=line_width)
        ax.set_title(title, fontsize=title_fontsize, pad=title_spacing)
        ax.set_xlabel(xlabel, fontsize=fontsize_labels)
        ax.set_ylabel(ylabel, fontsize=fontsize_labels)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # Apply standardized y-axis limits
        if standardize_yaxis:
            ax.set_ylim(y_min, y_max)

        # Set number of y-ticks if specified
        if num_y_ticks is not None:
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=num_y_ticks))

        # Add grid
        if grid:
            ax.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Remove unused axes
    for ax in axes[len(data_list):]:
        fig.delaxes(ax)

    # Main title
    fig.suptitle(main_title, fontsize=fontsize_title, weight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the figure
    if return_fig:
        return fig
    else:
        plt.show()

def get_distinct_colors(n_colors, cmap=None):
    """
    Generate visually distinct colors using a combination of built-in categorical 
    and perceptually uniform colormaps.

    Parameters:
    --------------
    n_colors (int): 
        The number of distinct colors to generate.
    cmap (str, optional): 
        Name of a colormap to use as the primary source of colors. If not provided, 
        defaults to a combination of standard categorical colormaps. If the requested 
        number of colors exceeds what is available, additional colors are sampled 
        from other colormaps to fill the gap.

    Returns:
    --------------
    colors (list): 
        A list of RGBA tuples representing distinct colors. If the number of 
        requested colors exceeds what is available from standard categorical maps, 
        additional colors are sampled from a continuous colormap.
    """
    base_maps = ['Set3', 'tab10', 'Accent', 'Dark2']
    # Remove user-specified cmap if it's in base_maps to avoid duplication
    if cmap in base_maps:
        base_maps.remove(cmap)
        base_maps.insert(0,cmap)
    elif cmap is not None:
        base_maps.insert(0,cmap)
    colors = []

    for cmap_name in base_maps:
        cmap = cm.get_cmap(cmap_name)
        cmap_colors = [cmap(i) for i in range(cmap.N)]
        colors.extend(cmap_colors)

        if len(colors) >= n_colors:
            break

    if len(colors) < n_colors:
        # Fill remaining with perceptually uniform colors
        extra_needed = n_colors - len(colors)
        viridis = cm.get_cmap('gist_rainbow')
        colors.extend([viridis(i / extra_needed) for i in range(extra_needed)])

    return colors[:n_colors]

def plot_nnm_spectral_components(nnmf_components, freqs, x_lim=None, highlight_freq=True, 
                                 title='Spectral Components from NNMF Decomposition', cmap=None, bands=None, band_colors=None, 
                                 figsize=(10, 5), fontsize_labels=13, fontsize_title=16, band_legend_anchor=(1.28, 1), save_path=None, return_fig=False):
    """
    Plot the spectral components obtained from NNM decomposition with optional
    frequency band highlighting.

    Parameters:
    --------------
    nnmf_components (numpy.ndarray): 
        Array of shape (n_components, n_freqs) representing the decomposed 
        spectral components for each component.
    freqs (numpy.ndarray): 
        1D array representing the frequency axis, should match the second 
        dimension of `nnmf_components`.
    x_lim (int, optional): 
        The upper limit of the frequency axis (x-axis). If None, it will default 
        to the maximum value in `freqs`.
    highlight_freq (bool, optional): 
        Whether to highlight canonical or custom frequency bands, default is True.
    title (str, optional): 
        Title of the plot. Default is "Spectral Components from NNMF Decomposition".
    cmap (str, optional): 
        Name of a colormap to use for state line colors (default is 'Set3').
    bands (dict, optional): 
        Dictionary defining frequency bands. Keys are band names and values 
        are (start, end) tuples in Hz. If None, default bands will be used.
    band_colors (dict, optional): 
        Dictionary mapping band names to color names. Keys must match those in `bands`.
    figsize (tuple, optional): 
        Tuple defining figure size in inches, default is (10, 5).
    band_legend_anchor (tuple or None, optional): 
        Tuple for `bbox_to_anchor` to control frequency band legend placement. 
        Default is (1.28, 1). If set to None, legend is placed at 'lower right'.
    save_path (str), optional, default=None
        If a string is provided, it saves the figure to that specified path

    """
    if x_lim is None:
        x_lim = int(np.max(freqs))

    # Default frequency bands and colors
    default_bands = {
        'Delta': (0, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
        'Beta': (13, 30), 'Gamma': (30, x_lim)
    }
    default_band_colors = {
        'Delta': 'orange', 'Theta': 'cyan', 'Alpha': 'magenta',
        'Beta': 'black', 'Gamma': 'green'
    }

    # Use provided or default bands/colors
    bands = bands if bands is not None else default_bands
    band_colors = band_colors if band_colors is not None else default_band_colors

    # Validate consistency
    if set(bands.keys()) != set(band_colors.keys()):
        raise ValueError("The keys of 'bands' and 'band_colors' must match exactly.")

    n_components = nnmf_components.shape[0]

    # Assign distinct colors for each component
    if cmap is not None:
        # Assign distinct colors for each component
        valid_cmaps = plt.colormaps()
        if isinstance(cmap, str) and cmap in valid_cmaps:
            cmap = plt.get_cmap(cmap)
            component_colors = [cmap(i) for i in range(n_components)]
        else:
            warnings.warn(f"Invalid colormap '{cmap}'. Falling back to 'Set3'. "
                          f"Use one of: {', '.join(valid_cmaps[:5])}... etc.")
            cmap = plt.get_cmap('Set3')
            component_colors = [cmap(i) for i in range(n_components)]
    elif n_components <= 10:
        cmap = plt.get_cmap('Set3')
        component_colors = [cmap(i) for i in range(n_components)]
    else:
        component_colors = get_distinct_colors(n_components, cmap)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each component
    for i in range(n_components):
        ax.plot(
            freqs, nnmf_components[i],
            label=f'Component {i + 1}',
            linewidth=2.5,
            alpha=1,
            color=component_colors[i],
        )

    # Highlight frequency bands
    if highlight_freq:
        for band, (start, end) in bands.items():
            ax.axvspan(start, end, color=band_colors[band], alpha=0.2)

    ax.set_xlabel('Frequency (Hz)', fontsize = fontsize_labels)
    ax.set_ylabel('Component Weight', fontsize = fontsize_labels)
    ax.set_title(title, fontsize = fontsize_title)
    ax.set_xlim(0, x_lim)

    # Legend for components
    state_legend = ax.legend(loc='upper right', title='NNMF Components')

    # Optional frequency band legend
    if highlight_freq:
        band_handles = [
            Patch(facecolor=band_colors[band], edgecolor='none', alpha=0.5,
                label=f'{band}: {start}-{end} Hz')
            for band, (start, end) in bands.items()
        ]

        if band_legend_anchor is not None:
            band_legend = ax.legend(
                handles=band_handles,
                loc='upper right',
                bbox_to_anchor=band_legend_anchor,
                title='Frequency Bands'
            )
        else:
            band_legend = ax.legend(
                handles=band_handles,
                loc='lower right',
                title='Frequency Bands'
            )

        ax.add_artist(state_legend)

    plt.tight_layout()
    # Save the figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight') 

    if return_fig:
        return fig
    else:
        plt.show()

def plot_state_psd(psd, freqs, significant_states=None, x_lim=None, cmap=None, highlight_freq=False, bands=None,
    band_colors=None, title='Power Spectral Density (PSD) per State', log_scale_y=False, log_scale_x=False, figsize=(10, 5), 
    fontsize_labels=13, fontsize_title=16, band_legend_anchor=(1.28, 1), label_line=None, save_path=None, return_fig=False):
    """
    Plot the power spectral density (PSD) for each state, with optional 
    highlighting of frequency bands and significant states.

    Parameters:
    --------------
    psd (numpy.ndarray): 
        Array of shape (n_freqs, num_states) representing the PSD of each state.
    freqs (numpy.ndarray): 
        1D array representing the frequency axis, should match the second 
        dimension of `psd`.
    significant_states (set, optional): 
        Set of 1-based indices indicating which states are considered significant. 
        Significant states are highlighted with a solid line and an asterisk.
    x_lim (int, optional): 
        The upper limit of the frequency axis (x-axis). If None, it will default 
        to the maximum value in `freqs`.
    cmap (str, optional): 
        Name of a colormap to use for state line colors (default is 'Set3').
    highlight_freq (bool, optional): 
        Whether to highlight canonical or custom frequency bands. Default is True.
    bands (dict, optional): 
        Dictionary defining frequency bands. Keys are band names and values are 
        (start, end) tuples in Hz. If None, default bands will be used.
    band_colors (dict, optional): 
        Dictionary mapping band names to color names. Keys must match those in `bands`.
    title (str, optional): 
        Title of the plot. Default is "Power Spectral Density (PSD) per State".
    log_scale_y (bool, optional): 
        Whether to apply a logarithmic scale to the y-axis (power). Default is True.
    log_scale_x (bool, optional): 
        Whether to apply a logarithmic scale to the x-axis (frequency). Default is False.
    figsize (tuple, optional): 
        Tuple defining figure size in inches. Default is (10, 5).
    fontsize_labels (int): 
        Font size for x and y axis labels. Default is 13.
    fontsize_title (int): 
        Font size for the plot title. Default is 16.
    band_legend_anchor (tuple or None, optional): 
        Tuple for `bbox_to_anchor` to control frequency band legend placement. 
        Default is (1.28, 1). If set to None, legend is placed at 'upper right'.
    label_line (str or list, optional): 
        Custom labels for each line. Can be a string (prefix) or a list of names. 
        If not provided, states are labeled as "State 1", "State 2", etc.
    save_path (str, optional, default=None): 
        If a string is provided, the figure will be saved to the specified path.

    """

    if significant_states is None:
        significant_states = set()

    if x_lim is None:
        x_lim = int(np.max(freqs))

    if np.iscomplexobj(psd):
        print("Warning: PSD contains complex values — imaginary parts will be discarded.")
        psd = np.real(psd)

    default_bands = {
        'Delta': (0, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
        'Beta': (13, 30), 'Gamma': (30, x_lim)
    }
    default_band_colors = {
        'Delta': 'orange', 'Theta': 'cyan', 'Alpha': 'magenta',
        'Beta': 'black', 'Gamma': 'green'
    }
    bands = bands if bands is not None else default_bands
    band_colors = band_colors if band_colors is not None else default_band_colors

    if set(bands.keys()) != set(band_colors.keys()):
        raise ValueError("The keys of 'bands' and 'band_colors' must match exactly.")

    num_states = psd.shape[1] if psd.ndim == 2 else 1
    psd = psd[:, np.newaxis] if psd.ndim==1 else psd

    if isinstance(label_line, list):
        if len(label_line) != num_states:
            raise ValueError("Length of 'label_line' list must match number of lines to plot.")
        line_labels = label_line
    elif isinstance(label_line, str):
        line_labels = [f"{label_line} {i+1}" if num_states > 1 else label_line for i in range(num_states)]
    else:
        line_labels = [f"State {i+1}" if num_states > 1 else "State" for i in range(num_states)]

    # Assign distinct colors for each component
    if cmap is not None:
        # Assign distinct colors for each component
        valid_cmaps = plt.colormaps()
        if isinstance(cmap, str) and cmap in valid_cmaps:
            if num_states <= 10:
                cmap = plt.get_cmap(cmap)
                component_colors = [cmap(i) for i in range(num_states)]
            else: 
                component_colors = get_distinct_colors(num_states, cmap)
        else:
            warnings.warn(f"Invalid colormap '{cmap}'. Falling back to 'Set3'. "
                          f"Use one of: {', '.join(valid_cmaps[:5])}... etc.")
            cmap = plt.get_cmap('Set3')
            component_colors = [cmap(i) for i in range(num_states)]
    elif num_states ==1:
        cmap = plt.get_cmap('tab10')
        component_colors = [cmap(i) for i in range(num_states)]
    elif num_states <= 10:
        cmap = plt.get_cmap('Set3')
        component_colors = [cmap(i) for i in range(num_states)]
    else:
        component_colors = get_distinct_colors(num_states, cmap)

    fig, ax = plt.subplots(figsize=figsize)
    if log_scale_y:
        if np.any(psd <= 0):
            warnings.warn("log_scale_y=True, but PSD contains non-positive values. Log scale will be skipped.")
            log_scale_y = False
        else:
            ax.set_yscale('log')
        

    if log_scale_x:
        ax.set_xscale('log')
    for state in range(num_states):
        state_num = state + 1
        color = component_colors[state]
        label = line_labels[state]
        y_vals = psd[:, state]

        if state_num in significant_states:
            ax.plot(freqs, y_vals, label=f"{label} *", linewidth=2.5, linestyle='-', alpha=1.0, color=color, zorder=3)
        elif significant_states == set():
            ax.plot(freqs, y_vals, label=label, linewidth=2.5, linestyle='-', alpha=1.0, color=color, zorder=2)
        else:
            ax.plot(freqs, y_vals, label=label, linewidth=1.5, linestyle='--', alpha=0.8, color=color, zorder=2)

    for state in significant_states:
        max_abs_val = psd[np.abs(psd[state - 1]).argmax()][state - 1]
        ax.text(x_lim - 15, max_abs_val, f"{line_labels[state - 1]} *", fontsize=12, fontweight='bold')

    if highlight_freq:
        for band, (start, end) in bands.items():
            ax.axvspan(start, end, color=band_colors[band], alpha=0.2)

    ax.set_xlabel('Frequency (Hz)', fontsize=fontsize_labels)
    ax.set_ylabel('Power', fontsize=fontsize_labels)
    ax.set_title(title, fontsize=fontsize_title)
    ax.set_xlim(0, x_lim)



    if label_line is None:
        legend_title = 'States'
    elif isinstance(label_line, str):
        legend_title = label_line if num_states > 1 else ''  # use 'label_line' as title if multiple lines
    elif isinstance(label_line, list):
        legend_title = 'Lines'
    else:
        legend_title = ''

    state_legend = ax.legend(loc='upper right', title=legend_title)

    if highlight_freq:
        band_handles = [
            Patch(facecolor=band_colors[band], edgecolor='none', alpha=0.5,
                label=f'{band}: {start}-{end} Hz')
            for band, (start, end) in bands.items()
        ]
        band_legend = ax.legend(
            handles=band_handles,
            loc='upper right' if band_legend_anchor is None else 'upper right',
            bbox_to_anchor=band_legend_anchor if band_legend_anchor else None,
            title='Frequency Bands'
        )
        ax.add_artist(state_legend)

    plt.tight_layout()
    # Save the figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight') 
    if return_fig:
        return fig
    else:
        plt.show()


def plot_state_coherence(coh, freqs, significant_states=None, x_lim=None, cmap=None,
                                highlight_freq=False, bands=None, band_colors=None, 
                                title='State Coherence between Two Channels', figsize=(10, 5), 
                                fontsize_labels=13, fontsize_title=16, 
                                band_legend_anchor=(1.28, 1), label_line=None, save_path=None, return_fig=False):
    """
    Plot the coherence between two channels for each state, with optional 
    highlighting of frequency bands and significant states.

    Parameters:
    --------------
    coh (numpy.ndarray): 
        Array of shape (n_freqs, num_states) representing the coherence between 
        two selected channels for each state.
    freqs (numpy.ndarray): 
        1D array representing the frequency axis, should match the first 
        dimension of `coh`.
    significant_states (set, optional): 
        Set of 1-based indices indicating which states are considered significant. 
        Significant states are highlighted with a solid line and an asterisk.
    x_lim (int, optional): 
        The upper limit of the frequency axis (x-axis). If None, it will default 
        to the maximum value in `freqs`.
    cmap (str, optional): 
        Name of a colormap to use for state line colors (default is 'Set3').
    highlight_freq (bool, optional): 
        Whether to highlight canonical or custom frequency bands. Default is False.
    bands (dict, optional): 
        Dictionary defining frequency bands. Keys are band names and values are 
        (start, end) tuples in Hz. If None, default bands will be used.
    band_colors (dict, optional): 
        Dictionary mapping band names to color names. Keys must match those in `bands`.
    title (str, optional): 
        Title of the plot. Default is "State Coherence between Two Channels".
    figsize (tuple, optional): 
        Tuple defining figure size in inches. Default is (10, 5).
    fontsize_labels (int): 
        Font size for x and y axis labels. Default is 13.
    fontsize_title (int): 
        Font size for the plot title. Default is 16.
    band_legend_anchor (tuple or None, optional): 
        Tuple for `bbox_to_anchor` to control frequency band legend placement. 
        Default is (1.28, 1). If set to None, legend is placed at 'upper right'.
    label_line (str or list, optional): 
        Custom labels for each line. Can be a string (prefix) or a list of names. 
        If not provided, states are labeled as "State 1", "State 2", etc.
    save_path (str, optional, default=None): 
        If a string is provided, the figure will be saved to the specified path.
    """
    if significant_states is None:
        significant_states = set()

    if x_lim is None:
        x_lim = int(np.max(freqs))

    coh = np.real(coh)
    n_freqs, num_states = coh.shape

    # Labels
    if isinstance(label_line, list):
        if len(label_line) != num_states:
            raise ValueError("Length of 'label_line' must match number of states.")
        line_labels = label_line
    elif isinstance(label_line, str):
        line_labels = [f"{label_line} {i+1}" for i in range(num_states)]
    else:
        line_labels = [f"State {i+1}" for i in range(num_states)]

    # Bands and colors
    default_bands = {
        'Delta': (0, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
        'Beta': (13, 30), 'Gamma': (30, x_lim)
    }
    default_band_colors = {
        'Delta': 'orange', 'Theta': 'cyan', 'Alpha': 'magenta',
        'Beta': 'black', 'Gamma': 'green'
    }
    bands = bands if bands is not None else default_bands
    band_colors = band_colors if band_colors is not None else default_band_colors

    if set(bands.keys()) != set(band_colors.keys()):
        raise ValueError("Keys of bands and band_colors must match.")

    # Assign distinct colors for each component
    if cmap is not None:
        # Assign distinct colors for each component
        valid_cmaps = plt.colormaps()
        if isinstance(cmap, str) and cmap in valid_cmaps:
            if num_states <= 10:
                cmap = plt.get_cmap(cmap)
                component_colors = [cmap(i) for i in range(num_states)]
            else: 
                component_colors = get_distinct_colors(num_states, cmap)
        else:
            warnings.warn(f"Invalid colormap '{cmap}'. Falling back to 'Set3'. "
                          f"Use one of: {', '.join(valid_cmaps[:5])}... etc.")
            cmap = plt.get_cmap('Set3')
            component_colors = [cmap(i) for i in range(num_states)]
    elif num_states <= 10:
        cmap = plt.get_cmap('Set3')
        component_colors = [cmap(i) for i in range(num_states)]
    else:
        component_colors = get_distinct_colors(num_states, cmap)


    fig, ax = plt.subplots(figsize=figsize)

    for state in range(num_states):
        state_idx = state + 1
        label = line_labels[state]
        color = component_colors[state]
        y_vals = coh[:, state]

        if state_idx in significant_states:
            ax.plot(freqs, y_vals, label=f"{label} *", linewidth=2.5, linestyle='-', alpha=1.0, color=color)
        elif significant_states == set():
            ax.plot(freqs, y_vals, label=label, linewidth=2.5, linestyle='-', alpha=1.0, color=color)
        else:
            ax.plot(freqs, y_vals, label=label, linewidth=1.5, linestyle='--', alpha=0.8, color=color)

    # Highlight frequency bands
    if highlight_freq:
        for band, (start, end) in bands.items():
            ax.axvspan(start, end, color=band_colors[band], alpha=0.2)

    ax.set_xlabel('Frequency (Hz)', fontsize=fontsize_labels)
    ax.set_ylabel('Coherence', fontsize=fontsize_labels)
    ax.set_title(title, fontsize=fontsize_title)
    ax.set_xlim(0, x_lim)

    # Legends
    state_legend = ax.legend(loc='upper right', title='States')

    if highlight_freq:
        band_handles = [
            Patch(facecolor=band_colors[band], edgecolor='none', alpha=0.5,
                  label=f'{band}: {start}-{end} Hz')
            for band, (start, end) in bands.items()
        ]
        band_legend = ax.legend(
            handles=band_handles,
            loc='upper right' if band_legend_anchor is None else 'upper right',
            bbox_to_anchor=band_legend_anchor,
            title='Frequency Bands'
        )
        ax.add_artist(state_legend)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if return_fig:
        return fig
    else:
        plt.show()



def check_exists(filename, fallback_directory="."):
    """
    Check if a file exists, optionally falling back to a secondary directory.

    Parameters
    ----------
    filename (str)
        File path or name to check.
    fallback_directory (str), optional
        Directory to look in if the file is not found at the original path.
        Default is current directory.

    Returns
    -------
    str
        Full path to the found file.

    Raises
    ------
    FileNotFoundError
        If the file is not found in either location.
    """
    if not os.path.exists(filename):
        fallback = os.path.join(fallback_directory, filename)
        if os.path.exists(fallback):
            return fallback
        else:
            raise FileNotFoundError(filename)
    return filename

def validate(array, correct_dimensionality, allow_dimensions, error_message):
    """
    Validate and reshape an array to the correct dimensionality.

    Parameters
    ----------
    array (np.ndarray)
        The array to validate.
    correct_dimensionality : int
        The target number of dimensions.
    allow_dimensions : list of int
        Acceptable dimensionalities that will be expanded to the target.
    error_message (str)
        Error message to raise if dimensionality is incorrect.

    Returns
    -------
    np.ndarray
        Array reshaped to the correct number of dimensions.

    Raises
    ------
    ValueError
        If the input does not meet dimensionality requirements.
    """
    array = np.array(array)
    for dimensionality in allow_dimensions:
        if array.ndim == dimensionality:
            for _ in range(correct_dimensionality - dimensionality):
                array = array[np.newaxis, ...]
    if array.ndim != correct_dimensionality:
        raise ValueError(error_message)
    return array

def parcel_vector_to_voxel_grid(mask_file, parcellation_file, vector):
    """
    Map a vector of parcel values onto a 3D voxel grid using a mask and parcellation.

    Parameters
    ----------
    mask_file (str)
        Path to a NIfTI mask file.
    parcellation_file (str)
        Path to a NIfTI parcellation file.
    vector (np.ndarray)
        1D array of values (one per parcel). Shape must match number of parcels.

    Returns
    -------
    np.ndarray
        3D voxel grid with mapped values at each brain voxel.

    Raises
    ------
    ValueError
        If the number of parcels in the parcellation does not match the length of the vector.
    """
        
    logging.getLogger("nibabel.global").setLevel(logging.ERROR)
    mask_file = check_exists(mask_file)
    parcellation_file = check_exists(parcellation_file)

    mask = nib.load(mask_file)
    mask_grid = mask.get_fdata().ravel(order="F")
    non_zero_voxels = mask_grid != 0

    parcellation = nib.load(parcellation_file)
    parcellation_grid = parcellation.get_fdata()
    if parcellation_grid.ndim == 3:
        unique_values = np.unique(parcellation_grid)[1:]
        parcellation_grid = np.array([(parcellation_grid == v).astype(int) for v in unique_values])
        parcellation_grid = np.rollaxis(parcellation_grid, 0, 4)
        parcellation = nib.Nifti1Image(parcellation_grid, parcellation.affine, parcellation.header)

    parcellation = resample_to_img(parcellation, mask, force_resample=True, copy_header=True)
    parcellation_grid = parcellation.get_fdata()
    n_parcels = parcellation.shape[-1]

    if vector.shape[0] != n_parcels:
        raise ValueError("parcellation_file has a different number of parcels to the vector")

    voxel_weights = parcellation_grid.reshape(-1, n_parcels, order="F")[non_zero_voxels]
    voxel_weights /= voxel_weights.max(axis=0, keepdims=True)
    voxel_values = voxel_weights @ vector

    voxel_grid = np.zeros(mask_grid.shape[0])
    voxel_grid[non_zero_voxels] = voxel_values
    voxel_grid = voxel_grid.reshape(mask.shape, order="F")

    return voxel_grid

def get_custom_colormap():
    """
    Create a custom colormap for brain activation plotting.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        A custom colormap transitioning through turquoise, blue, gray, and red-yellow.
    """
    colors = ["#00FA9A", "#40E0D0", "#0000FF", "#BFBFBF", "#FF0000", "#FFA500", "#FFFF00"]
    positions = [0.0, 0.16, 0.33, 0.5, 0.66, 0.83, 1.0]
    return LinearSegmentedColormap.from_list("brain_activation_updated", list(zip(positions, colors)), N=256)



def save_figure(fig, path, fig_format, show=False):
    """
    Save a matplotlib figure to disk and optionally close it.

    Parameters
    ----------
    fig (matplotlib.figure.Figure)
        The figure to save.
    path (str):
        Output path where the figure will be saved.
    fig_format (str):
        Format to save the figure (e.g., 'svg', 'png').
    show (bool):
        Whether to keep the figure open (True) or close it (False).
    """
    fig.savefig(path, format=fig_format)
    if not show:
        plt.close(fig)

def plot_brain_state_maps(power_map, mask_file, parcellation_file, filename=None, fig_format="png", component=0, subtract_mean=False,
                        mean_weights=None, match_color_scale=False, plot_kwargs=None, show_plots=True, combined=False,
                        titles=None, n_rows=1, save_figures=False, figure_filenames=None, save_folder_name="brain_maps", return_fig=False):
    """
    Plots or saves power spectral brain state maps projected to surface.

    Parameters:
    --------------
    power_map (np.ndarray):
        Array of shape (n_components, n_modes, n_channels) or similar.
    mask_file (str):
        Path to NIfTI mask file.
    parcellation_file (str):
        Path to NIfTI parcellation file.
    filename (str, optional):
        Base filename for saving output. Supports .nii/.nii.gz/.png/.svg/.pdf.
    fig_format (str, optional), default='png':
        File format for figure export (e.g., "pdf", "png").
    component (int, optional):
        Index of the spectral component to plot.
    subtract_mean (bool, optional), default=False:
        Whether to subtract mean across modes.
    mean_weights (np.ndarray, optional):
        Weights for computing the average across modes.
    match_color_scale (bool, optional), default=False:
        Force consistent vmin/vmax across plots.
    plot_kwargs : dict, optional
        Keyword arguments passed to `nilearn.plotting.plot_img_on_surf`.
        Common options include:
            - surf_mesh (str) or dict, default='fsaverage5'
                Cortical mesh to use for plotting.
            - hemispheres : list of str, default=['left', 'right']
                Hemispheres to show ('left', 'right', or both).
            - views : list of str, default=['lateral', 'medial']
                View angles for each hemisphere.
            - inflate (bool), default=False
                Whether to use an inflated surface.
            - bg_on_data (bool), default=False
                Whether to blend background surface with data overlay.
            - title (str), optional
                Title shown above each surface plot.
            - colorbar (bool), default=True
                Show colorbar alongside the figure.
            - vmin, vmax : float, optional
                Value range for colormap.
            - threshold : float, optional
                Values below this (in absolute value) are masked out.
            - symmetric_cbar (bool) or 'auto', default='auto'
                Whether to center the colorbar symmetrically around zero.
            - cmap (str) or colormap, default='cold_hot'
                Colormap used for surface data.
            - cbar_tick_format (str), default='%i'
                Tick formatting for the colorbar.
    show_plots (bool, optional):
        Whether to display the plots.
    combined (bool, optional):
        Combine plots into a single figure (enables save_figures).
    titles (list or bool, optional):
        List of titles for each mode or True for auto-generated labels.
    n_rows (int, optional):
        Number of rows in the combined figure.
    save_figures (bool, optional):
        Whether to save each plot as a file.
    figure_filenames (str or list, optional):
        Base name or list of full paths for saving each plot.
    save_folder_name (str, optional): 
        Name of the output folder where saved figures will be stored. Default is "brain_maps".
    """

    power_map = np.squeeze(power_map)
    if power_map.ndim > 1 and power_map.shape[-1] == power_map.shape[-2]:
        power_map = np.diagonal(power_map, axis1=-2, axis2=-1)
        if power_map.ndim == 1:
            power_map = power_map[np.newaxis, ...]
    else:
        power_map = power_map[np.newaxis, ...]

    power_map = validate(power_map, 3, [2], "power_map.shape is incorrect")
    n_modes = power_map.shape[1]

    if subtract_mean and n_modes > 1:
        power_map -= np.average(power_map, axis=1, weights=mean_weights)[:, np.newaxis, ...]

    power_map = power_map[component]
    mask_file = check_exists(mask_file)
    parcellation_file = check_exists(parcellation_file)
    power_map = [parcel_vector_to_voxel_grid(mask_file, parcellation_file, p) for p in power_map]
    power_map = np.moveaxis(power_map, 0, -1)
    mask = nib.load(mask_file)

    if plot_kwargs is None:
        plot_kwargs = {}
    # Use custom colormap by default unless overridden
    if "cmap" not in plot_kwargs:
        plot_kwargs["cmap"] = get_custom_colormap()

    if combined:
        if not save_figures:
            print("[Warning] 'combined=True' requires 'save_figures=True'. Enabling save_figures automatically.")
            save_figures = True
        if fig_format.lower() != "png":
            print("[Warning] switching fig_format to 'png' when 'combined=True'.")
            fig_format = "png"

    if "cbar_tick_format" in plot_kwargs:
        if plot_kwargs["cbar_tick_format"] is True:
            plot_kwargs["cbar_tick_format"] = "%.2f"
        elif plot_kwargs["cbar_tick_format"] is False:
            plot_kwargs["cbar_tick_format"] = ""

    if titles is None:
        titles = [None] * n_modes
    elif titles is True:
        titles = [f"State {i+1}" for i in range(n_modes)]
    elif isinstance(titles, str):
        titles = [f"{titles} {i+1}" for i in range(n_modes)]
    elif isinstance(titles, list):
        if len(titles) != n_modes:
            raise ValueError("Length of 'titles' must match number of modes.")
    
    if match_color_scale:
        if plot_kwargs.get("symmetric_cbar", False) is True:
            abs_max = np.nanmax(np.abs(power_map))
            plot_kwargs["vmin"] = -abs_max
            plot_kwargs["vmax"] = abs_max
        else:
            plot_kwargs["vmin"] = np.nanmin(power_map)
            plot_kwargs["vmax"] = np.nanmax(power_map)

    output_files = []
    # PATH_OUTPUT = Path(".") if filename is None else Path(filename).parent
    # base_filename = Path(filename).stem if filename else "power_map"
    PATH_OUTPUT, base_filename = __resolve_figure_directory(save_figures, filename, default_folder=save_folder_name)

    for i in trange(n_modes, desc="Saving images", disable=not show_plots):
        nii = nib.Nifti1Image(power_map[:, :, :, i], mask.affine, mask.header)

        fig, ax = plotting.plot_img_on_surf(nii, output_file=None, **plot_kwargs)

        if plot_kwargs.get("colorbar", True):
            for axes_obj in fig.axes:
                if hasattr(axes_obj, 'get_position'):
                    bbox = axes_obj.get_position()
                    if bbox.width < 0.4 and bbox.height < 0.05:
                        axes_obj.set_position([0.2, 0.05, 0.6, 0.025])
                        if hasattr(axes_obj, 'collections') and axes_obj.collections:
                            colorbar = getattr(axes_obj.collections[0], 'colorbar', None)
                            if colorbar and plot_kwargs.get("cbar_tick_format"):
                                colorbar.locator = ticker.MaxNLocator(nbins=4)
                                colorbar.formatter = ticker.FormatStrFormatter(plot_kwargs["cbar_tick_format"])
                                colorbar.update_ticks()

        if titles:
            fig.suptitle(titles[i], fontsize=20)

        # Save figure if requested
        if save_figures or combined:
            base = figure_filenames if isinstance(figure_filenames, str) else base_filename
            path_fig = PATH_OUTPUT / __generate_filename(base, i, fig_format)
            fig.savefig(path_fig, format=fig_format)
            output_files.append(path_fig)

            if not show_plots:
                plt.close(fig)

    if combined:
        n_columns = -(n_modes // -n_rows)
        fig, axes_grid = plt.subplots(n_rows, n_columns, figsize=(n_columns * 5, n_rows * 5))
        for i, ax in enumerate(axes_grid.flatten()):
            ax.axis("off")
            if i < len(output_files):
                ax.imshow(plt.imread(output_files[i]))
        fig.tight_layout()
        combined_path = PATH_OUTPUT / f"{base_filename}_combined.{fig_format}"
        fig.savefig(combined_path)
        if not show_plots:
            plt.close(fig)
    if return_fig:
        return fig
    else:
        plt.show()

def update_save_flags(save_figures, combined, fig_format):
    """
    Updates save_figures and fig_format based on combined flag.

    Parameters
    ----------
    save_figures : bool
        Whether to save individual figures.
    combined : bool
        Whether to save a combined multi-panel figure.
    fig_format : str
        Desired figure format (e.g., 'pdf', 'png').

    Returns
    -------
    save_figures : bool
        Updated save_figures flag.
    fig_format : str
        Updated fig_format (forces 'png' if combined).
    """
    if combined:
        if not save_figures:
            print("[Info] 'combined=True' now also saves individual figures.")
            save_figures = True
        if fig_format.lower() != "png":
            print("[Info] Combined figure forced to PNG format.")
            fig_format = "png"
    return save_figures, fig_format


def get_parcellation_centers(parcellation_file):
    """
    Extracts MNI coordinates for each parcel in a 4D NIfTI parcellation.

    Parameters:
    --------------
    parcellation_file (str):
        Path to a 4D binary NIfTI file where each volume corresponds to a parcel.

    Returns:
    --------------
    centers (np.ndarray):
        Array of shape (n_parcels, 3) containing the MNI coordinates for each parcel.
    """
        
    img = nib.load(parcellation_file)
    data = img.get_fdata()
    affine = img.affine

    if data.ndim == 4:
        n_parcels = data.shape[-1]
        centers = []
        for i in range(n_parcels):
            parcellation_coords = np.argwhere(data[..., i] > 0)
            if parcellation_coords.size == 0:
                centers.append([np.nan, np.nan, np.nan])
            else:
                voxel_center = parcellation_coords.mean(axis=0)
                world_center = nib.affines.apply_affine(affine, voxel_center)
                centers.append(world_center)
        return np.array(centers)
    else:
        raise ValueError("Parcellation file must be 4D.")
    

def plot_connectivity_maps(connectivity_map, parcellation_file, filename=None, fig_format="png", component=None, threshold=0,
                           match_color_scale = True, plot_kwargs=None, show_plots=True, axes=None, combined=False,
                           save_figures=False, titles=None, n_rows=1, figure_filenames=None, save_folder_name="connectivity_maps", return_fig=False):
    """
    Plot connectivity maps, such as functional or spectral connectivity, using a parcellation layout.

    Parameters:
    --------------
    connectivity_map (numpy.ndarray): 
        Array of shape (n_modes, n_channels, n_channels) or (n_components, n_modes, n_channels, n_channels). 
        Represents connectivity matrices for each mode (or component and mode).
    parcellation_file (str): 
        Path to a parcellation file used to define node coordinates for plotting the connectome.
    filename (str, optional): 
        If provided, the base filename for saving the figure(s). The appropriate format 
        (e.g., .png, .svg) will be determined by `fig_format`.
    fig_format (str, optional): 
        Format to save the figures, e.g., 'png', 'svg', or 'pdf'. Default is 'png'.
    component (int, optional): 
        If connectivity_map is 4D, this selects which component to plot. If None, all components are plotted.
    threshold (float, optional): 
        Minimum absolute value for showing a connection. Values below this threshold are not shown. 
        Default is 0 (no thresholding).
    match_color_scale (bool, optional): 
        Whether to use the same color scale across all plots. Default is True.
    plot_kwargs (dict, optional): 
        Additional keyword arguments passed to the plotting function (e.g., `edge_cmap`, `node_size`).
    show_plots (bool, optional): 
        Whether to display the figures on screen. Default is True.
    axes (matplotlib.axes.Axes or array-like, optional): 
        Axes to use for plotting, if already created externally. If None, new axes will be generated.
    combined (bool, optional): 
        If True, all maps are shown in a single figure. Otherwise, one figure per map is created.
    save_figures (bool, optional): 
        Whether to save the plotted figures. Default is False.
    titles (list of str, optional): 
        Titles to use for each connectivity map. If None, titles will be generated automatically.
    n_rows (int, optional): 
        Number of rows to use when arranging subplots (if `combined=True`). Default is 1.
    figure_filenames (list of str, optional): 
        List of custom filenames for each individual figure (only used if `combined=False` 
        and `save_figures=True`).
    save_folder_name (str, optional): 
        Name of the output folder where saved figures will be stored. Default is "connectivity_maps".
    """

    connectivity_map = np.copy(connectivity_map)
    # Standardize shape
    if connectivity_map.ndim == 2:
        connectivity_map = connectivity_map[np.newaxis, np.newaxis, ...]
    elif connectivity_map.ndim == 3:
        connectivity_map = connectivity_map[np.newaxis, ...]
    elif connectivity_map.ndim != 4:
        raise ValueError("connectivity_map must be 2D, 3D or 4D")

    if isinstance(threshold, (float, int)):
        threshold = np.array([threshold] * connectivity_map.shape[1])

    if np.any(threshold > 1) or np.any(threshold < 0):
        raise ValueError("threshold must be between 0 and 1.")
    

    if component is None:
        component = 0
    conn_map = connectivity_map[component]
    n_modes = conn_map.shape[0]

    if match_color_scale:
        # Create a copy to avoid modifying original data
        conn_map_ = np.copy(conn_map)

        # Zero out the diagonal (for square matrices)
        if conn_map_.ndim == 3 and conn_map_.shape[1] == conn_map_.shape[2]:
            for i in range(conn_map_.shape[0]):
                np.fill_diagonal(conn_map_[i], 0)

        abs_max = np.nanmax(np.abs(conn_map_))

        if abs_max > 0 and not np.isnan(abs_max):
            plot_kwargs = plot_kwargs or {}
            plot_kwargs["edge_vmin"] = -abs_max
            plot_kwargs["edge_vmax"] = abs_max
        else:
            print("[Warning] conn_map contains only zeros or NaNs after removing diagonals.")

    # set titles
    if titles is None:
        titles = [None] * n_modes
    elif titles is True:
        titles = [f"State {i+1}" for i in range(n_modes)]
    elif isinstance(titles, str):
        titles = [f"{titles} {i+1}" for i in range(n_modes)]
    elif isinstance(titles, list):
        if len(titles) != n_modes:
            raise ValueError("Length of 'titles' must match number of modes.")

    if combined:
        # Automatically adjust saving behavior for combined plots
        save_figures, fig_format = update_save_flags(save_figures, combined, fig_format)
    if save_figures:
        if isinstance(figure_filenames, list):
            if len(figure_filenames) != n_modes:
                raise ValueError("Length of figure_filenames must match number of modes.")

    # Zero out diagonal
    for c in conn_map:
        np.fill_diagonal(c, 0)

    parcellation_coords = get_parcellation_centers(parcellation_file)
    default_plot_kwargs = {"node_size": 10, "node_color": "black"}
    axes = axes or [None] * conn_map.shape[0]
    output_files = []
    kwargs = __override_dict_defaults(default_plot_kwargs, plot_kwargs)
    PATH_OUTPUT, base_filename = __resolve_figure_directory(save_figures, filename, default_folder=save_folder_name)

    for i in trange(conn_map.shape[0], desc="Saving images"):

        # Only show colorbar if matrix isn't all zeros
        kwargs["colorbar"] = np.any(conn_map[i][~np.eye(conn_map[i].shape[-1], dtype=bool)] != 0)
        
        base = figure_filenames if isinstance(figure_filenames, str) else base_filename
        PATH_FIG = os.path.join(PATH_OUTPUT, __generate_filename(base, i, fig_format)) if save_figures else None
        #fig = plt.figure()
        display = plotting.plot_connectome(
            conn_map[i],
            parcellation_coords,
            edge_threshold=f"{threshold[i] * 100}%",
            output_file=None,  # Don't save here
            axes=axes[i],
            **kwargs,
        )
        if titles:
            # Manually add title using matplotlib
            fig = plt.gcf()
            fig.suptitle(titles[i], fontsize=20, color="black", y=0.95)

        if save_figures or combined:
            if PATH_FIG is None:
                # If we're only creating the combined image, still need to save a temporary one
                PATH_FIG = os.path.join(PATH_OUTPUT, __generate_filename(f"{base_filename}_combined_part", i, fig_format))
            fig.savefig(PATH_FIG, format=fig_format)
            if not show_plots:
                plt.close(fig)
            output_files.append(PATH_FIG)

    if combined:
        n_columns = -(n_modes // -n_rows)
        fig, axes_grid = plt.subplots(n_rows, n_columns, figsize=(n_columns * 5, n_rows * 5))
        for i, ax in enumerate(axes_grid.flatten()):
            ax.axis("off")
            if i < n_modes:
                ax.imshow(plt.imread(output_files[i]))
        filename = filename or "connectivity_combined"
        fig.tight_layout()
        fig.savefig(os.path.join(PATH_OUTPUT, f"{base_filename}_combined.{fig_format}"))
        if not show_plots:
            plt.close(fig)
        for image_path in output_files:
            os.remove(image_path)

    if return_fig:
        return fig
    else:
        plt.show()

def __resolve_figure_directory(save_figures, filename, default_folder="Figures"):
    return resolve_figure_directory(save_figures, filename, default_folder)      
def __generate_filename(base, index, extension):
    return generate_filename(base, index, extension)     
def __override_dict_defaults(default_dict, override_dict):
    return override_dict_defaults(default_dict, override_dict)   

