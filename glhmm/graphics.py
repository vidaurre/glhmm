#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic graphics - Gaussian Linear Hidden Markov Model
@author: Diego Vidaurre 2023
"""
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

import warnings
from matplotlib import cm, colors
from matplotlib.colors import LogNorm, LinearSegmentedColormap, to_rgba_array
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import utils
# import utils



def show_trans_prob_mat(hmm,only_active_states=False,show_diag=True,show_colorbar=True):
    """Displays the transition probability matrix of a given HMM.

    Parameters:
    -----------
    hmm: HMM object
        An instance of the HMM class containing the transition probability matrix to be visualized.
    only_active_states : bool, optional, default=False
        Whether to display only active states or all states in the matrix.
    show_diag : bool, optional, defatult=True
        Whether to display the diagonal elements of the matrix or not.
    show_colorbar : bool, optional, default=True
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
    palette : str, default = 'Oranges'
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


def show_temporal_statistic(Gamma,indices,statistic='FO',type_plot='barplot'):
    """Plots a statistic over time for a set of sessions.

    Parameters:
    -----------
    Gamma : array of shape (n_samples, n_states)
        The state timeseries probabilities.
    indices: numpy.ndarray of shape (n_sessions,)
        The session indices to plot.
    statistic: str, default='FO'
        The statistic to compute and plot. Can be 'FO', 'switching_rate' or 'FO_entropy'.
    type_plot: str, default='barplot'
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
    only_active_states: bool, optional, default=False
        If True, only the beta coefficients of active states are shown.
    recompute_states: bool, optional, default=False
        If True, the betas will be recomputed from the data and the state time courses
    X: numpy.ndarray, optional, default=None
        The timeseries of set of variables 1.
    Y: numpy.ndarray, optional, default=None
        The timeseries of set of variables 2.
    Gamma: numpy.ndarray, optional, default=None
        The state time courses
    show_average: bool, optional, default=None
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
    coolwarm_cmap = plt.get_cmap('coolwarm').reversed()
    coolwarm_cmap2 = plt.get_cmap('autumn')
    copper_cmap = plt.get_cmap('copper').reversed()
    # Define the colors for the colormap
    copper_color1 = to_rgba_array(copper_cmap(1))[0][:3]
    # Define the colors for the colormap
    red = (1,0,0)
    red2 = (66/255, 13/255, 9/255)
    orange =(1, 0.5, 0)
    red_color1 = to_rgba_array(coolwarm_cmap(0))[0][:3]
    # red_color2 = to_rgba_array(coolwarm_cmap(0.1))[0][:3]
    # red_color3 = to_rgba_array(coolwarm_cmap(0.20))[0][:3]
    # red_color4 = to_rgba_array(coolwarm_cmap(0.25))[0][:3]
    # warm_color1 = to_rgba_array(coolwarm_cmap2(0.4))[0][:3]
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
       cmap_list (numpy.ndarray): Original color array for the colormap.

    Returns:
    ----------  
        modified_cmap (numpy.ndarray): Modified colormap array.
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

def plot_p_value_matrix(pval, alpha = 0.05, normalize_vals=True, figsize=(9, 5), steps=11, title_text="Heatmap (p-values)", annot=True, cmap_type='default', cmap_reverse=True, xlabel="", ylabel="", xticklabels=None, none_diagonal = False, num_colors = 259):
    from matplotlib import cm, colors
    import seaborn as sb
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    """
    Plot a heatmap of p-values.

    Parameters
    ----------
    pval : numpy.ndarray
        The p-values data to be plotted.
    normalize_vals : bool, optional
        If True, the data range will be normalized from 0 to 1 (Default=False).
    figsize : tuple, optional
        Figure size in inches (width, height) (Default=(12, 7)).
    steps : int, optional
        Number of steps for x and y-axis ticks (Default= 11).
    title_text : str, optional
        Title text for the heatmap (Default= Heatmap (p-values)).
    annot : bool, optional
        If True, annotate each cell with the numeric value (Default= True).
    cmap : str, optional
        Colormap to use. Default is a custom colormap based on 'coolwarm'.
    xlabel : str, optional
        X-axis label. If not provided, default labels based on the method will be used.
    ylabel : str, optional
        Y-axis label. If not provided, default labels based on the method will be used.
    xticklabels : List[str], optional
        If not provided, labels will be numbers equal to shape of pval.shape[1].
        Else you can define your own labels, e.g., xticklabels=['sex', 'age'].
    none_diagonal : bool, optional
        If you want to turn the diagonal into NaN numbers (Default=False).

    Returns
    -------
    None
        Displays the heatmap plot.
    """
    if pval.ndim==0:
        pval = np.reshape(pval, (1, 1))
        
    fig, ax = plt.subplots(figsize=figsize)
    if len(pval.shape)==1:
        pval =np.expand_dims(pval,axis=0)
    if cmap_type=='default':
        if normalize_vals:
            color_array = np.logspace(-3, 0, num_colors).reshape(1, -1)

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
            color_array = np.logspace(-3, 0, num_colors).reshape(1, -1)
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
        norm = LogNorm(vmin=1e-3, vmax=1)

        heatmap = sb.heatmap(pval, ax=ax, cmap=cmap, annot=annot, fmt=".3f", cbar=False, norm=norm)
    else:
        heatmap = sb.heatmap(pval, ax=ax, cmap=cmap, annot=annot, fmt=".3f", cbar=False)

    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title_text, fontsize=14)
    # Set the x-axis ticks
    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)) + 0.5)
        ax.set_xticklabels(xticklabels, rotation="horizontal", fontsize=10)
    elif pval.shape[1]>1:
        ax.set_xticks(np.linspace(0, pval.shape[1]-1, steps).astype(int)+0.5)
        ax.set_xticklabels(np.linspace(1, pval.shape[1], steps).astype(int), rotation="horizontal", fontsize=10)
    else:
        ax.set_xticklabels([])
    # Set the y-axis ticks
    if pval.shape[0]>1:
        ax.set_yticks(np.linspace(0, pval.shape[0]-1, steps).astype(int)+0.5)
        ax.set_yticklabels(np.linspace(1, pval.shape[0], steps).astype(int), rotation="horizontal", fontsize=10)
    else:
        ax.set_yticklabels([])
    # Create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    
    if normalize_vals:   
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        colorbar = plt.colorbar(heatmap.get_children()[0], cax=cax, ticks=np.logspace(-3, 0, num_colors))
        colorbar.update_ticks()
        
         # Round the tick values to three decimal places
        rounded_ticks = [round(tick, 3) for tick in colorbar.get_ticks()]
        
        if figsize[-1] ==1:
            # Set colorbar ticks based on the same log scale
            tick_positions = [0, 0.001, 0.01, 0.05, 0.3, 1]
        else:
            # Set colorbar ticks based on the same log scale
            tick_positions = [0, 0.001, 0.01, 0.05, 0.1, 0.3, 1]
        tick_labels = [f'{tick:.3f}' if tick in tick_positions else '' for tick in rounded_ticks]
        unique_values_set = set()
        unique_values_array = ['' if value == '' or value in unique_values_set else (unique_values_set.add(value), value)[1] for value in tick_labels]

        indices_not_empty = [index for index, value in enumerate(unique_values_array) if value != '']

        colorbar.set_ticklabels(unique_values_array)
        colorbar.ax.tick_params(axis='y')

        for idx, tick_line in enumerate(colorbar.ax.yaxis.get_ticklines()):
            if idx not in indices_not_empty:
                tick_line.set_visible(False)
            
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        # Create a custom colorbar
        colorbar = plt.colorbar(heatmap.get_children()[0], cax=cax)
        # Set the ticks to range from the bottom to the top of the colorbar
        # Get the minimum and maximum values from your data
        min_value = np.nanmin(pval)
        max_value = np.nanmax(pval)

        # Set ticks with at least 5 values evenly spaced between min and max
        colorbar.set_ticks(np.linspace(min_value, max_value, 5).round(2))
        #colorbar.set_ticks([0, 0.25, 0.5, 1])  # Adjust ticks as needed
        

    # Show the plot
    plt.show()
    
def plot_correlation_matrix(corr_vals, performed_tests, normalize_vals=False, figsize=(9, 5), steps=11, title_text="Heatmap (p-values)", annot=True, cmap_type='default', cmap_reverse=True, xlabel="", ylabel="", xticklabels=None, none_diagonal = False, num_colors = 256):
    from matplotlib import cm, colors
    import seaborn as sb
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    """
    Plot a heatmap of p-values.

    Parameters
    ----------
    corr_vals : numpy.ndarray
        base statistics of corelation coefficients.
    performed_tests : dict
        Holds information about the different test statistics that has been applied
    normalize_vals : bool, optional
        If True, the data range will be normalized from 0 to 1 (Default=False).
    figsize : tuple, optional
        Figure size in inches (width, height) (Default=(12, 7)).
    steps : int, optional
        Number of steps for x and y-axis ticks (Default= 11).
    title_text : str, optional
        Title text for the heatmap (Default= Heatmap (p-values)).
    annot : bool, optional
        If True, annotate each cell with the numeric value (Default= True).
    cmap : str, optional
        Colormap to use. Default is a custom colormap based on 'coolwarm'.
    xlabel : str, optional
        X-axis label. If not provided, default labels based on the method will be used.
    ylabel : str, optional
        Y-axis label. If not provided, default labels based on the method will be used.
    xticklabels : List[str], optional
        If not provided, labels will be numbers equal to shape of corr_vals.shape[1].
        Else you can define your own labels, e.g., xticklabels=['sex', 'age'].
    none_diagonal : bool, optional
        If you want to turn the diagonal into NaN numbers (Default=False).

    Returns
    -------
    None
        Displays the heatmap plot.
    """
    if performed_tests["t_test_cols"]!=[] or performed_tests["f_test_cols"]!=[]:
        raise ValueError("Cannot plot the base statistics for the correlation coefficients because different test statistics have been used.")
    
    if corr_vals.ndim==0:
        corr_vals = np.reshape(corr_vals, (1, 1))
        
    fig, ax = plt.subplots(figsize=figsize)
    if len(corr_vals.shape)==1:
        corr_vals =np.expand_dims(corr_vals,axis=0)

    if cmap_type=='default':
        # seismic_cmap = cm.seismic.reversed()
        coolwarm_cmap = cm.coolwarm.reversed()
        
        #seismic_cmap = cm.RdBu.reversed()
        # Generate an array of values representing the colormap
        color_array = np.linspace(0, 1, num_colors).reshape(1, -1)
        cmap_list = coolwarm_cmap(color_array)[0]
        cmap = colors.ListedColormap(cmap_list)
    else:
        # Get the colormap dynamically based on the input string
        cmap = getattr(cm, cmap_type, None)
        if cmap_reverse:
            cmap =cmap.reversed()

    if normalize_vals:
        # Normalize the data range from -1 to 1
        norm = plt.Normalize(vmin=-1, vmax=1)
        heatmap = sb.heatmap(corr_vals, ax=ax, cmap=cmap, annot=annot, fmt=".3f", cbar=False, norm=norm)
    else:
        heatmap = sb.heatmap(corr_vals, ax=ax, cmap=cmap, annot=annot, fmt=".3f", cbar=False)
    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title_text, fontsize=14)
    # Set the x-axis ticks
    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)) + 0.5)
        ax.set_xticklabels(xticklabels, rotation="horizontal", fontsize=10)
    elif corr_vals.shape[1]>1:
        ax.set_xticks(np.linspace(0, corr_vals.shape[1]-1, steps).astype(int)+0.5)
        ax.set_xticklabels(np.linspace(1, corr_vals.shape[1], steps).astype(int), rotation="horizontal", fontsize=10)
    else:
        ax.set_xticklabels([])
    # Set the y-axis ticks
    if corr_vals.shape[0]>1:
        ax.set_yticks(np.linspace(0, corr_vals.shape[0]-1, steps).astype(int)+0.5)
        ax.set_yticklabels(np.linspace(1, corr_vals.shape[0], steps).astype(int), rotation="horizontal", fontsize=10)
    else:
        ax.set_yticklabels([])
    # Create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # Create a custom colorbar
    colorbar = plt.colorbar(heatmap.get_children()[0], cax=cax)
    # Set the ticks to range from the bottom to the top of the colorbar
    # Get the minimum and maximum values from your data
    min_value = np.nanmin(corr_vals)
    max_value = np.nanmax(corr_vals)

    if normalize_vals:
        colorbar.set_ticks(np.linspace(-1, 1, 9).round(2))
    else:
        # Set ticks with at least 5 values evenly spaced between min and max
        colorbar.set_ticks(np.linspace(min_value, max_value, 7).round(2))

        
    # Show the plot
    plt.show()

  
def plot_permutation_distribution(test_statistic, title_text="Permutation Distribution",xlabel="Test Statistic Values",ylabel="Density"):
    """
    Plot the histogram of the permutation with the observed statistic marked.

    Parameters
    ----------
    test_statistic : numpy.ndarray
        An array containing the permutation values.
    title_text : str, optional
        Title text of the plot (Default="Permutation Distribution").
    xlabel : str, optional
        Text of the xlabel (Default="Test Statistic Values").
    ylabel : str, optional
        Text of the ylabel (Default="Density").

    Returns
    -------
    None
        Displays the histogram plot.
    """
    plt.figure()
    sb.histplot(test_statistic, kde=True)
    plt.axvline(x=test_statistic[0], color='red', linestyle='--', label='Observed Statistic')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title_text, fontsize=14)
        
    plt.legend()
    plt.show()



def plot_scatter_with_labels(p_values, alpha=0.05, title_text="", xlabel=None, ylabel=None, xlim_start=0.9, ylim_start=0):
    """
    Create a scatter plot to visualize p-values with labels indicating significant points.

    Parameters
    ----------
    p_values : numpy.ndarray
        An array of p-values. Can be a 1D array or a 2D array with shape (1, 5).
    alpha : float, optional
        Threshold for significance (Default=0.05).
    title_text : str, optional
        The title text for the plot (Default="").
    xlabel : str, optional
        The label for the x-axis (Default=None).
    ylabel : str, optional
        The label for the y-axis (Default=None).
    xlim_start : float, optional
        Start position of x-axis limits (Default=-5).
    ylim_start : float, optional
        Start position of y-axis limits (Default=-0.1).

    Returns
    -------
    None

    Note
    ----
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
    plt.show()
    
    
def plot_vpath(vpath, signal =[], xlabel = "Time Steps", figsize=(7, 4), ylabel = "", yticks=None,line_width=2, label_signal="Signal"):
    # Assuming vpath is your data matrix
    num_states = vpath.shape[1]

    # Create a Seaborn color palette
    colors = sb.color_palette("Set3", n_colors=num_states)

    # Plot the stack plot using Seaborn
    fig, axes = plt.subplots(figsize=figsize)  # Adjust the figure size for better readability
    axes.stackplot(np.arange(vpath.shape[0]), vpath.T, colors=colors, labels=[f'State {i + 1}' for i in range(num_states)])

    # Set labels and legend to the right of the figure
    axes.set_xlabel(xlabel, fontsize=14)
    axes.set_ylabel(ylabel, fontsize=14)
    axes.legend(title='States', loc='upper left', bbox_to_anchor=(1, 1))  # Adjusted legend position

    if yticks:
        scaled_values = [int(val * len(np.unique(signal))) for val in np.unique(signal)]
        # Set y-ticks with formatted integers
        axes.set_yticks(np.unique(signal), scaled_values)
    else:
        # Remove x-axis tick labels
        axes.set_yticks([])

    # Remove the frame around the plot
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    # Add a plot of the signal (replace this with your actual signal data)
    # com_signal = np.sin(np.linspace(0, 10, vpath.shape[0])) + 2
    if signal is not None:
        axes.plot(signal, color='black', label=label_signal, linewidth=line_width)
    axes.legend(loc='upper left', bbox_to_anchor=(1, 0.8))  # Adjusted legend position


    # Increase tick label font size
    axes.tick_params(axis='both', labelsize=12)
    plt.tight_layout() 
    # Show the plot
    plt.show()
    
def plot_average_probability(Gamma_reconstruct, title='Average probability for each state', fontsize=16, figsize=(7, 5), vertical_lines=None, line_colors=None, highlight_boxes=False):

    """
    Plots the average probability for each state over time.

    Parameters
    ----------
    Gamma_reconstruct : numpy.ndarray
        3D array representing reconstructed gamma values.
        Shape: (num_timepoints, num_trials, num_states)
    title : str, optional
        Title for the plot (Default='Average probability for each state').
    fontsize : int, optional
        Font size for labels and title (Default=16).
    figsize : tuple, optional
        Figure size (width, height) in inches (Default=(8, 6)).
    vertical_lines : list of tuples, optional
        List of pairs specifying indices for vertical lines (Default=None).
    line_colors : list of str or bool, optional
        List of colors for each pair of vertical lines. If True, generates random colors
        (unless a list is provided) (Default=None).
    highlight_boxes : bool, optional
        Whether to include highlighted boxes for each pair of vertical lines (Default=False).
    
    Returns
    -------
    None
    """

    # Initialize an array for average gamma values
    Gamma_avg = np.zeros((Gamma_reconstruct.shape[0], Gamma_reconstruct.shape[-1]))

    # Calculate and store average gamma values
    for i in range(Gamma_reconstruct.shape[0]):
        filtered_values = Gamma_reconstruct[i, :, :]
        Gamma_avg[i, :] = np.mean(filtered_values, axis=0).round(3)

    # Set figure size
    fig, axes = plt.subplots(1, figsize=figsize)

    # Plot each line with a label
    for state in range(Gamma_reconstruct.shape[-1]):
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

    # Show the plot
    plt.show()
    

def plot_condition_difference(Gamma_reconstruct, R_trials, title='Average Probability and Difference', fontsize=16, figsize=(9, 2), vertical_lines=None, line_colors=None, highlight_boxes=False):
    """
    Plots the average probability for each state over time for two conditions and their difference.

    Parameters:
    -----------
    Gamma_reconstruct : numpy.ndarray
        3D array representing reconstructed gamma values.
        Shape: (num_timepoints, num_trials, num_states)
    R_trials : numpy.ndarray
        1D array representing the condition for each trial.
        Should have the same length as the second dimension of Gamma_reconstruct.
    title : str, optional
        Title for the plot (Default='Average Probability and Difference').
    fontsize : int, optional
        Font size for labels and title (Default=16).
    figsize : tuple, optional
        Figure size (width, height) in inches (Default=(9, 2)).
    vertical_lines : list of tuples, optional
        List of pairs specifying indices for vertical lines (Default=None).
    line_colors : list of str or bool, optional
        List of colors for each pair of vertical lines. If True, generates random colors
        (unless a list is provided) (Default= None).
    highlight_boxes : bool, optional
        Whether to include highlighted boxes for each pair of vertical lines (Default=False).

    Example usage:
    --------------
    plot_condition_difference(Gamma_reconstruct, R_trials, vertical_lines=[(10, 100)], highlight_boxes=True)
    """

    filt_val = np.zeros((2, Gamma_reconstruct.shape[0], Gamma_reconstruct.shape[2]))

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot for each condition
    for condition in range(2):
        for i in range(Gamma_reconstruct.shape[0]):
            filtered_values = Gamma_reconstruct[i, (R_trials == condition + 1), :]
            filt_val[condition, i, :] = np.mean(filtered_values, axis=0).round(3)
        axes[condition].plot(filt_val[condition, :, :])
        axes[condition].set_title(f"Condition {condition + 1}")
        axes[condition].set_xticks(np.linspace(0, Gamma_reconstruct.shape[0] - 1, 5).astype(int))
        axes[condition].set_yticks(np.linspace(axes[condition].get_ylim()[0], axes[condition].get_ylim()[1], 5).round(2))

    # Find the element-wise difference
    difference = filt_val[0, :, :] - filt_val[1, :, :]

    # Plot the difference
    axes[2].plot(difference)
    axes[2].set_title("Difference")
    axes[2].set_xticks(np.linspace(0, Gamma_reconstruct.shape[0] - 1, 5).astype(int))
    axes[2].set_yticks(np.linspace(axes[2].get_ylim()[0], axes[2].get_ylim()[1], 5).round(2))

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
        ax.set_xlabel('Timepoints', fontsize=12)
        ax.set_ylabel('Average probability', fontsize=12)

    # Label each state on the right for the last figure (axes[2])
    state_labels = [f"State {state+1}" for state in range(Gamma_reconstruct.shape[2])]
    axes[2].legend(state_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

    fig.suptitle(title, fontsize=fontsize)

    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    
def plot_p_values_over_time(pval, figsize=(8, 4), total_time_seconds=None, xlabel="Time points", 
                            ylabel="P-values (Log Scale)",title_text="P-values over time", xlim_start=0, 
                            tick_positions=[0, 0.001, 0.01, 0.05, 0.1, 0.3, 1], num_colors=259, 
                            alpha=0.05,plot_style = "line", linewidth=2.5):
    """
    Plot a scatter plot of p-values over time with a log-scale y-axis and a colorbar.

    Parameters:
    -----------
    pval : numpy.ndarray
        The p-values data to be plotted.
    total_time_seconds (float, optional): 
        Total time duration in seconds. If provided, time points will be scaled accordingly.
    xlabel (str, optional): 
        Label for the x-axis. Default is 'Time points'.
    ylabel (str, optional): 
        Label for the y-axis. Default is 'Y-axis (log scale)'.
    title_text (str, optional): 
        Title for the plot. Default is 'P-values over time'.
    tick_positions (list, optional): 
        Specific values to mark on the y-axis. Default is [0, 0.001, 0.01, 0.05, 0.1, 0.3, 1].
    num_colors (int, optional): 
        Resolution for the color bar. Default is 259.
    alpha (float, optional): 
        Alpha value is the threshold we set for the p-values when doing visualization. Default is 0.05.
    plot_style (str, optional): 
        Style of plot. Default is 'line'.    
    Returns:
    -----------
    None (displays the plot).
    """
    if pval.ndim != 1:
        # Raise an exception and stop function execution
        raise ValueError("To use the function 'plot_p_values_over_time', the variable for p-values must be one-dimensional.")

    # Generate time points based on total_time_seconds
    if total_time_seconds:
        time_points = np.linspace(0, total_time_seconds, len(pval))
    else:
        time_points = np.arange(len(pval))

    # Convert to log scale
    color_array = np.logspace(-3, 0, num_colors).reshape(1, -1)

    if alpha is None:
        # Create custom colormap
        coolwarm_cmap = custom_colormap()
        # Create a new colormap with the modified color_array
        cmap_list = coolwarm_cmap(color_array)[0]
        modified_cmap = interpolate_colormap(cmap_list)
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

        # shift values a bit
        # cmap_red[:,:3] -= 0.15
        # # Set values above 1 to 1
        # # overwrite the values below alpha
        # cmap_red[cmap_red < 0] = 0

        # overwrite the values below alpha
        cmap_list[:num_elements_red,:]=cmap_red
        cmap_list[num_elements_red:,:]=cmap_blue
        cmap = LinearSegmentedColormap.from_list('custom_colormap', cmap_list)
            
    # Create the line plot with varying color based on p-values
    _, ax = plt.subplots(figsize=figsize)

    # Normalize the data to [0, 1] for the colormap with logarithmic scale
    norm = LogNorm(vmin=1e-3, vmax=1)

    if plot_style == "line":
        if alpha !=None:
            # Plot the line segments with varying colors
            for i in range(len(time_points)-1):
                if pval[i+1]>alpha:
                    color = cmap(norm(pval[i+1]))
                else:
                    color = cmap(norm(pval[i]))
                ax.plot([time_points[i], time_points[i+1]], [pval[i], pval[i+1]], color=color, linewidth=linewidth)
        else:
            for i in range(len(time_points)-1):
                if pval[i+1]>0.05:
                    color = cmap(norm(pval[i+1]))
                else:
                    color = cmap(norm(pval[i]))
                ax.plot([time_points[i], time_points[i+1]], [pval[i], pval[i+1]], color=color, linewidth=linewidth)
    elif plot_style=="scatter":
        ax.scatter(time_points, pval, c=pval, cmap=cmap, norm=LogNorm(vmin=1e-3, vmax=1))
    elif plot_style=="scatter_line":
        ax.scatter(time_points, pval, c=pval, cmap=cmap, norm=LogNorm(vmin=1e-3, vmax=1))    
            # Draw lines between points
        ax.plot(time_points, pval, color='black', linestyle='-', linewidth=1)
    # Add labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title_text, fontsize=14)
    
    # Set axis limits to focus on the relevant data range
    ax.set_xlim(xlim_start, len(pval) + 1)
    ax.set_ylim([0.0008, 1.5])
    # Set y-axis to log scale
    ax.set_yscale('log')
    # Mark specific values on the y-axis
    plt.yticks([0.001, 0.01, 0.05, 0.1, 0.3, 1], ['0.001', '0.01', '0.05', '0.1', '0.3', '1'])
    # Add a colorbar to show the correspondence between colors and p-values
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=np.logspace(-3, 0, num_colors), format="%1.0e")
    colorbar.update_ticks()

    # Round the tick values to three decimal places
    rounded_ticks = [round(tick, 3) for tick in colorbar.get_ticks()]
    tick_labels = [f'{tick:.3f}' if tick in tick_positions else '' for tick in rounded_ticks]
    unique_values_set = set()
    unique_values_array = ['' if value == '' or value in unique_values_set else (unique_values_set.add(value), value)[1] for value in tick_labels]

    indices_not_empty = [index for index, value in enumerate(unique_values_array) if value != '']

    colorbar.set_ticklabels(unique_values_array)
    colorbar.ax.tick_params(axis='y')

    for idx, tick_line in enumerate(colorbar.ax.yaxis.get_ticklines()):
        if idx not in indices_not_empty:
            tick_line.set_visible(False)

    plt.show()
    
    
def plot_p_values_bar(pval,variables=[],  figsize=(9, 4), num_colors=256, xlabel="",
                        ylabel="P-values (Log Scale)", title_text="Bar Plot",
                        tick_positions=[0, 0.001, 0.01, 0.05, 0.1, 0.3, 1], top_adjustment=0.9, alpha = 0.05, pad_title=20):
    """
    Visualize a bar plot with LogNorm and a colorbar.

    Parameters:
    - variables (list): List of categories or variables.
    - pval (array-like): Array of p-values.
    - figsize (tuple, optional): Figure size, default is (9, 4).
    - num_colors (int, optional): Number of colors in the colormap, default is 256.
    - xlabel (str, optional): X-axis label, default is "Categories".
    - ylabel (str, optional): Y-axis label, default is "Values (log scale)".
    - title_text (str, optional): Plot title, default is "Bar Plot with LogNorm".
    - tick_positions (list, optional): Positions of ticks on the colorbar, default is [0, 0.001, 0.01, 0.05, 0.1, 0.3, 1].
    top_adjustment (float, optional): Adjustment for extra space between title and plot, default is 0.9.

    Returns:
    None
    """
    # Choose a colormap
    coolwarm_cmap = custom_colormap()

    # Convert to log scale
    color_array = np.logspace(-3, 0, num_colors).reshape(1, -1)

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

        # shift values a bit
        # cmap_red[:,:3] -= 0.15
        # # Set values above 1 to 1
        # # overwrite the values below alpha
        # cmap_red[cmap_red < 0] = 0

        # overwrite the values below alpha
        cmap_list[:num_elements_red,:]=cmap_red
        cmap_list[num_elements_red:,:]=cmap_blue

    
    # Create a LinearSegmentedColormap
    colormap = LinearSegmentedColormap.from_list('custom_colormap', cmap_list)

    # Plot the bars with LogNorm
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(pval, (float, np.ndarray)) and np.size(pval) == 1:
    # It's a scalar, create a list with a single element
        variables = [f"Var 1"] if variables==[] else variables
    else:
        # It's an iterable, use len()
        variables =[f"Var {i+1}" for i in np.arange(len(pval))] if variables==[] else variables


    bars = plt.bar(variables, pval, color=colormap(LogNorm(vmin=1e-3, vmax=1)(pval)))
    # Remove the legend
    #plt.legend().set_visible(False)

    # Add data labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 3), ha='center', va='bottom', color='black', fontweight='bold')


    # Set y-axis to log scale
    ax.set_yscale('log')

    # Customize plot
    plt.yscale('log')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title_text, fontsize=14, pad=pad_title)

    # Mark specific values on the y-axis
    plt.yticks([0.001, 0.01, 0.05, 0.1, 0.3, 1], ['0.001', '0.01', '0.05', '0.1', '0.3', '1'])

    # Add a colorbar to show the correspondence between colors and p-values
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = plt.colorbar(plt.cm.ScalarMappable(cmap=colormap, norm=LogNorm(vmin=1e-3, vmax=1)), cax=cax, ticks=np.logspace(-3, 0, num_colors), format="%1.0e")
    colorbar.update_ticks()

    # Round the tick values to three decimal places
    rounded_ticks = [round(tick, 3) for tick in colorbar.get_ticks()]
    tick_labels = [f'{tick:.3f}' if tick in tick_positions else '' for tick in rounded_ticks]
    unique_values_set = set()
    unique_values_array = ['' if value == '' or value in unique_values_set else (unique_values_set.add(value), value)[1] for value in tick_labels]

    indices_not_empty = [index for index, value in enumerate(unique_values_array) if value != '']

    colorbar.set_ticklabels(unique_values_array)
    colorbar.ax.tick_params(axis='y')

    for idx, tick_line in enumerate(colorbar.ax.yaxis.get_ticklines()):
        if idx not in indices_not_empty:
            tick_line.set_visible(False)

    # Add extra space between title and plot
    plt.subplots_adjust(top=top_adjustment)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()

    plt.show()
