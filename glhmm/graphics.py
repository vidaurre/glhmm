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

def plot_p_value_matrix(pval, alpha = 0.05, normalize_vals=True, figsize=(9, 5), 
                         title_text="Heatmap (p-values)", annot=False, 
                        cmap_type='default', cmap_reverse=True, xlabel="", ylabel="", 
                        xticklabels=None, x_tick_min=None, x_tick_max=None, num_x_ticks=5, none_diagonal = False, num_colors = 259, xlabel_rotation=0):
    from matplotlib import cm, colors
    import seaborn as sb
    from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    annot (bool, optional), default=True:
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
    none_diagonal (bool, optional), default=False:
        If you want to turn the diagonal into NaN numbers.
    num_colors (numpy.ndarray), default=259:
        Define the number of different shades of color.
    xlabel_rotation (numpy-mdarray), default=0
        The degree of rotation for the labels in the x-axis
    """
    if pval.ndim==0:
        pval = np.reshape(pval, (1, 1))
    if xlabel_rotation==45:
        ha ="right"
    else:
        ha = "center"    


    fig, axes = plt.subplots(figsize=figsize)
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

        heatmap = sb.heatmap(pval, ax=axes, cmap=cmap, annot=annot, fmt=".3f", cbar=False, norm=norm)
    else:
        heatmap = sb.heatmap(pval, ax=axes, cmap=cmap, annot=annot, fmt=".3f", cbar=False)

    # Add labels and title
    axes.set_xlabel(xlabel, fontsize=12)
    axes.set_ylabel(ylabel, fontsize=12)
    axes.set_title(title_text, fontsize=14)
    # Number of x-tick steps
    steps=len(pval)
    
    # define x_ticks
    x_tick_positions = np.linspace(0, len(pval), num_x_ticks).astype(int)

    # Generate x-tick labels based on user input or default to time points
    if x_tick_min is not None and x_tick_max is not None:
        x_tick_labels = np.linspace(x_tick_min, x_tick_max, num_x_ticks).round(2)
    else:
        x_tick_labels = x_tick_positions


    # Set the x-axis ticks
    if xticklabels is not None:
        axes.set_xticks(np.arange(len(xticklabels)) + 0.5)
        axes.set_xticklabels(xticklabels, rotation=xlabel_rotation, fontsize=10, ha=ha)
    elif pval.shape[1]>1:
        axes.set_xticks(np.linspace(0, pval.shape[1]-1, steps).astype(int)+0.5)
        axes.set_xticklabels(np.linspace(1, pval.shape[1], steps).astype(int), rotation=xlabel_rotation, fontsize=10, ha=ha)
    else:
        axes.set_xticklabels([])
    # Set the y-axis ticks
    if pval.shape[0]>1:
        axes.set_yticks(np.linspace(0, pval.shape[0]-1, steps).astype(int)+0.5)
        axes.set_yticklabels(np.linspace(1, pval.shape[0], steps).astype(int), rotation=xlabel_rotation, fontsize=10, ha=ha)
    else:
        axes.set_yticklabels([])
    # Create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    
    if normalize_vals:   
        divider = make_axes_locatable(axes)
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
        divider = make_axes_locatable(axes)
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
    
def plot_correlation_matrix(corr_vals, performed_tests, normalize_vals=False, 
                            figsize=(9, 5), title_text="Correlation Coefficients Heatmap", 
                            annot=False, cmap_type='default', cmap_reverse=True, xlabel="", ylabel="", 
                            xticklabels=None,xlabel_rotation=45, none_diagonal = False, num_colors = 256):
    from matplotlib import cm, colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    """
    Plot a heatmap of correlation coefficients.

    Parameters:
    -----------
    corr_vals (numpy.ndarray)
        Base statistics of correlation coefficients.
    performed_tests (dict)
        Holds information about the different test statistics that have been applied.
    normalize_vals (bool, optional)
        If True, the data range will be normalized from 0 to 1 (default is False).
    figsize (tuple, optional), default=(9, 5):
        Figure size in inches (width, height).
    title_text (str, optional), default="Correlation Coefficients Heatmap"
        Title text for the heatmap.
    annot (bool, optional), default=True:
        If True, annotate each cell with the numeric value.
    cmap_type (str, optional), default='default':
        Colormap to use.
    cmap_reverse (bool, optional), default=True:
        If True, reverse the colormap.
    xlabel (str, optional), default='':
        X-axis label. If not provided, default labels based on the method will be used.
    ylabel (str, optional), default='':
        Y-axis label. If not provided, default labels based on the method will be used.
    xticklabels (List[str], optional), default=None:
        If not provided, labels will be numbers equal to the shape of corr_vals.shape[1].
        Else, you can define your own labels, e.g., xticklabels=['sex', 'age'].
    none_diagonal (bool, optional), default=False:
        If True, turn the diagonal into NaN numbers.
    num_colors (int, optional), default=256:
        Number of colors to use in the colormap.
    """
    if performed_tests["t_test_cols"]!=[] or performed_tests["f_test_cols"]!=[]:
        raise ValueError("Cannot plot the base statistics for the correlation coefficients because different test statistics have been used.")
    
    if corr_vals.ndim==0:
        corr_vals = np.reshape(corr_vals, (1, 1))
    if xlabel_rotation==45:
        ha ="right"
    else:
        ha = "center"    
    # Number of x-tick steps
    steps=len(corr_vals)
    fig, axes = plt.subplots(figsize=figsize)
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
        heatmap = sb.heatmap(corr_vals, ax=axes, cmap=cmap, annot=annot, fmt=".3f", cbar=False, norm=norm)
    else:
        heatmap = sb.heatmap(corr_vals, ax=axes, cmap=cmap, annot=annot, fmt=".3f", cbar=False)
    # Add labels and title
    axes.set_xlabel(xlabel, fontsize=12)
    axes.set_ylabel(ylabel, fontsize=12)
    axes.set_title(title_text, fontsize=14)
    # Set the x-axis ticks
    if xticklabels is not None:
        axes.set_xticks(np.arange(len(xticklabels)) + 0.5)
        axes.set_xticklabels(xticklabels, rotation=xlabel_rotation, ha=ha,fontsize=10)
    elif corr_vals.shape[1]>1:
        axes.set_xticks(np.linspace(0, corr_vals.shape[1]-1, steps).astype(int)+0.5)
        axes.set_xticklabels(np.linspace(1, corr_vals.shape[1], steps).astype(int), rotation=xlabel_rotation, ha=ha, fontsize=10)
    else:
        axes.set_xticklabels([])
    # Set the y-axis ticks
    if corr_vals.shape[0]>1:
        axes.set_yticks(np.linspace(0, corr_vals.shape[0]-1, steps).astype(int)+0.5)
        axes.set_yticklabels(np.linspace(1, corr_vals.shape[0], steps).astype(int), rotation=xlabel_rotation, ha=ha, fontsize=10)
    else:
        axes.set_yticklabels([])
    # Create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.01)
    # Create a custom colorbar
    colorbar = plt.colorbar(heatmap.get_children()[0], cax=cax)
    # Set the ticks to range from the bottom to the top of the colorbar
    # Get the minimum and maximum values from your data
    min_value = np.nanmin(corr_vals).round(2)
    max_value = np.floor(np.nanmax(corr_vals) * 100) / 100 

    if normalize_vals:
        colorbar.set_ticks(np.linspace(-1, 1, 7).round(2))
    else:
        # Set ticks with at least 5 values evenly spaced between min and max
        colorbar.set_ticks(np.linspace(min_value, max_value, 7).round(2))
        

        
    # Show the plot
    plt.show()

  
def plot_permutation_distribution(test_statistic, title_text="Permutation Distribution",xlabel="Test Statistic Values",ylabel="Density"):
    """
    Plot the histogram of the permutation with the observed statistic marked.

    Parameters:
    -----------
    test_statistic (numpy.ndarray)
        An array containing the permutation values.
    title_text (str, optional), default="Permutation Distribution":
        Title text of the plot.
    xlabel (str, optional), default="Test Statistic Values"
        Text of the xlabel.
    ylabel (str, optional), default="Density"
        Text of the ylabel.
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
    plt.show()
    
import seaborn as sns

def plot_vpath(viterbi_path, signal=None, idx_data=None, figsize=(7, 4), fontsize_labels=13, fontsize_title=16, yticks=None, time_conversion_rate=None, xlabel="Timepoints", ylabel="", title="Viterbi Path", signal_label="Signal", show_legend=True, vertical_linewidth=1.5):
    """
    Plot Viterbi path with optional signal overlay.

    Parameters:
    -----------
    viterbi_path
        The Viterbi path data matrix.
    signal : array-like, optional
        Signal data to overlay on the plot. Default is None.
    idx_data : array-like, optional
        Array representing time intervals. Default is None.
    figsize : tuple, optional
        Figure size. Default is (7, 4).
    fontsize_labels : int, optional
        Font size for axis labels. Default is 13.
    fontsize_title : int, optional
        Font size for plot title. Default is 16.
    yticks : bool, optional
        Whether to show y-axis ticks. Default is None.
    time_conversion_rate : float, optional
        Conversion rate from time steps to seconds. Default is None.
    xlabel : str, optional
        Label for the x-axis. Default is "Timepoints".
    ylabel : str, optional
        Label for the y-axis. Default is "".
    title : str, optional
        Title for the plot. Default is "Viterbi Path".
    signal_label : str, optional
        Label for the signal plot. Default is "Signal".
    show_legend : bool, optional
        Whether to show the legend. Default is True.
    vertical_linewidth : float, optional
        Line width for vertical gray lines. Default is 1.5.
    """
    num_states = viterbi_path.shape[1]
    colors = sns.color_palette("Set3", n_colors=num_states)
    if num_states > len(colors):
        extra_colors = sns.color_palette("husl", n_colors=num_states - len(colors))
        colors.extend(extra_colors)

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
        if time_conversion_rate is not None:
            time_seconds = np.arange(len(signal)) / time_conversion_rate
            axes.plot(time_seconds, signal, color='black', label=signal_label)
            axes.set_xlabel(xlabel, fontsize=fontsize_labels)
        else:
            axes.plot(signal, color='black', label=signal_label)

    # Draw vertical gray lines for T_t intervals
    if idx_data is not None:
        for idx in idx_data[:-1, 1]:
            axes.axvline(x=idx, color='gray', linestyle='--', linewidth=vertical_linewidth)

    # Show legend
    if show_legend:
        axes.legend(title='States', loc='upper left', bbox_to_anchor=(1, 1))

    if yticks and signal is not None:
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

    # Adjust tick label font size
    axes.tick_params(axis='both', labelsize=fontsize_labels)

    plt.tight_layout() 
    plt.show()

    
def plot_average_probability(Gamma_data, title='Average probability for each state', fontsize=16, figsize=(7, 5), vertical_lines=None, line_colors=None, highlight_boxes=False):

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

    # Show the plot
    plt.show()

def plot_FO(FO, figsize=(8, 4), fontsize_labels=13, fontsize_title=16, width=0.8,xlabel='Subject',ylabel='Fractional occupancy',title='State Fractional Occupancies', show_legend=True, num_ticks=10):
    """
    Plot fractional occupancies for different states.

    Parameters:
    -----------
    FO (numpy.ndarray):
        Fractional occupancy data matrix.
    figsize (tuple, optional), default=(8,4):
        Figure size.
    fontsize_labels (int, optional), default=13:
        Font size for axes labels.
    fontsize_title (int, optional), default=16:
        Font size for plot title.
    width (float, optional), default=0.5:
        Width of the bars.
    xlabel (str, optional), default='Subject':
        Label for the x-axesis.
    ylabel (str, optional), default='Fractional occupancy':
        Label for the y-axesis.
    title (str, optional), default='State Fractional Occupancies':
        Title for the plot.
    show_legend (bool, optional), default=True:
        Whether to show the legend.
    """
    fig, axes = plt.subplots(figsize=figsize)
    bottom = np.zeros(FO.shape[0])
    sessions = np.arange(1, FO.shape[0] + 1)
    num_states = FO.shape[1]
    colors = sns.color_palette("Set3", n_colors=num_states)
    if num_states > len(colors):
        extra_colors = sns.color_palette("husl", n_colors=num_states - len(colors))
        colors.extend(extra_colors)
        
    for k in range(num_states):
        p = axes.bar(sessions, FO[:, k], bottom=bottom, color=colors[k], width=width)
        bottom += FO[:, k]
    
    axes.set_xticks(sessions)
    axes.set_xlabel(xlabel, fontsize=fontsize_labels)
    axes.set_ylabel(ylabel, fontsize=fontsize_labels)
    axes.set_title(title, fontsize=fontsize_title)
    
    ticks = np.linspace(1, FO.shape[0], FO.shape[0]).astype(int)
    # If there are more than 10 states then make a steps of 5
    if len(ticks)>10:
        n_ticks = num_ticks
    else:
        n_ticks = len(ticks)
    axes.set_xticks(np.linspace(1, FO.shape[0], n_ticks).astype(int))
    axes.set_yticks(np.linspace(0, 1, 5))
    
    # Remove the frame around the plot
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    # Adjust tick label font size
    axes.tick_params(axis='both', labelsize=fontsize_labels)

    if show_legend:
        legend = axes.legend(['State {}'.format(i+1) for i in range(FO.shape[1])], fontsize=fontsize_labels, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout() 
    plt.show()


def plot_switching_rates(SR, figsize=(8, 4), fontsize_labels=13, fontsize_title=16, width=0.18, xlabel='Subject', ylabel='Switching Rate', title='State Switching Rates', show_legend=True, num_ticks=10):
    """
    Plot switching rates for different states.

    Parameters:
    -----------
    SR (numpy.ndarray):
        Switching rate data matrix.
    figsize (tuple, optional), default=(8, 4):
        Figure size.
    fontsize_labels (int, optional), default=13:
        Font size for axes labels.
    fontsize_title (int, optional), default=16:
        Font size for plot title.
    width (float, optional), default=0.18:
        Width of the bars.
    xlabel (str, optional), default='Subject':
        Label for the x-axesis.
    ylabel (str, optional), default='Switching Rate':
        Label for the y-axesis.
    title (str, optional), default='State Switching Rates':
        Title for the plot.
    show_legend (bool, optional), default=True:
        Whether to show the legend.
    """
    fig, axes = plt.subplots(figsize=figsize, constrained_layout=True)
    multiplier = 0
    sessions = np.arange(1, SR.shape[0] + 1)
    num_states = SR.shape[1]
    colors = sns.color_palette("Set3", n_colors=num_states)
    if num_states > len(colors):
        extra_colors = sns.color_palette("husl", n_colors=num_states - len(colors))
        colors.extend(extra_colors)

    for k in range(num_states):
        offset = width * multiplier
        rects = axes.bar(sessions + offset, SR[:, k], width, color=colors[k])
        multiplier += 1
    
    axes.set_xticks(sessions)
    axes.set_xlabel(xlabel, fontsize=fontsize_labels)
    axes.set_ylabel(ylabel, fontsize=fontsize_labels)
    axes.set_title(title, fontsize=fontsize_title)
    
    ticks = np.linspace(1, SR.shape[0], SR.shape[0]).astype(int)
    # If there are more than 10 states then make a steps of 5
    if len(ticks)>10:
        n_ticks = num_ticks
    else:
        n_ticks = len(ticks)
    axes.set_xticks(np.linspace(1, SR.shape[0], n_ticks).astype(int))
    
    # Remove the frame around the plot
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    # Adjust tick label font size
    axes.tick_params(axis='both', labelsize=fontsize_labels)

    if show_legend:
        axes.legend(['State {}'.format(i+1) for i in range(num_states)], fontsize=fontsize_labels, loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()

def plot_state_lifetimes(LT, figsize=(8, 4), fontsize_labels=13, fontsize_title=16, width=0.18, xlabel='Subject', ylabel='Lifetime', title='State Lifetimes', show_legend=True, num_ticks=10):
    """
    Plot state lifetimes for different states.

    Parameters:
    -----------
    LT (numpy.ndarray): 
        State lifetime (dwell time) data matrix.
    figsize (tuple, optional), default=(8, 4):
        Figure size.
    fontsize_labels (int, optional), default=13:
        Font size for axeses labels.
    fontsize_title (int, optional), default=16:
        Font size for plot title.
    width (float, optional), default=0.18:
        Width of the bars.
    xlabel (str, optional), default='Subject':
        Label for the x-axesis.
    ylabel (str, optional), default='Lifetime':
        Label for the y-axesis.
    title (str, optional), default='State Lifetimes':
        Title for the plot.
    show_legend (bool, optional), default=True:
        Whether to show the legend.
    """
    fig, axes = plt.subplots(figsize=figsize, constrained_layout=True)
    multiplier = 0
    sessions = np.arange(1, LT.shape[0] + 1)
    num_states = LT.shape[1]
    colors = sns.color_palette("Set3", n_colors=num_states)
    if num_states > len(colors):
        extra_colors = sns.color_palette("husl", n_colors=num_states - len(colors))
        colors.extend(extra_colors)

    for k in range(num_states):
        offset = width * multiplier
        rects = axes.bar(sessions + offset, LT[:, k], width, color=colors[k])
        multiplier += 1
    
    axes.set_xticks(sessions, sessions)
    axes.set_xlabel(xlabel, fontsize=fontsize_labels)
    axes.set_ylabel(ylabel, fontsize=fontsize_labels)
    axes.set_title(title, fontsize=fontsize_title)
    
    ticks = np.linspace(1, LT.shape[0], LT.shape[0]).astype(int)
    # If there are more than 10 states then make a steps of 5
    if len(ticks)>10:
        n_ticks = num_ticks
    else:
        n_ticks = len(ticks)
    axes.set_xticks(np.linspace(1, LT.shape[0], n_ticks).astype(int))
    
    # Remove the frame around the plot
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    # Adjust tick label font size
    axes.tick_params(axis='both', labelsize=fontsize_labels)

    if show_legend:
        axes.legend(['State {}'.format(i+1) for i in range(num_states)], fontsize=fontsize_labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()

def plot_state_prob_and_covariance(init_stateP, TP, state_means, state_FC, cmap='viridis', figsize=(9, 7), num_ticks=5):
    """
    Plot HMM parameters.

    Parameters:
    -----------
    init_stateP : array-like
        Initial state probabilities.
    TP : array-like
        Transition probabilities.
    state_means : array-like
        State means.
    state_FC : array-like
        State covariances.
    cmap : str or Colormap, optional
        The colormap to be used for plotting. Default is 'viridis'.
    figsize : tuple, optional
        Figure size. Default is (9, 7).
    num_ticks : int, optional
        Number of ticks for the colorbars
    """
    # Define the number of plots and their layout
    num_plots = 3 + state_FC.shape[2]  # Number of plots including initial stateP, TP, state_means, and state_FC
    num_cols = min(num_plots, 3)  # Maximum number of columns
    num_rows = (num_plots - 1) // 3 + 1  # Calculate number of rows

    # Create the figure and subplots
    fig, axes = plt.subplots(num_rows, 3, figsize=figsize)  # Adjust figsize as needed

    # Plot initial state probabilities
    im0 = axes[0, 0].imshow(init_stateP.reshape(-1, 1), cmap=cmap)
    axes[0, 0].set_title("Initial state probabilities")
    axes[0, 0].set_xticks([])
    cbar0 = fig.colorbar(im0, ax=axes[0, 0])
    cbar0.set_ticks(np.linspace(init_stateP.min(), init_stateP.max(), num=num_ticks).round(2))
    ticks = np.linspace(0, init_stateP.shape[0]-1, init_stateP.shape[0]).astype(int)
    # If there are more than 10 states then make a steps of 5
    if len(ticks)>10:
        num_state = num_ticks
    else:
        num_state = len(ticks)
    axes[0, 0].set_yticks(np.linspace(0, init_stateP.shape[0]-1, num_state).astype(int))
    axes[0, 0].set_yticklabels(ticks + 1)  # Increment ticks by 1 for labels    
        
        
    # Plot transition probabilities
    im1 = axes[0, 1].imshow(TP, cmap=cmap)
    axes[0, 1].set_title("Transition probabilities")
    cbar1 = fig.colorbar(im1, ax=axes[0, 1])
    cbar1.set_ticks(np.linspace(TP.min(), TP.max(), num=num_ticks).round(2))
    ticks = np.linspace(0, TP.shape[0]-1, TP.shape[0]).astype(int)
    # If there are more than 10 states then make a steps of 5
    axes[0, 1].set_xticks(np.linspace(0, TP.shape[0]-1, num_state).astype(int))
    axes[0, 1].set_xticklabels(ticks + 1)  # Increment ticks by 1 for labels
    axes[0, 1].set_yticks(np.linspace(0, TP.shape[0]-1, num_state).astype(int))
    axes[0, 1].set_yticklabels(ticks + 1)  # Increment ticks by 1 for labels
    
    # Plot state means
    num_ticks = max(5, min(state_means.shape))
    im2 = axes[0, 2].imshow(state_means, cmap=cmap, aspect='auto')
    axes[0, 2].set_title("State means")
    cbar2 = fig.colorbar(im2, ax=axes[0, 2])
    cbar2.set_ticks(np.linspace(state_means.min(), state_means.max(), num=num_ticks).round(2))
    # Set ticks and labels
    ticks = np.linspace(0, state_means.shape[1]-1, num_ticks).astype(int)
    axes[0, 2].set_xticks(ticks)
    axes[0, 2].set_xticklabels(ticks + 1)  # Increment ticks by 1 for labels
    axes[0, 2].set_yticks(np.linspace(1, state_means.shape[0], num_ticks).astype(int))

    # Plot state covariances
    min_value = np.min(state_FC)
    max_value = np.max(state_FC)
    # Limits the number of ticks
    if len(ticks)>10:
        num_state = num_ticks
    else:
        num_state = len(ticks)
        
    ticks = np.linspace(0, state_FC.shape[0] - 1, num_state).astype(int)
    # Plot state covariances
    for k in range((num_cols*num_rows) -3): # have to fill the remaning number of subplots
        row_idx = (k + 3) // 3  # Shift row index by 3 to start from the second row
        col_idx = (k + 3) % 3
        if k < num_plots - 3:
            im = axes[row_idx, col_idx].imshow(state_FC[:, :, k], cmap=cmap, vmin=min_value, vmax=max_value)
            axes[row_idx, col_idx].set_title("State covariance\nstate #%s" % (k + 1))
            # Adjust tick locations
            axes[row_idx, col_idx].set_xticks(ticks)
            axes[row_idx, col_idx].set_yticks(ticks)
            axes[row_idx, col_idx].set_xticklabels(ticks + 1)  # Increment ticks by 1 for labels
            axes[row_idx, col_idx].set_yticklabels(ticks + 1) # Increment ticks by 1 for
            cbar = fig.colorbar(im, ax=axes[row_idx, col_idx])
            cbar.set_ticks(np.linspace(min_value, max_value, num=num_ticks).round(2))
        else:
            axes[row_idx, col_idx].axis('off')  # Leave empty plots blank

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()
    
def plot_condition_difference(Gamma_epoch, R_trials, title='Average Probability and Difference', fontsize=16, figsize=(9, 2), vertical_lines=None, line_colors=None, highlight_boxes=False, 
                              stimulus_onset=None, x_tick_min=None, x_tick_max=None, num_x_ticks=5):
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
    fontsize (int, optional), default=16:
        Font size for labels and title.
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
    Example usage:
    --------------
    plot_condition_difference(Gamma_epoch, R_trials, vertical_lines=[(10, 100)], highlight_boxes=True)
    """

    # Check if stimulus_onset is a number
    if stimulus_onset is not None and not isinstance(stimulus_onset, (int, float)):
        raise ValueError("stimulus_onset must be a number.")
    
    filt_val = np.zeros((2, Gamma_epoch.shape[0], Gamma_epoch.shape[2]))

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    conditions = np.unique(R_trials)

    # Variables to store global min and max y-values
    global_min = float('inf')
    global_max = float('-inf')

    # Plot for each condition
    for condition in conditions:
        for i in range(Gamma_epoch.shape[0]):
            filtered_values = Gamma_epoch[i, (R_trials == condition), :]
            filt_val[condition, i, :] = np.mean(filtered_values, axis=0).round(3)
        # Update global min and max y-values
        current_min = filt_val[condition, :, :].min()
        current_max = filt_val[condition, :, :].max()
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)

    # Generate x-tick positions based on the number of timepoints
    num_timepoints = Gamma_epoch.shape[0]
    x_tick_positions = np.linspace(0, num_timepoints - 1, 5).astype(int)

    # Generate x-tick labels based on user input or default to time points
    if x_tick_min is not None and x_tick_max is not None:
        x_tick_labels = np.linspace(x_tick_min, x_tick_max, num_x_ticks).round(2)
    else:
        x_tick_labels = x_tick_positions

    # Plot for each condition with standardized y-axis
    for condition in conditions:
        axes[condition].plot(filt_val[condition, :, :])
        axes[condition].set_title(f"Condition {condition}")
        axes[condition].set_xticks(x_tick_positions)
        axes[condition].set_xticklabels(x_tick_labels)
        axes[condition].set_yticks(np.linspace(global_min, global_max, 5).round(2))
        axes[condition].set_ylim(global_min, global_max)  # Set standardized y-limits # Set standardized y-limits

        
    # Find the element-wise difference
    difference = filt_val[0, :, :] - filt_val[1, :, :]

    # Plot the difference
    axes[2].plot(difference)
    axes[2].set_title("Difference")
    axes[2].set_xticks(np.linspace(0, Gamma_epoch.shape[0] - 1, 5).astype(int))
    axes[2].set_yticks(np.linspace(axes[2].get_ylim()[0], axes[2].get_ylim()[1], 5).round(2))
    axes[2].set_xticks(x_tick_positions)
    axes[2].set_xticklabels(x_tick_labels)

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
        ax.set_xlabel('Timepoints', fontsize=12)
        ax.set_ylabel('Average probability', fontsize=12)

    # Label each state on the right for the last figure (axes[2])
    state_labels = [f"State {state+1}" for state in range(Gamma_epoch.shape[2])]
    axes[2].legend(state_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

    fig.suptitle(title, fontsize=fontsize)

    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    
def plot_p_values_over_time(pval, figsize=(8, 4), xlabel="Timepoints", ylabel="P-values (Log Scale)",
                            title_text="P-values over time", stimulus_onset=None, x_tick_min=None, x_tick_max=None, num_x_ticks=5, 
                            tick_positions=[0, 0.001, 0.01, 0.05, 0.1, 0.3, 1], num_colors=259, 
                            alpha=0.05,plot_style = "line", linewidth=2.5,
                            ):
    """
    Plot a scatter plot of p-values over time with a log-scale y-axis and a colorbar.

    Parameters:
    -----------
    pval (numpy.ndarray):
        The p-values data to be plotted.
    figsize : tuple, optional, default=(8, 4):
        Figure size in inches (width, height).
    total_time_seconds : float, optional, default=None
        Total time duration in seconds. If provided, time points will be scaled accordingly.
    xlabel (str, optional), default="Timepoints":
        Label for the x-axis.
    ylabel (str, optional), default="P-values (Log Scale)":
        Label for the y-axis.
    title_text (str, optional), default="P-values over time":
        Title for the plot.
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
    """

    # Check if stimulus_onset is a number
    if stimulus_onset is not None and not isinstance(stimulus_onset, (int, float)):
        raise ValueError("stimulus_onset must be a number.")
    
    if pval.ndim != 1:
        # Raise an exception and stop function execution
        raise ValueError("To use the function 'plot_p_values_over_time', the variable for p-values must be one-dimensional.")

    # Generate Timepoints based on total_time_seconds
    # if total_time_seconds:
    #     time_points = np.linspace(0, total_time_seconds, len(pval))
    # else:
    #     time_points = np.arange(len(pval))
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

        # overwrite the values below alpha
        cmap_list[:num_elements_red,:]=cmap_red
        cmap_list[num_elements_red:,:]=cmap_blue
        cmap = LinearSegmentedColormap.from_list('custom_colormap', cmap_list)
            
    # Create the line plot with varying color based on p-values
    _, axes = plt.subplots(figsize=figsize)

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
                axes.plot([time_points[i], time_points[i+1]], [pval[i], pval[i+1]], color=color, linewidth=linewidth)
        else:
            for i in range(len(time_points)-1):
                if pval[i+1]>0.05:
                    color = cmap(norm(pval[i+1]))
                else:
                    color = cmap(norm(pval[i]))
                axes.plot([time_points[i], time_points[i+1]], [pval[i], pval[i+1]], color=color, linewidth=linewidth)
    elif plot_style=="scatter":
        axes.scatter(time_points, pval, c=pval, cmap=cmap, norm=LogNorm(vmin=1e-3, vmax=1))
    elif plot_style=="scatter_line":
        axes.scatter(time_points, pval, c=pval, cmap=cmap, norm=LogNorm(vmin=1e-3, vmax=1))    
            # Draw lines between points
        axes.plot(time_points, pval, color='black', linestyle='-', linewidth=1)
    # Add labels and title
    axes.set_xlabel(xlabel, fontsize=12)
    axes.set_ylabel(ylabel, fontsize=12)
    axes.set_title(title_text, fontsize=14)
    
    # define x_ticks
    x_tick_positions = np.linspace(0, len(pval), num_x_ticks).astype(int)

    # Generate x-tick labels based on user input or default to time points
    if x_tick_min is not None and x_tick_max is not None:
        x_tick_labels = np.linspace(x_tick_min, x_tick_max, num_x_ticks).round(2)
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
    # Add a colorbar to show the correspondence between colors and p-values
    divider = make_axes_locatable(axes)
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

    
    # Add stimulus onset line and label
    if stimulus_onset is not None:
        axes.axvline(x=stimulus_onset, color='black', linestyle='--', linewidth=2)
            
    plt.show()
    
    
def plot_p_values_bar(pval,xticklabels=[],  figsize=(9, 4), num_colors=256, xlabel="",
                        ylabel="P-values (Log Scale)", title_text="Bar Plot",
                        tick_positions=[0, 0.001, 0.01, 0.05, 0.1, 0.3, 1], top_adjustment=0.9, 
                        alpha = 0.05, pad_title=20, xlabel_rotation=45, pval_text_hight_same=False):
    """
    Visualize a bar plot with LogNorm and a colorbar.

    Parameters:
    -----------
    pval (numpy.ndarray):
        Array of p-values to be plotted.
    xticklabels (list, optional), default=[]:
        List of categories or variables.
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
    tick_positions (list, optional), default=[0, 0.001, 0.01, 0.05, 0.1, 0.3, 1]
        Positions of ticks on the colorbar.
    top_adjustment (float, optional), default=0.9:
        Adjustment for extra space between title and plot.
    alpha (float, optional), default=0.05:
        Alpha value is the threshold we set for the p-values when doing visualization.
    pad_title (int, optional), default=20:
        Padding for the plot title.
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
    fig, axes = plt.subplots(figsize=figsize)
    if isinstance(pval, (float, np.ndarray)) and np.size(pval) == 1:
    # It's a scalar, create a list with a single element
        xticklabels = [f"Var 1"] if xticklabels==[] else xticklabels
    else:
        # It's an iterable, use len()
        xticklabels =[f"Var {i+1}" for i in np.arange(len(pval))] if xticklabels==[] else xticklabels


    bars = plt.bar(xticklabels, pval, color=colormap(LogNorm(vmin=1e-3, vmax=1)(pval)))
    # Remove the legend
    #plt.legend().set_visible(False)

    # Add data labels on top of the bars
    if pval_text_hight_same:
        yval_hight_list=[]
        for bar in bars:
            # Get the yval_heights
            yval_hight_list.append(bar.get_height())
        yval_height =np.max(np.array(yval_hight_list))
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval_height + 0.5, round(yval, 3), ha='center', va='bottom', color='black', fontweight='bold')
    else:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 3), ha='center', va='bottom', color='black', fontweight='bold')


    # Set y-axis to log scale
    axes.set_yscale('log')

    # Customize plot
    plt.yscale('log')
    axes.set_xlabel(xlabel, fontsize=12)
    axes.set_ylabel(ylabel, fontsize=12)
    axes.set_title(title_text, fontsize=14, pad=pad_title)

    # Set xticks and rotate xtick labels
    axes.set_xticks(np.arange(len(xticklabels)))
    if xlabel_rotation==45:
        ha ='right'
        axes.set_xticklabels(xticklabels, rotation=xlabel_rotation, ha=ha)
    else:
        ha ='center'
        axes.set_xticklabels(xticklabels, rotation=xlabel_rotation, ha=ha)
    # Mark specific values on the y-axis
    plt.yticks([0.001, 0.01, 0.05, 0.1, 0.3, 1], ['0.001', '0.01', '0.05', '0.1', '0.3', '1'])

    # Add a colorbar to show the correspondence between colors and p-values
    divider = make_axes_locatable(axes)
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
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    plt.tight_layout()

    plt.show()
