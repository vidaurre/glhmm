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


import warnings
def plot_heatmap(pval, plot_method="pval", alpha = 0.05, normalize_vals=True, figsize=(12, 7), steps=11, title_text="Heatmap (p-values)", annot=True, cmap='default', xlabel="", ylabel="", xticklabels=None, none_diagonal = False):
    from matplotlib import cm, colors
    import seaborn as sb
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    """
    Plot a heatmap of p-values.

    Parameters
    ----------
    pval : numpy.ndarray
        The p-values data to be plotted.
    plot_method : str, optional
        This variable is used to define what kind of plot we are making.
        Valid options are "pval", "corr_coef" (Default="pval").
    normalize_vals : bool, optional
        If True, the data range will be normalized from 0 to 1 (Default=True).
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
    allowed_methods = ["pval", "corr_coef"]

    if plot_method not in allowed_methods:
        message = f"Warning: Unexpected method '{method}'. Please use 'pval' or 'corr_coeff'."
        warnings.warn(message, UserWarning)
        
    if pval.ndim==0:
        pval = np.reshape(pval, (1, 1))
        
    fig, ax = plt.subplots(figsize=figsize)
    if len(pval.shape)==1:
        pval =np.expand_dims(pval,axis=0)
    if plot_method =="pval":
        if cmap=='default':
            # Reverse colormap
            coolwarm_cmap = cm.coolwarm.reversed()
            # Generate an array of values representing the colormap
            num_colors = 256  # You can adjust the number of colors as needed
            color_array = np.linspace(0, 1, num_colors).reshape(1, -1)
            # Make a jump in color after alpha
            color_array[color_array > alpha] += 0.3
            # Create a new colormap with 
            new_cmap_list = coolwarm_cmap(color_array)[0]
            # if zero_white:
            #     # white at the lowest value
            #     new_cmap_list[0] = [1, 1, 1, 1]  # Set the first color to white

            # Create a new colormap with the modified color_array
            cmap = colors.ListedColormap(new_cmap_list)
            # Set the value of 0 to white in the colormap
        if none_diagonal:
            # Set the diagonal elements to NaN
            np.fill_diagonal(pval, np.nan)

        if normalize_vals:
            # Normalize the data range from 0 to 1
            norm = plt.Normalize(vmin=0, vmax=1)

            heatmap = sb.heatmap(pval, ax=ax, cmap=cmap, annot=annot, fmt=".3f", cbar=False, norm=norm)
        else:
            heatmap = sb.heatmap(pval, ax=ax, cmap=cmap, annot=annot, fmt=".3f", cbar=False)

    elif plot_method =="corr_coef":
        if cmap=='default':
            # seismic_cmap = cm.seismic.reversed()
            seismic_cmap = cm.seismic.reversed()
            #seismic_cmap = cm.RdBu.reversed()
            # Generate an array of values representing the colormap
            num_colors = 256  # You can adjust the number of colors as needed
            color_array = np.linspace(0, 1, num_colors).reshape(1, -1)
            # # Make a jump in color after alpha
            # color_array[color_array<0.5]+=0.1
            # color_array[color_array>0.5]-=0.1
            # # Set values in the specified index intervals to 0.5
            # indices=np.where(color_array == 0.5)[1]

            # # Set values in the specified index range to 5
            # color_array[0,np.min(indices):np.max(indices) + 1] = 0.49
            # Create a new colormap with 
            new_cmap_list = seismic_cmap(color_array)[0]
            cmap = colors.ListedColormap(new_cmap_list)
            
        if normalize_vals:
            # Normalize the data range from 0 to 1
            norm = plt.Normalize(vmin=-1, vmax=1)

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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # Create a custom colorbar
    colorbar = plt.colorbar(heatmap.get_children()[0], cax=cax)
    # colorbar.set_ticks([0, 0.25, 0.5, 1])  # Adjust ticks as needed

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

    # Customize plot background and grid style
    sb.set_style("white")
    ax.grid(color='lightgray', linestyle='--')

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
    
def plot_average_probability(Gamma_reconstruct, title='Average probability for each state', fontsize=16, figsize=(8, 6), vertical_lines=None, line_colors=None, highlight_boxes=False):
    import matplotlib.patches as patches
    import random
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
    plt.figure(figsize=figsize)

    # Plot each line with a label
    for state in range(Gamma_reconstruct.shape[-1]):
        plt.plot(Gamma_avg[:, state], label=f'State {state + 1}')

    # Add vertical lines and highlight boxes
    if vertical_lines:
        for idx, pair in enumerate(vertical_lines):
            if line_colors is True:
                if isinstance(line_colors, list) and len(line_colors) == len(vertical_lines):
                    color = line_colors[idx]
                else:
                    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            else:
                color = line_colors[idx] if line_colors and len(line_colors) > idx else 'gray'
            
            plt.axvline(x=pair[0], color=color, linestyle='--', linewidth=1)
            plt.axvline(x=pair[1], color=color, linestyle='--', linewidth=1)

            if highlight_boxes:
                rect = patches.Rectangle((pair[0], plt.ylim()[0]), pair[1] - pair[0], plt.ylim()[1] - plt.ylim()[0], linewidth=0, edgecolor='none', facecolor=color, alpha=0.2)
                plt.gca().add_patch(rect)

    # Add labels and legend
    plt.xlabel('Timepoints', fontsize=fontsize)
    plt.ylabel('Average probability', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)

    # Place legend to the right of the figure
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
