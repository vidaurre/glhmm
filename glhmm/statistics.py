"""
Permutation testing from Gaussian Linear Hidden Markov Model
@author: Nick Y. Larsen 2023
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from scipy.stats import pearsonr

def across_subjects(D_data, R_data, method="regression", Nperm=1000, confounds = None, dict_fam = None, test_statistic_option=False):
    from glhmm.palm_functions import palm_quickperms
    """
    Perform across subjects permutation testing.
    This function conducts statistical tests (regression, correlation, or correlation_com) between two datasets, `D_data`
    representing the measured data and `R_data` representing the dependent-variable, across different subjects using
    permutation testing. 
    This test is useful if we want to test for differences between the different subjects in ones study. 
    
                                    
    Parameters:
    --------------
        D_data (numpy.ndarray): Input data array of shape that can be either a 2D array or a 3D array.
                                For 2D the data is represented as a (n, p) matrix, where n represent 
                                the number of subjects, and p represents the number of predictors.
                                For a 3D array,it got a shape (T, n, q), where the first dimension 
                                represents timepoints, the second dimension represents the number of subjects, 
                                and the third dimension represents features. 
                                For 3D, permutation testing is performed per timepoint for each subject.              
        R_data (numpy.ndarray): The dependent-variable can be either a 2D array or a 3D array. 
                                For 2D array, it got a shape of (n, q), where n represent 
                                the number of subjects, and q represents the outcome of the dependent variable
                                For a 3D array,it got a shape (T, n, q), where the first dimension 
                                represents timepoints, the second dimension represents the number of subjects, 
                                and the third dimension represents a dependent variable   
                                For 3D, permutation testing is performed per timepoint for each subject                 
        method (str, optional): The statistical method to be used for the permutation test. Valid options are
                                "regression", "correlation", or "correlation_com". (default: "regression").      
                                Note: "correlation_com" stands for correlation combined and returns the both the statistical significance of Pearson's correlation coefficient and 2-tailed p-value                                         
        Nperm (int): Number of permutations to perform (default: 1000).                       
        confounds (numpy.ndarray or None, optional): 
                                The confounding variables to be regressed out from the input data (D_data).
                                If provided, the regression analysis is performed to remove the confounding effects. 
                                (default: None):     
        dict_fam (dict): 
                                Dictionary containing family structure information.                          
                                    -file_location (str): The file location of the family structure data in CSV format.
                                    -M (numpy.ndarray, optional): The matrix of attributes, which is not typically required.
                                                            Defaults to None.
                                    -CMC (bool, optional): A flag indicating whether to use the Conditional Monte Carlo method (CMC).
                                                    Defaults to False.
                                    -EE (bool, optional): A flag indicating whether to assume exchangeable errors, which allows permutation.
                                                    Defaults to True. Other option is not available            
        test_statistic_option (bool, optional): 
                                If True, the function will return the test statistic for each permutation.
                                (default: False) 
                                
    Returns:
    ----------  
        result (dict): A dictionary containing the following keys. Depending on the `test_statistic_option` and `method`, it can return the p-values, 
            correlation coefficients, test statistics.
            'pval': p-values for the test (T, p) if method=="Regression", else (T, p, q).
            'test_statistic': Test statistic is the permutation distribution with the shape (T, Nperm, p) if test_statistic_option is True, else None.
            'corr_coef': Correlation Coefficients for the test T, p, q) if method=="correlation or "correlation_com", else None.
            'pval_list': a list of p-values for each time point (T, Nperm, p) if test_statistic_option is True and method is "correlation_com", else None.
            'test_type': the type of test, which is the name of the function
            'method': the method used for analysis Valid options are
                    "regression", "correlation", or "correlation_com". (default: "regression").
                  
    Note:
        The function automatically determines whether permutation testing is performed per timepoint for each subject or
        for the whole data based on the dimensionality of `D_data`.
        The function assumes that the number of rows in `D_data` and `R_data` are equal
    """
    # Check validity of method and data_type
    valid_methods = ["regression", "correlation", "correlation_com"]
    check_value_error(method in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    # Get the shapes of the data
    n_T, _, n_p, n_q, D_data, R_data = get_input_shape(D_data, R_data)
    # Note for convension we wrote (T, p, q) => (n_T, n_p, n_q)
    
    # Crate the family structure 
    if dict_fam is not None:
        dict_mfam=fam_dict(dict_fam, Nperm) # modified dictionary of family structure

    # Initialize arrays based on shape of data shape and defined options
    pval, corr_coef, test_statistic_list, pval_list = initialize_arrays(D_data,R_data, n_p, n_q, n_T, method, Nperm, test_statistic_option)

    for t in tqdm(range(n_T)) if n_T > 1 else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        R_t = deconfound_Fnc(R_data[t, :], confounds)
        D_t = D_data[t, :]
        # Create test_statistic and pval_perms based on method
        test_statistic, pval_perms, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, D_t)

        if dict_fam is None:
            # Get indices for permutation
            permute_idx_list = across_subjects_permutation(Nperm, R_t)
        else:
            permute_idx_list = palm_quickperms(dict_mfam["EB"], M=dict_mfam["M"], nP=dict_mfam["nP"], 
                                               CMC=dict_mfam["CMC"], EE=dict_mfam["EE"])
            # Need to convert the index so it starts from 0
            permute_idx_list = permute_idx_list-1
            
        #for perm in range(Nperm):
        for perm in tqdm(range(Nperm)) if n_T == 1 else range(n_T):
            # Perform permutation on R_t
            Rin = R_t[permute_idx_list[:, perm]]
            # Calculate the permutation distribution
            test_statistic, pval_perms = test_statistic_calculations(D_t, Rin, perm, pval_perms, test_statistic, proj, method)
        # Calculate p-values
        pval, corr_coef = get_pval(test_statistic, pval_perms, Nperm, method, t, pval, corr_coef)
        
        # Output test statistic if it is set to True can be hard for memory otherwise
        if test_statistic_option==True:
            test_statistic_list[t,:] = test_statistic
            #  if pval_perms is empty (evaluates to False), the right-hand side of the assignment will be pval_list[t, :] itself, meaning that the array will remain unchanged.
            pval_list[t, :] = pval_perms if np.any(pval_perms) else pval_list[t, :]
    pval =np.squeeze(pval) if np.abs(np.sum(pval))>0 else [] 
    corr_coef =np.squeeze(corr_coef) if corr_coef is not None else []
    test_statistic_list =np.squeeze(test_statistic_list) if test_statistic_list is not None else []
    pval_list =np.squeeze(pval_list) if pval_list is not None  else []
    
    
    # Return results
    result = {
        'pval': pval,
        'corr_coef': corr_coef,
        'test_statistic': test_statistic_list,
        'pval_list': pval_list,
        'test_type': 'across_subjects',
        'method': method}
    return result



def across_trials_within_session(D_data, R_data, idD_data, method="regression", Nperm=1000, confounds=None,test_statistic_option=False):
    """
    This function conducts statistical tests (regression, correlation, or correlation_com) between two datasets, `D_data`
    representing the measured data  and `R_data` representing the dependent-variable, across different trials within a session 
    using permutation testing. This test is useful if we want to test for differences between trials in one or more sessions. 
    An example could be if we want to test if any learning is happening during a session that might speed up reaction times.
                      
    Parameters:
    --------------
        D_data (numpy.ndarray): Input data array of shape that can be either a 2D array or a 3D array.
                                For 2D array, it got a shape of (n, p), where n represent 
                                the number of trials, and p represents the number of predictors (e.g., brain region)
                                For a 3D array,it got a shape (T, n, p), where the first dimension 
                                represents timepoints, the second dimension represents the number of trials, 
                                and the third dimension represents features/predictors. 
                                In the latter case, permutation testing is performed per timepoint for each subject.              
        R_data (numpy.ndarray): The dependent-variable can be either a 2D array or a 3D array. 
                                For 2D array, it got a shape of (n, q), where n represent 
                                the number of trials, and q represents the outcome/dependent variable
                                For a 3D array,it got a shape (T, n, q), where the first dimension 
                                represents timepoints, the second dimension represents the number of trials, 
                                and the third dimension represents a dependent variable                    
        idD_data (numpy.ndarray): The indices for each trial within the session. It should be a 2D array where each row
                                  represents the start and end index for a trial.    
        method (str, optional): The statistical method to be used for the permutation test. Valid options are
                                "regression", "correlation", or "correlation_com". (default: "regression").
                                Note: "correlation_com" stands for correlation combined and returns the both the statistical significance of Pearson's correlation coefficient and 2-tailed p-value
        Nperm (int): Number of permutations to perform (default: 1000). 
        confounds (numpy.ndarray or None, optional): 
                                The confounding variables to be regressed out from the input data (D_data).
                                If provided, the regression analysis is performed to remove the confounding effects. 
                                (default: None):                                                              
        test_statistic_option (bool, optional): 
                                If True, the function will return the test statistic for each permutation.
                                (default: False) 
                                
                                                      
    Returns:
    ----------  
        result (dict): A dictionary containing the following keys. Depending on the `test_statistic_option` and `method`, it can return the p-values, 
            correlation coefficients, test statistics.
            'pval': p-values for the test (T, p) if method=="Regression", else (T, p, q).
            'test_statistic': Test statistic is the permutation distribution with the shape (T, Nperm, p) if test_statistic_option is True, else None.
            'corr_coef': Correlation Coefficients for the test T, p, q) if method=="correlation or "correlation_com", else None.
            'pval_list': a list of p-values for each time point (T, Nperm, p) if test_statistic_option is True and method is "correlation_com", else None.
            'test_type': the type of test, which is the name of the function
            'method': the method used for analysis Valid options are
                    "regression", "correlation", or "correlation_com". (default: "regression").

    Note:
        The function automatically determines whether permutation testing is performed per timepoint for each subject or
        for the whole data based on the dimensionality of `D_data`.
        The function assumes that the number of rows in `D_data` and `R_data` are equal
    """
    
    # Check validity of method
    valid_methods = ["regression", "correlation", "correlation_com"]
    check_value_error(method in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    # Get input shape information
    n_T, _, n_p, n_q, D_data, R_data = get_input_shape(D_data, R_data)
    n_q = R_data.shape[-1]
    
    # Get indices for permutation
    if len(idD_data.shape)==2:
        idx_array = get_indices_array(idD_data)
    else:
        idx_array =idD_data.copy()        

    # Initialize arrays based on shape of data shape and defined options
    pval, corr_coef, test_statistic_list, pval_list = initialize_arrays(D_data, R_data, n_p, n_q, n_T, method, Nperm, test_statistic_option)


    for t in tqdm(range(n_T)) if n_T > 1 else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        R_t = deconfound_Fnc(R_data[t, :], confounds)
        D_t = D_data[t, :]
        
        # Create test_statistic and pval_perms based on method
        test_statistic, pval_perms, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, D_t)
        
        # Calculate permutation matrix of D_t 
        permute_idx_list = within_session_across_trial_permutation(Nperm,R_t, idx_array)
                 
        for perm in range(Nperm):
        #for perm in tqdm(range(Nperm)) if n_T == 1 else range(n_T):
            # Perform permutation on R_t
            Rin = R_t[permute_idx_list[:, perm]]
            # Calculate the permutation distribution
            test_statistic, pval_perms = test_statistic_calculations(D_t, Rin, perm, pval_perms, test_statistic, proj, method)
        # Calculate p-values
        pval, corr_coef = get_pval(test_statistic, pval_perms, Nperm, method, t, pval, corr_coef)
        if test_statistic_option==True:
            test_statistic_list[t,:] = test_statistic
            #  if pval_perms is empty (evaluates to False), the right-hand side of the assignment will be pval_list[t, :] itself, 
            #  meaning that the array will remain unchanged.
            pval_list[t, :] = pval_perms if np.any(pval_perms) else pval_list[t, :]
    pval =np.squeeze(pval) if np.abs(np.sum(pval))>0 else [] 
    corr_coef =np.squeeze(corr_coef) if corr_coef is not None  else []
    test_statistic_list =np.squeeze(test_statistic_list) if test_statistic_list is not None else []
    pval_list =np.squeeze(pval_list) if pval_list is not None  else []
    
    # Return results
    result = {
        'pval': pval,
        'corr_coef': corr_coef,
        'test_statistic': test_statistic_list,
        'pval_list': pval_list,
        'test_type': 'across_trials_within_session',
        'method': method}
    
    return result
   
def across_sessions_within_subject(D_data, R_data, idD_data, method="regression", Nperm=1000, confounds=None,test_statistic_option=False):
    """
    This function conducts statistical tests (regression, correlation, or correlation_com) between two datasets, `D_data`
    representing the measured data and `R_data` representing the dependent-variable, across sessions within the same subject 
    while keeping the trial order the same using permutation testing. This is useful for checking out the effects of 
    long-term treatments or tracking changes in brain responses across sessions over time 
                        
                                
    Parameters:
    --------------
        D_data (numpy.ndarray): Input data array of shape that can be either a 2D array or a 3D array.
                                For 2D array, it got a shape of (n, p), where n_ST represent 
                                the number of subjects, and each column represents a feature (e.g., brain region). 
                                For a 3D array,it got a shape (T, n, p), where the first dimension 
                                represents timepoints, the second dimension represents the number of trials, 
                                and the third dimension represents features/predictors.             
        R_data (numpy.ndarray): The dependent-variable can be either a 2D array or a 3D array. 
                                For 2D array, it got a shape of (n, q), where n represent 
                                the number of trials, and q represents the outcome/dependent variable
                                For a 3D array,it got a shape (T, n, q), where the first dimension 
                                represents timepoints, the second dimension represents the number of trials, 
                                and the third dimension represents a dependent variable                   
        idD_data (numpy.ndarray): The indices for each trial within the session. It should be a 2D array where each row
                                  represents the start and end index for a trial.    
        method (str, optional): The statistical method to be used for the permutation test. Valid options are
                                "regression", "correlation", or "correlation_com". (default: "regression").
                                Note: "correlation_com" stands for correlation combined and returns the both the statistical significance of Pearson's correlation coefficient and 2-tailed p-value
        Nperm (int): Number of permutations to perform (default: 1000).                
        confounds (numpy.ndarray or None, optional): 
                                The confounding variables to be regressed out from the input data (D_data).
                                If provided, the regression analysis is performed to remove the confounding effects. 
                                (default: None):                                                              
        test_statistic_option (bool, optional): 
                                If True, the function will return the test statistic for each permutation.
                                (default: False) 
                                   
                                
    Returns:
    ----------  
        result (dict): A dictionary containing the following keys. Depending on the `test_statistic_option` and `method`, it can return the p-values, 
            correlation coefficients, test statistics.
            'pval': p-values for the test (T, p) if method=="Regression", else (T, p, q).
            'test_statistic': Test statistic is the permutation distribution with the shape (T, Nperm, p) if test_statistic_option is True, else None.
            'corr_coef': Correlation Coefficients for the test T, p, q) if method=="correlation or "correlation_com", else None.
            'pval_list': a list of p-values for each time point (T, Nperm, p) if test_statistic_option is True and method is "correlation_com", else None.
            'test_type': the type of test, which is the name of the function
            'method': the method used for analysis Valid options are
                    "regression", "correlation", or "correlation_com". (default: "regression").
                    Note: "correlation_com" stands for correlation combined and returns the both the statistical significance of Pearson's correlation coefficient and 2-tailed p-value
                  
    Note:
        The function automatically determines whether permutation testing is performed per timepoint for each subject or
        for the whole data based on the dimensionality of `D_data`.
        The function assumes that the number of rows in `D_data` and `R_data` are equal

    """ 
    # Check validity of method
    valid_methods = ["regression", "correlation", "correlation_com"]
    check_value_error(method in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    # Get indices for permutation
    if len(idD_data.shape)==2:
        idx_array = get_indices_array(idD_data)
    else:
        idx_array =idD_data.copy()

    _, trial_per_subject = np.unique(idx_array, return_counts=True)
    if len(set(trial_per_subject)) != 1:
        raise ValueError("Warning: Unequal number of trials per subject prohibs permutation between subjects when exchangeable is False.")
    
    # Get input shape information
    n_T, _, n_p,n_q, D_data, R_data = get_input_shape(D_data, R_data)
    #n_q = R_data.shape[-1]
    
# Initialize arrays based on shape of data shape and defined options
    pval, corr_coef, test_statistic_list, pval_list = initialize_arrays(D_data, R_data, n_p, n_q, n_T, method, Nperm, test_statistic_option)
    for t in tqdm(range(n_T)) if n_T > 1 else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        R_t = deconfound_Fnc(R_data[t, :], confounds)
        D_t = D_data[t, :]
        
        # Create test_statistic and pval_perms based on method
        test_statistic, pval_perms, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, D_t)
        
        # Calculate permutation matrix of D_t 
        permute_idx_list = within_session_across_session_permutation(Nperm, D_t, idx_array)
        
        for perm in range(Nperm):
        #for perm in tqdm(range(Nperm)) if n_T == 1 else range(n_T):
            # Perform permutation on R_t
            Rin = R_t[permute_idx_list[:, perm]]
            # Calculate the permutation distribution
            test_statistic, pval_perms = test_statistic_calculations(D_t, Rin, perm, pval_perms, test_statistic, proj, method)
        # Caluclate p-values
        pval, corr_coef = get_pval(test_statistic, pval_perms, Nperm, method, t, pval, corr_coef)
        if test_statistic_option==True:
            test_statistic_list[t,:] = test_statistic
            #  if pval_perms is empty (evaluates to False), the right-hand side of the assignment will be pval_list[t, :] itself, meaning that the array will remain unchanged.
            pval_list[t, :] = pval_perms if np.any(pval_perms) else pval_list[t, :]
    pval =np.squeeze(pval) if np.abs(np.sum(pval))>0 else [] 
    corr_coef =np.squeeze(corr_coef) if corr_coef is not None  else []
    test_statistic_list =np.squeeze(test_statistic_list) if test_statistic_list is not None  else []
    pval_list =np.squeeze(pval_list) if pval_list is not None  else []
              
    # Return values
    result = {
        'pval': pval,
        'corr_coef': [] if np.sum(corr_coef)==0 else corr_coef,
        'test_statistic': [] if np.sum(test_statistic_list)==0 else test_statistic_list,
        'pval_list': [] if np.sum(pval_list)==0 else pval_list,
        'test_type': 'across_sessions_within_subject',
        'method': method}
    return result
    
def across_visits(input_data, vpath_data, n_states, method="regression", Nperm=1000, confounds=None, test_statistic_option=False, statistic ="mean"):
    from itertools import combinations
    """
    Perform permutation testing within a session for continuous data.

    This function conducts statistical tests (regression, correlation, or correlation_com, one_vs_rest and state_pairs) between a hidden state path
    (`vpath_data`) and a dependent variable (`Y_data`) within each session using permutation testing. 
    The goal is to test if the decodeed vpath is the most like path chosen

    Parameters:
    --------------            
        input_data (numpy.ndarray):     The dependent-variable with shape (n, q). Where n is the number of samples (n_timepoints X n_trials) and 
                                    q represents a dependent/target variables       
        vpath_data (numpy.ndarray): The hidden state path data of the continuous measurements represented as a (n, p) matrix. 
                                    It could be a 2D matrix where each row represents a trials over a period of time and
                                    each column represents a state variable and gives the shape ((n_timepoints X n_trials), n_states). 
                                    If it is a 1D array of of shape ((n_timepoints X n_trials),) where each row value represent a giving state.                                 
        n_states (int):             The number of hidden states in the hidden state path data.
        method (str, optional):     The statistical method to be used for the permutation test. Valid options are
                                    "regression", "correlation", "correlation_com", "one_vs_rest" or "state_pairs".
                                    Note: "correlation_com" exports both correlation coefficients and p-values. (default: "regression").
        Nperm (int):                Number of permutations to perform (default: 1000). 
        test_statistic_option (bool, optional): 
                                    If True, the function will return the test statistic for each permutation.
                                    (default: False) 
        statistic (str, optional)   The chosen statistic to be calculated when applying the methods "one_vs_rest" or "state_pairs".
                                    Valid options are "mean" or "median". (default: "mean)
                                
                                
    Returns:
    ----------  
        result (dict): A dictionary containing the following keys. Depending on the `test_statistic_option` and `method`, it can return the p-values, 
            correlation coefficients, test statistics.
            'pval': p-values for the test (T, p) if method=="Regression", else (T, p, q).
            'test_statistic': Test statistic is the permutation distribution with the shape (T, Nperm, p) if test_statistic_option is True, else None.
            'corr_coef': Correlation Coefficients for the test T, p, q) if method=="correlation or "correlation_com", else None.
            'pval_list': a list of p-values for each time point (T, Nperm, p) if test_statistic_option is True and method is "correlation_com", else None.
            'test_type': the type of test, which is the name of the function
            'method': the method used for analysis Valid options are
                    "regression", "correlation", or "correlation_com", "one_vs_rest" and "state_pairs" (default: "regression").
                
    Note:
        The function assumes that the number of rows in `vpath_data` and `Y_data` are equal
    """
    
    # Check validity of method
    valid_methods = ["regression", "correlation", "correlation_com", "one_vs_rest", "state_pairs"]
    check_value_error(method in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    valid_statistic = ["mean", "median"]
    check_value_error(statistic.lower() in valid_statistic, "Invalid option specified for 'statistic'. Must be one of: " + ', '.join(valid_statistic))
    
    # Convert vpath from matrix to vector
    vpath_array=generate_vpath_1D(vpath_data)

    if method == 'regression':
        # Get input shape information
        n_T, _, n_p, n_q, input_data, vpath_data= get_input_shape(input_data, vpath_array)
    else:
        # Get input shape information
        n_T, _, n_p, n_q, input_data, vpath_data= get_input_shape(input_data, vpath_data)  

    # Initialize arrays based on shape of data shape and defined options
    pval, corr_coef, test_statistic_list, pval_list = initialize_arrays(input_data,vpath_data, n_p, n_q,
                                                                            n_T, method, Nperm,
                                                                            test_statistic_option)

    # Print tqdm over n_T if there are more than one timepoint
    for t in tqdm(range(n_T)) if n_T > 1 else range(n_T):
        # Correct for confounds and center data_t
        data_t = deconfound_Fnc(input_data[t, :], confounds)
        # Create test_statistic and pval_perms based on method
        if method != "state_pairs":
            ###################### Permutation testing for other tests beside state pairs #################################
            # Create test_statistic and pval_perms based on method
            test_statistic, pval_perms, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, 
                                                                               data_t)
        
            # Perform permutation testing
            for perm in tqdm(range(Nperm)) if n_T == 1 else range(n_T):
                # Create vpath_surrogate
                vpath_surrogate= surrogate_state_time(perm, vpath_array,n_states)
                if method =="one_vs_rest":
                    for state in range(n_states):
                        test_statistic[perm,state] =calculate_baseline_difference(vpath_surrogate, data_t, state+1, statistic.lower())
                elif method =="regression":
                    test_statistic, pval_perms = test_statistic_calculations(data_t,vpath_surrogate , perm, pval_perms,
                                                                            test_statistic, proj, method)
                else:
                    # Apply 1 hot encoding
                    vpath_surrogate_onehot = viterbi_path_to_stc(vpath_surrogate,n_states)
                    # Apply t-statistic on the vpath_surrogate
                    test_statistic, pval_perms = test_statistic_calculations(data_t,vpath_surrogate_onehot , perm, pval_perms,
                                                                                test_statistic, proj, method)

            pval, corr_coef = get_pval(test_statistic, pval_perms, Nperm, method, t, pval, corr_coef)
        ###################### Permutation testing for state pairs #################################
        elif method =="state_pairs":
            # Run this code if it is "state_pairs"
            # Correct for confounds and center data_t
            data_t = deconfound_Fnc(input_data[t, :], confounds)
            # Generates all unique combinations of length 2 
            pairwise_comparisons = list(combinations(range(1, n_states + 1), 2))
            test_statistic = np.zeros((Nperm, len(pairwise_comparisons)))
            pval = np.zeros((n_states, n_states))
            # Iterate over pairwise state comparisons
            for idx, (state_1, state_2) in tqdm(enumerate(pairwise_comparisons), total=len(pairwise_comparisons), desc="Pairwise comparisons"):    
                # Generate surrogate state-time data and calculate differences for each permutation
                for perm in range(Nperm):
                    vpath_surrogate = surrogate_state_time(perm, vpath_array, n_states)
                    test_statistic[perm,idx] = calculate_statepair_difference(vpath_surrogate, data_t, state_1, state_2, statistic)
                
                p_val= np.sum(test_statistic[:,idx] >= test_statistic[0,idx], axis=0) / (Nperm + 1)
                pval[state_1-1, state_2-1] = p_val
                pval[state_2-1, state_1-1] = 1 - p_val
            corr_coef =[]
            pval_perms = []
            
        if test_statistic_option:
            test_statistic_list[t, :] = test_statistic
            # If pval_perms is empty (evaluates to False), the right-hand side of the assignment will be pval_list[t, :]
            # itself, meaning that the array will remain unchanged.
            pval_list[t, :] = pval_perms if np.any(pval_perms) else pval_list[t, :]

    pval =np.squeeze(pval) if np.abs(np.sum(pval))>0 else [] 
    corr_coef =np.squeeze(corr_coef) if corr_coef is not None else []
    test_statistic_list =np.squeeze(test_statistic_list) if test_statistic_list is not None else []
    pval_list =np.squeeze(pval_list) if pval_list is not None else []
    
    # Return results
    result = {
        
        'pval': pval,
        'corr_coef': corr_coef,
        'test_statistic': test_statistic_list,
        'pval_list': pval_list,
        'test_type': 'across_visits',
        'method': method} 
    return result



def check_value_error(condition, error_message):
    """
    Checks a given condition and raises a ValueError with the specified error message if the condition is not met.

    Parameters:
    --------------
        condition (bool): The condition to check.
        error_message (str): The error message to raise if the condition is not met.
    """
    # Check if a condition is False and raise a ValueError with the given error message
    if not condition:
        raise ValueError(error_message)


def get_input_shape(D_data, R_data):
    """
    Computes the input shape parameters for permutation testing.

    Parameters:
    --------------
        D_data (numpy.ndarray): The input data array.
        R_data (numpy.ndarray): The dependent variable.

    Returns:
    ----------  
        n_T (int): The number of timepoints.
        n_ST (int): The number of subjects or trials.
        n_p (int): The number of features.
        D_data (numpy.ndarray): The updated input data array.
        R_data (numpy.ndarray): The updated dependent variable.
    """
    # Get the input shape of the data and perform necessary expansions if needed
    if R_data.ndim == 1:
        R_data = np.expand_dims(R_data, axis=1)
    
    if len(D_data.shape) == 1:
        # This is normally the case when using viterbi path
        print("performing permutation testing for viterbi path")
        D_data = np.expand_dims(D_data, axis=1)
        D_data = np.expand_dims(D_data, axis=0)
        R_data = np.expand_dims(R_data, axis=0)
        n_T, n_ST, n_p = D_data.shape
        n_q = R_data.shape[-1]
    elif len(D_data.shape) == 2:
        # Performing permutation testing for the whole data
        
        D_data = np.expand_dims(D_data, axis=0)
        if D_data.ndim !=R_data.ndim:
            R_data = np.expand_dims(R_data, axis=0)
        n_T, n_ST, n_p = D_data.shape
        n_q = R_data.shape[-1]

    else:
        # Performing permutation testing per timepoint
        print("performing permutation testing per timepoint")
        n_T, n_ST, n_p = D_data.shape

        # Tile the R_data if it doesn't match the number of timepoints in D_data
        if R_data.shape[0] != D_data.shape[0]:
            R_data = np.tile(R_data, (D_data.shape[0],1,1)) 
        n_q = R_data.shape[-1]
    return n_T, n_ST, n_p, n_q, D_data, R_data

def fam_dict(dict_fam, Nperm):
    """
    Process a dictionary containing family structure information.

    Parameters:
    --------------
        dict_fam (dict): Dictionary containing family structure information.
            file_location (str): The file location of the family structure data in CSV format.
            M (numpy.ndarray, optional): The matrix of attributes, which is not typically required.
                                    Defaults to None.
            nP (int): The number of permutations to generate.
            CMC (bool, optional): A flag indicating whether to use the Conditional Monte Carlo method (CMC).
                            Defaults to False.
            EE (bool, optional): A flag indicating whether to assume exchangeable errors, which allows permutation.
                            Defaults to True.             
        Nperm (int): Number of permutations.

    Returns:
    ----------  
        dict_mfam (dict): Modified dictionary with processed values.
            EB (numpy.ndarray): Block structure representing relationships between subjects.
            M (numpy.ndarray, optional): The matrix of attributes, which is not typically required.
                                    Defaults to None.
            nP (int): The number of permutations to generate.
            CMC (bool, optional): A flag indicating whether to use the Conditional Monte Carlo method (CMC).
                            Defaults to False.
            EE (bool, optional): A flag indicating whether to assume exchangeable errors, which allows permutation.
                            Defaults to True.
    """
    
    # dict_fam: dictionary of family structure
    # Nperm: number of permutations

    default_values = {
        'file_location' : 'None',
        'M': 'None',
        'CMC': 'False',
        'EE': 'False',
        'nP': Nperm
    }
    dict_mfam =dict_fam.copy()

    # Validate and load family structure data
    if 'file_location' not in dict_mfam:
        raise ValueError("The 'file_location' variable must be defined in dict_fam.")
    
    # Convert the DataFrame to a matrix
    EB = pd.read_csv(dict_mfam['file_location'], header=None).to_numpy()
    
    # Check for invalid keys in dict_fam
    invalid_keys = set(dict_mfam.keys()) - set(default_values.keys())
    if not invalid_keys== set():
        valid_keys = ['M', 'CMC', 'EE']
        check_value_error(
            invalid_keys in valid_keys, "Invalid keys in dict_fam: Must be one of: " + ', '.join(valid_keys)
        )
    
    # Set default values for M, CMC, and EE
    del dict_mfam['file_location']
    dict_mfam['EB'] = EB
    dict_mfam['nP'] = Nperm
    dict_mfam.setdefault('M', default_values['M'])
    dict_mfam.setdefault('CMC', default_values['CMC'])
    dict_mfam.setdefault('EE', default_values['EE'])
    
    return dict_mfam

def initialize_arrays(D_data, R_data, n_p, n_q, n_T, method, Nperm, test_statistic_option):
    from itertools import combinations
    """
    Initializes the result arrays for permutation testing.

    Parameters:
    --------------
        D_data (numpy.ndarray): The independt variable
        R_data (numpy.ndarray): The dependent variable
        n_p (int): The number of features.
        n_q (int): The number of predictions.
        n_T (int): The number of timepoints.
        method (str): The method to use for permutation testing.
        Nperm (int): Number of permutations.
        test_statistic_option (bool): If True, return the test statistic values.

    Returns:
    ----------  
        pval (numpy array): p-values for the test (n_T, n_p) if test_statistic_option is False, else None.
        corr_coef (numpy array): Correlation coefficient for the test (n_T, n_p, n_q) if method=correlation or method = correlation_com, else None.
        test_statistic_list (numpy array): Test statistic values (n_T, Nperm, n_p) or (n_T, Nperm, n_p, n_q) if method=correlation or method = correlation_com, else None.
        pval_list (numpy array): P-values for each time point (n_T, Nperm, n_p) or (n_T, Nperm, n_p, n_q) if test_statistic_option is True and method is "correlation_com", else None.
    """

    # Initialize the arrays based on the selected method and data dimensions
    if  method == "regression":
        pval = np.zeros((n_T, n_q))
        corr_coef = None
        if test_statistic_option==True:
            test_statistic_list = np.zeros((n_T, Nperm, n_q))
            pval_list = np.zeros((n_T, Nperm, n_p))
        else:
            test_statistic_list= None
            pval_list =None
            
    elif method == "correlation_com" or method == "correlation" :
        pval = np.zeros((n_T, n_p, n_q))
        corr_coef = pval.copy()
        
        if test_statistic_option==True:    
            test_statistic_list = np.zeros((n_T, Nperm, n_p, n_q))
            pval_list = np.zeros((n_T, Nperm, n_p, n_q))
        else:
            test_statistic_list= None
            pval_list =None
    elif method == "state_pairs":
        pval = np.zeros((n_T, R_data.shape[-1], R_data.shape[-1]))
        corr_coef = []
        pairwise_comparisons = list(combinations(range(1, R_data.shape[-1] + 1), 2))
        if test_statistic_option==True:    
            test_statistic_list = np.zeros((n_T, Nperm, len(pairwise_comparisons)))
            pval_list = np.zeros((n_T, Nperm, n_p))
        else:
            test_statistic_list= None
            pval_list =None
    elif method == "one_vs_rest":
        pval = np.zeros((n_T, n_p, n_q))
        corr_coef = np.zeros((n_T, n_p, n_q))
        if test_statistic_option==True:
            test_statistic_list = np.zeros((n_T, Nperm, n_q))
            pval_list = np.zeros((n_T, Nperm, n_q))
        else:
            test_statistic_list= None
            pval_list =None   

    return pval, corr_coef, test_statistic_list, pval_list


def deconfound_Fnc(R_data, confounds=None):
    """
    Calculate the R_data array for permutation testing.

    Parameters:
    --------------
        R_data (numpy.ndarray): The input data array.
        confounds (numpy.ndarray or None): The confounds array (default: None).

    Returns:
    ----------  
        numpy.ndarray: Calculated R_t array.
    """
    
    # Calculate the centered data matrix based on confounds (if provided)
    if confounds is not None:
        # Centering confounds
        confounds = confounds - np.mean(confounds, axis=0)
        # Centering R_data
        R_data = R_data - np.mean(R_data, axis=0)
        # Regressing out confounds from R_data
        R_t = R_data - confounds @ np.linalg.pinv(confounds) @ R_data
    else:
        # Centering R_data
        R_t = R_data - np.mean(R_data, axis=0)
    return R_t

def initialize_permutation_matrices(method, Nperm, n_p, n_q, D_data):
    """
    Initializes the permutation matrices and projection matrix for permutation testing.

    Parameters:
    --------------
        method (str): The method to use for permutation testing.
        Nperm (int): The number of permutations.
        n_p (int): The number of features.
        n_q (int): The number of predictions.
        D_data (numpy.ndarray): The independent variable.
        

    Returns:
    ----------  
        test_statistic (numpy.ndarray): The permutation array.
        pval_perms (numpy.ndarray): The p-value permutation array.
        proj (numpy.ndarray or None): The projection matrix (None for correlation methods).
    """
    # Initialize the permutation matrices based on the selected method
    if method in {"correlation", "correlation_com"}:
        test_statistic = np.zeros((Nperm, n_p, n_q))
        pval_perms = np.zeros((Nperm, n_p, n_q))
        proj = None
    else:
        # Regression got a N by q matrix
        test_statistic = np.zeros((Nperm, n_q))
        pval_perms = [] # empty
        # Define regularization parameter
        regularization = 0.001
        # Regularized parameter estimation
        regularization_matrix = regularization * np.eye(D_data.shape[1])
        # Projection matrix
        # This matrix is then used to project permuted data matrix (Din) to obtain the regression coefficients (beta)
        #proj = np.matmul(np.linalg.inv(np.matmul(D_data.T,D_data) +regularization_matrix),D_data.T
        proj = np.linalg.inv(D_data.T @ D_data + regularization_matrix) @ D_data.T

    return test_statistic, pval_perms, proj

def across_subjects_permutation(Nperm, D_t):
    """
    Generates between-subject indices for permutation testing.

    Parameters:
    --------------
        Nperm (int): The number of permutations.
        D_t (numpy.ndarray): The preprocessed data array.
        
    Returns:
    ----------  
        permute_idx_list (numpy.ndarray): Permutation matrix of subjects it got a shape (n_ST, Nperm)
    """
    permute_idx_list = np.zeros((D_t.shape[0],Nperm), dtype=int)
    for perm in range(Nperm):
        if perm == 0:
            permute_idx_list[:,perm] = np.arange(D_t.shape[0])
        else:
            permute_idx_list[:,perm] = np.random.permutation(D_t.shape[0])
    return permute_idx_list

def get_pval(test_statistic, pval_perms, Nperm, method, t, pval, corr_coef):
    """
    Computes p-values and correlation matrix for permutation testing.

    Parameters:
    --------------
        test_statistic (numpy.ndarray): The permutation array.
        pval_perms (numpy.ndarray): The p-value permutation array.
        Nperm (int): The number of permutations.
        method (str): The method used for permutation testing.
        t (int): The timepoint index.
        pval (numpy.ndarray): The p-value array.
        corr_coef (numpy.ndarray): The correlation p-value array.

    Returns:
    ----------  
        
        pval (numpy.ndarray): Updated updated p-value .
        corr_coef (numpy.ndarray): Updated correlation p-value arrays.
        
    # Ref: https://github.com/OHBA-analysis/HMM-MAR/blob/master/utils/testing/permtest_aux.m
    """
    # Positive direction
    if method == "regression" or method == "one_vs_rest":
        # Count every time there is an higher estimated explaied variace (R^2)
        pval[t, :] = np.sum(test_statistic >= test_statistic[0,:], axis=0) / (Nperm + 1)
    elif method == "correlation":
        corr_coef[t, :] = np.sum(test_statistic >= test_statistic[0,:], axis=0) / (Nperm + 1)
    elif method == "correlation_com":
        corr_coef[t, :] = np.sum(test_statistic >= test_statistic[0,:], axis=0) / (Nperm + 1)
        # p-values are left tailed, so in this case we are looking for values more extreme than the unpermuted
        pval[t, :] = np.sum(pval_perms <= pval_perms[0,:], axis=0) / (Nperm + 1)

    return pval, corr_coef


def get_indices_array(idD_data):
    """
    Generates an indices array based on given data indices.

    Parameters:
    --------------
        idD_data (numpy.ndarray): The data indices array.

    Returns:
    ----------  
        idx_array (numpy.ndarray): The generated indices array.
    """
    # Get an array of indices based on the given idD_data ranges
    max_value = np.max(idD_data[:, 1])
    idx_array = np.zeros(max_value + 1, dtype=int)
    for count, (start, end) in enumerate(idD_data):
        idx_array[start:end + 1] = count
    return idx_array


def within_session_across_trial_permutation(Nperm, R_t, idx_array):
    """
    Generates permutation matrix of within-session across-trial data based on given indices.

    Parameters:
    --------------
        Nperm (int): The number of permutations.
        R_t (numpy.ndarray): The preprocessed data array.
        idx_array (numpy.ndarray): The indices array.

    Returns:
    ----------  
        permute_idx_list (numpy.ndarray): Permutation matrix of subjects it got a shape (n_ST, Nperm)
    """
    # Perform within-session between-trial permutation based on the given indices
    # Createing the permutation matrix
    permute_idx_list = np.zeros((R_t.shape[0], Nperm), dtype=int)
    for perm in range(Nperm):
        if perm == 0:
            permute_idx_list[:,perm] = np.arange(R_t.shape[0])
        else:
            unique_indices = np.unique(idx_array)
            count = 0
            for i in unique_indices:
                if i ==0:
                    count =count+R_t[idx_array == unique_indices[i], :].shape[0]
                    permute_idx_list[0:count,perm]=np.random.permutation(np.arange(0,count))
                else:
                    idx_count=R_t[idx_array == unique_indices[i], :].shape[0]
                    count =count+idx_count
                    permute_idx_list[count-idx_count:count,perm]=np.random.permutation(np.arange(count-idx_count,count))
    return permute_idx_list

def permute_subject_trial_idx(idx_array):
    """
    Permutes an array based on unique values while maintaining the structure.
    
    Parameters:
    --------------
        idx_array (numpy.ndarray): Input array to be permuted.
    
    Returns:
    ----------  
        list: Permuted array based on unique values.
    """
    unique_values, value_counts = np.unique(idx_array, return_counts=True)
    
    permuted_array = []
    unique_values_perm = np.random.permutation(unique_values)
    
    for value in unique_values_perm:
        permuted_array.extend([value] * value_counts[np.where(unique_values == value)][0])
    
    return permuted_array


def within_session_across_session_permutation(Nperm, D_t, idx_array):
    """
    Generates permutation matrix of within-session across-session data based on given indices.

    Parameters:
    --------------
        Nperm (int): The number of permutations.
        D_t (numpy.ndarray): The preprocessed data array.
        idx_array (numpy.ndarray): The indices array.


    Returns:
    ----------  
        permute_idx_list (numpy.ndarray): The within-session continuos indices array.
    """
    permute_idx_list = np.zeros((D_t.shape[0],Nperm), dtype=int)
    for perm in range(Nperm):
        if perm == 0:
            permute_idx_list[:,perm] = np.arange(D_t.shape[0])
        else:
            idx_array_perm = permute_subject_trial_idx(idx_array)
            unique_indices = np.unique(idx_array_perm)
            positions_permute = [np.where(np.array(idx_array_perm) == i)[0] for i in unique_indices]
            permute_idx_list[:,perm] = np.concatenate(positions_permute,axis=0)
    return permute_idx_list


def generate_vpath_1D(vpath):
    """
    Convert a 2D array representing a matrix with one non-zero element in each row
    into a 1D array where each element is the column index of the non-zero element.

    Parameters:
        vpath(numpy.ndarray):       A 2D array where each row has only one non-zero element. 
                                    Or a 1D array where each row represents a sate number

    Returns:
        vpath_array(numpy.ndarray): A 1D array containing the column indices of the non-zero elements.
                                    If the input array is already 1D, it returns a copy of the input array.

    """
    if np.ndim(vpath) == 2:
        vpath_array = np.argmax(vpath, axis=1) + 1
    else:
        # Then it is already a vector
        vpath_array = vpath.copy()

    return vpath_array

def surrogate_state_time(perm, viterbi_path,n_states):
    """
    Generates surrogate state-time matrix based on a given Viterbi path.

    Parameters:
    --------------
        perm (int): The permutation number.
        viterbi_path (numpy.ndarray): 1D array or 2D matrix containing the Viterbi path.
        n_states (int): The number of states

    Returns:
    ----------  
        viterbi_path_surrogate (numpy.ndarray): A 1D array representing the surrogate Viterbi path
    """
       
    if perm == 0:
        if np.ndim(viterbi_path) == 2 and viterbi_path.shape[1] !=1:
            viterbi_path_surrogate = viterbi_path_to_stc(viterbi_path, n_states)
        elif np.ndim(viterbi_path) == 2 and viterbi_path.shape[1] ==1:
            viterbi_path_surrogate = np.squeeze(viterbi_path.copy().astype(int))
        else:
            viterbi_path_surrogate = viterbi_path.copy().astype(int)
            
    else:
        viterbi_path_surrogate = surrogate_viterbi_path(viterbi_path, n_states)
    return viterbi_path_surrogate

def viterbi_path_to_stc(viterbi_path, n_states):
    """
    Convert Viterbi path to state-time matrix.

    Parameters:
    --------------
        viterbi_path (numpy.ndarray): 1D array or 2D matrix containing the Viterbi path.
        n_states (int): Number of states in the hidden Markov model.

    Returns:
    ----------  
        stc (numpy.ndarray): State-time matrix where each row represents a time point and each column represents a state.
    """
    stc = np.zeros((len(viterbi_path), n_states), dtype=int)
    if np.min(viterbi_path)==0:
        stc[np.arange(len(viterbi_path)), viterbi_path] = 1
    else:
        stc[np.arange(len(viterbi_path)), viterbi_path-1] = 1
    return stc



def surrogate_viterbi_path(viterbi_path, n_states):
    """
    Generate surrogate Viterbi path based on state-time matrix.

    Parameters:
    --------------
        viterbi_path (numpy.ndarray):   1D array or 2D matrix containing the Viterbi path.
        n_states (int):                 Number of states in the hidden Markov model.

    Returns:
    ----------  
        viterbi_path_surrogate (numpy.ndarray): Surrogate Viterbi path as a 1D array representing the state indices.
                                            The number of states in the array varies from 1 to n_states
    """
    # Check if the input viterbi_path is a 1D array or 2D matrix
    if np.squeeze(viterbi_path).ndim == 1:
        viterbi_path_1D = np.squeeze(viterbi_path).copy()
        stc = viterbi_path_to_stc(viterbi_path_1D, n_states)
    else:
        viterbi_path_1D = np.argmax(viterbi_path, axis=1)
        stc = viterbi_path.copy()

    # Initialize the surrogate Viterbi path, state probability and previous state
    viterbi_path_surrogate = np.zeros(viterbi_path_1D.shape)
    state_probs = stc.mean(axis=0).cumsum()
    prev_state = None
    index = 0
    # Generate surrogate path
    while index < len(viterbi_path_1D):
        # Find the next index where the state changes
        t_next = np.where(viterbi_path_1D[index:] != viterbi_path_1D[index])[0]

        if len(t_next) == 0:
            t_next = len(viterbi_path_1D)
        else:
            #t_next = t_next[0]
            t_next = index + t_next[0]
            
        if prev_state is not None:
            # Create a copy of the state-time matrixte
            transition_prob_matrix = stc.copy()
            
            # Remove the column of the previous sta
            transition_prob_matrix = np.delete(transition_prob_matrix, prev_state, axis=1)
            # Find rows where every element is 0
            zero_rows = np.all(transition_prob_matrix == 0, axis=1)

            # Remove rows where every element is 0
            transition_prob_matrix_filt = transition_prob_matrix[~zero_rows]
            
            # Renormalize state probabilities to sum up to 1
            state_probs = transition_prob_matrix_filt.mean(axis=0).cumsum()
            # Generate a random number to determine the next state
            ran_num = random.uniform(0, 1)
            state = np.where(state_probs >= ran_num)[0][0]
            # Adjust state index if needed
            if state >=prev_state and n_states!=prev_state+1: #n_states!=prev_state+1 => Not equal to last state
                state+=1
            elif state+1==prev_state and n_states==prev_state+1:
                state=state+0
        else:
            # If it's the first iteration, randomly choose the initial state
            ran_num = random.uniform(0, 1)
            state = np.where(state_probs >= ran_num)[0][0]
        
        # Update the surrogate path. 
        state += 1 # We update with +1 in the end because the surrogate path starts from 0
        viterbi_path_surrogate[index:t_next] = state
        index = t_next
        prev_state = state - 1  # Update the previous state index
    
    return viterbi_path_surrogate.astype(int)

def calculate_baseline_difference(vpath_array, R_data, state, statistic):
    """
    Calculate the difference between the specified statistics of a state and all other states combined.

    Parameters:
    --------------
        vpath_data (numpy.ndarray): The Viterbi path as of integer values that range from 1 to n_states.
        R_data (numpy.ndarray):     The dependent-variable associated with each state.
        state(numpy.ndarray):       the state for which the difference is calculated.
        statistic (str)             The chosen statistic to be calculated. Valid options are "mean" or "median".

    Returns:
    ----------  
        difference (float)            the calculated difference between the specified state and all other states combined.
    """
    if statistic == 'median':
        state_R_data = np.median(R_data[vpath_array == state])
        other_R_data = np.median(R_data[vpath_array != state])
    elif statistic == 'mean':
        state_R_data = np.mean(R_data[vpath_array == state])
        other_R_data = np.mean(R_data[vpath_array != state])
    else:
        raise ValueError("Invalid stat value")
    # Detect any difference
    difference = np.abs(state_R_data) - np.abs(other_R_data)
    
    return difference

def calculate_statepair_difference(vpath_array, R_data, state_1, state_2, stat):
    """
    Calculate the difference between the specified statistics of two states.

    Parameters:
    --------------
        vpath_data (numpy.ndarray): The Viterbi path as of integer values that range from 1 to n_states.
        R_data (numpy.ndarray):     The dependent-variable associated with each state.
        state_1 (int):              First state for comparison.
        state_2 (int):              Second state for comparison.
        statistic (str):            The chosen statistic to be calculated. Valid options are "mean" or "median".

    Returns:
    ----------  
        difference (float):           The calculated difference between the two states.
    """
    if stat == 'mean':
        state_1_R_data = np.mean(R_data[vpath_array == state_1])
        state_2_R_data = np.mean(R_data[vpath_array == state_2])
    elif stat == 'median':
        state_1_R_data = np.median(R_data[vpath_array == state_1])
        state_2_R_data = np.median(R_data[vpath_array == state_2])
    else:
        raise ValueError("Invalid stat value")
    # Detect any difference
    difference = state_1_R_data - state_2_R_data
    return difference

def test_statistic_calculations(Din, Rin, perm, pval_perms, test_statistic, proj, method):
    """
    Calculates the test_statistic array and pval_perms array based on the given data and method.

    Parameters:
    --------------
        Din (numpy.ndarray): The data array.
        Rin (numpy.ndarray): The dependent variable.
        perm (int): The permutation index.
        pval_perms (numpy.ndarray): The p-value permutation array.
        test_statistic (numpy.ndarray): The permutation array.
        proj (numpy.ndarray or None): The projection matrix (None for correlation methods).
        method (str): The method used for permutation testing.

    Returns:
    ----------  
        test_statistic (numpy.ndarray): Updated test_statistic array.
        pval_perms (numpy.ndarray): Updated pval_perms array.
    """
    if method == 'regression':
        # Calculate regression_coefficients (beta)
        beta = proj @ Rin 
        # Calculate the root mean squared error
        test_statistic[perm,:] = np.sqrt(np.sum((Din @ beta - Rin) ** 2, axis=0))
    elif method == 'correlation':
        corr_coef = np.corrcoef(Din, Rin, rowvar=False)
        corr_matrix = corr_coef[:Din.shape[1], Din.shape[1]:]
        test_statistic[perm, :, :] = np.abs(corr_matrix)
    elif method == "correlation_com":
        corr_coef = np.corrcoef(Din, Rin, rowvar=False)
        corr_matrix = corr_coef[:Din.shape[1], Din.shape[1]:]
        pval_matrix = np.zeros(corr_matrix.shape)
        for i in range(Din.shape[1]):
            for j in range(Rin.shape[1]):
                _, pval_matrix[i, j] = pearsonr(Din[:, i], Rin[:, j])

        test_statistic[perm, :, :] = np.abs(corr_matrix)
        pval_perms[perm, :, :] = pval_matrix
    return test_statistic, pval_perms


def pval_test(pval, method='fdr_bh', alpha = 0.05):
    from statsmodels.stats import multitest as smt
    """
    This function performs multiple thresholding and correction for a 2D numpy array of p-values.

    Parameters:
    --------------
        pval: 2D numpy array of p-values.
        method: method used for FDR correction. Default is 'fdr_bh'.
            bonferroni : one-step correction
            sidak : one-step correction
            holm-sidak : step down method using Sidak adjustments
            holm : step-down method using Bonferroni adjustments
            simes-hochberg : step-up method (independent)   
            hommel : closed method based on Simes tests (non-negative)
            fdr_bh : Benjamini/Hochberg (non-negative)
            fdr_by : Benjamini/Yekutieli (negative)
            fdr_tsbh : two stage fdr correction (non-negative)
            fdr_tsbky : two stage fdr correction (non-negative)
        
        alpha: significance level. Default is 0.05.

    Returns:
    ---------- 
        p_values_corrected: 2D numpy array of corrected p-values.
        rejected_corrected: 2D numpy array of boolean values indicating rejected null hypotheses.
    """
    # Identify and replace NaN values in the pval array
    nan_mask = np.isnan(pval)
    non_nan_values = pval[~nan_mask]

    # Perform the FDR correction using statsmodels multitest module
    rejected, p_values_corrected, _, _ = smt.multipletests(non_nan_values.flatten(), alpha=0.05, method='fdr_bh', returnsorted=False)

    # Create arrays with NaN values for rejected and p_values_corrected
    rejected_corrected = np.empty_like(pval, dtype=bool)
    p_values_corrected_full = np.empty_like(pval)
    
    # Fill in non-NaN values with corrected values
    rejected_corrected[~nan_mask] = rejected
    p_values_corrected_full[~nan_mask] = p_values_corrected

    if np.sum(np.isnan(pval))>1:
        np.fill_diagonal(p_values_corrected_full, np.nan)
    # return the corrected p-values and boolean values indicating rejected null hypotheses
    return p_values_corrected_full,rejected_corrected


def get_timestamp_indices(n_timestamps, n_subjects):
    """
    Generate indices of the timestamps for each subject in the data.

    Parameters:
    --------------
    n_timestamps (int): Number of timestamps.
    n_subjects (int): Number of subjects.

    Returns:
    ----------  
    indices (ndarray): NumPy array representing the indices of the timestamps for each subject.

    Example:
    get_timestamp_indices(5, 3)
    array([[ 0,  5],
           [ 5, 10],
           [10, 15]])
    """
    indices = np.column_stack([np.arange(0, n_timestamps * n_subjects, n_timestamps),
                               np.arange(0 + n_timestamps, n_timestamps * n_subjects + n_timestamps, n_timestamps)])

    return indices
