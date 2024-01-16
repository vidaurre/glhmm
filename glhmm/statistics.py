"""
Permutation testing from Gaussian Linear Hidden Markov Model
@author: Nick Y. Larsen 2023
"""

import numpy as np
import pandas as pd
import random
import copy
from tqdm import tqdm
from scipy.stats import pearsonr

def across_subjects(D_data, R_data, method="regression", Nperm=1000, confounds = None, dict_fam = None, test_statistic_option=False, method_regression="RMSE"):
    # from glhmm.palm_functions import palm_quickperms
    # from palm_functions import palm_quickperms
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
        method_regression (str): The regression method used for the test statistics (default:RMSE).
                                 valid options are "RMSE" (Root mean squared error) or "R2" for explained variance.                 
                                
    Returns:
    ----------  
        result (dict): A dictionary containing the following keys. Depending on the `test_statistic_option` and `method`, it can return the p-values, 
            correlation coefficients, test statistics.
            'pval': p-values for the test (T, p) if method=="Regression", else (T, p, q).
            'test_statistic': Test statistic is the permutation distribution with the shape (T, Nperm, p) if test_statistic_option is True, else None.
            'corr_coef': Correlation Coefficients for the test T, p, q) if method=="correlation or "correlation_com", else None.
            'test_type': the type of test, which is the name of the function
            'method': the method used for analysis Valid options are
                    "regression", "correlation", or "correlation_com". (default: "regression").
                  
    Note:
        The function automatically determines whether permutation testing is performed per timepoint for each subject or
        for the whole data based on the dimensionality of `D_data`.
        The function assumes that the number of rows in `D_data` and `R_data` are equal
    """
    # pval_perms
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
    pval, corr_coef, test_statistic_list = initialize_arrays(D_data,R_data, n_p, n_q, n_T, method, Nperm, test_statistic_option)

    for t in tqdm(range(n_T)) if n_T > 1 else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        R_t, D_t = deconfound_Fnc(R_data[t, :],D_data[t, :], confounds)
        
        if method == "correlation" or method == "correlation_com":
            # Calculate correlation coefficient
            corr_coef[t, :] = get_corr_coef(D_t,R_t)
            test_statistic= None

        # Perform permutation if method not equal to "correlation"
        if method != "correlation":
            # Create test_statistic based on method
            test_statistic, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, D_t)

            if dict_fam is None:
                # Get indices for permutation
                permute_idx_list = across_subjects_permutation(Nperm, R_t)
            else:
                permute_idx_list = palm_quickperms(dict_mfam["EB"], M=dict_mfam["M"], nP=dict_mfam["nP"], 
                                                CMC=dict_mfam["CMC"], EE=dict_mfam["EE"])
                # Need to convert the index so it starts from 0
                permute_idx_list = permute_idx_list-1
                
            #for perm in range(Nperm):
            for perm in tqdm(range(Nperm)) if n_T == 1 else range(Nperm):
                # Perform permutation on R_t
                Rin = R_t[permute_idx_list[:, perm]]
                # Calculate the permutation distribution
                test_statistic = test_statistic_calculations(D_t, Rin, perm, test_statistic, proj, method, method_regression)
            # Calculate p-values
            pval = get_pval(test_statistic, Nperm, method, t, pval, method_regression)
            
            # Output test statistic if it is set to True can be hard for memory otherwise
            if test_statistic_option==True:
                test_statistic_list[t,:] = test_statistic
    pval =np.squeeze(pval) if np.abs(np.sum(pval))>0 else [] 
    corr_coef =np.squeeze(corr_coef) if corr_coef is not None else []
    test_statistic_list =np.squeeze(test_statistic_list) if test_statistic_list is not None else []
    
    # Return results
    result = {
        'pval': pval,
        'corr_coef': corr_coef,
        'test_statistic': test_statistic_list,
        'test_type': 'across_subjects',
        'method': method}
    return result



def across_trials_within_session(D_data, R_data, idx_data, method="regression", Nperm=1000, confounds=None, trial_timepoints=None,test_statistic_option=False, method_regression="RMSE"):
    """
    This function conducts statistical tests (regression, correlation, or correlation_com) between two datasets, `D_data`
    representing the measured data  and `R_data` representing the dependent-variable, across different trials within a session 
    using permutation testing. This test is useful if we want to test for differences between trials in one or more sessions. 
    An example could be if we want to test if any learning is happening during a session that might speed up times.
                      
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
        idx_data (numpy.ndarray): The indices for each trial within the session. It should be a 2D array where each row
                                  represents the start and end index for a trial.    
        method (str, optional): The statistical method to be used for the permutation test. Valid options are
                                "regression", "correlation", or "correlation_com". (default: "regression").
                                Note: "correlation_com" stands for correlation combined and returns the both the statistical significance of Pearson's correlation coefficient and 2-tailed p-value
        Nperm (int): Number of permutations to perform (default: 1000). 
        confounds (numpy.ndarray or None, optional): 
                                The confounding variables to be regressed out from the input data (D_data).
                                If provided, the regression analysis is performed to remove the confounding effects. 
                                (default: None):    
        trial_timepoints (int): Number of timepoints for each trial (default: None)                                                          
        test_statistic_option (bool, optional): 
                                If True, the function will return the test statistic for each permutation.
                                (default: False) 
        method_regression (str): The regression method used for the test statistics (default:RMSE).
                                 valid options are "RMSE" (Root mean squared error) or "R2" for explained variance. 
                                
                                                      
    Returns:
    ----------  
        result (dict): A dictionary containing the following keys. Depending on the `test_statistic_option` and `method`, it can return the p-values, 
            correlation coefficients, test statistics.
            'pval': p-values for the test (T, p) if method=="Regression", else (T, p, q).
            'test_statistic': Test statistic is the permutation distribution with the shape (T, Nperm, p) if test_statistic_option is True, else None.
            'corr_coef': Correlation Coefficients for the test T, p, q) if method=="correlation or "correlation_com", else None.
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
    if len(idx_data.shape)==2:
        idx_array = get_indices_array(idx_data)
    else:
        idx_array =idx_data.copy()        

    # Initialize arrays based on shape of data shape and defined options
    pval, corr_coef, test_statistic_list = initialize_arrays(D_data, R_data, n_p, n_q, n_T, method, Nperm, test_statistic_option)


    for t in tqdm(range(n_T)) if n_T > 1 else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        R_t, D_t = deconfound_Fnc(R_data[t, :],D_data[t, :], confounds)
        if method == "correlation" or method == "correlation_com":
            # Calculate correlation coefficient
            corr_coef[t, :] = get_corr_coef(D_t,R_t)
            test_statistic= None

        # Perform permutation if method not equal to "correlation"
        if method != "correlation":
            # Create test_statistic and pval_perms based on method
            test_statistic, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, D_t)
            
            # Calculate permutation matrix of D_t 
            permute_idx_list = within_session_across_trial_permutation(Nperm,R_t, idx_array,trial_timepoints)
                    
            for perm in range(Nperm):
            #for perm in tqdm(range(Nperm)) if n_T == 1 else range(n_T):
                # Perform permutation on R_t
                Rin = R_t[permute_idx_list[:, perm]]
                # Calculate the permutation distribution
                test_statistic = test_statistic_calculations(D_t, Rin, perm, test_statistic, proj, method, method_regression)
            # Calculate p-values
            pval = get_pval(test_statistic, Nperm, method, t, pval, method_regression)
            if test_statistic_option==True:
                test_statistic_list[t,:] = test_statistic
    pval =np.squeeze(pval) if np.abs(np.sum(pval))>0 else [] 
    corr_coef =np.squeeze(corr_coef) if corr_coef is not None  else []
    test_statistic_list =np.squeeze(test_statistic_list) if test_statistic_list is not None else []
    
    # Return results
    result = {
        'pval': pval,
        'corr_coef': corr_coef,
        'test_statistic': test_statistic_list,
        'test_type': 'across_trials_within_session',
        'method': method}
    
    return result
   
def across_sessions_within_subject(D_data, R_data, idx_data, method="regression", Nperm=1000, confounds=None,test_statistic_option=False, method_regression="RMSE"):
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
        idx_data (numpy.ndarray): The indices for each trial within the session. It should be a 2D array where each row
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
        method_regression (str): The regression method used for the test statistics (default:RMSE).
                                 valid options are "RMSE" (Root mean squared error) or "R2" for explained variance. 
                                   
                                
    Returns:
    ----------  
        result (dict): A dictionary containing the following keys. Depending on the `test_statistic_option` and `method`, it can return the p-values, 
            correlation coefficients, test statistics.
            'pval': p-values for the test (T, p) if method=="Regression", else (T, p, q).
            'test_statistic': Test statistic is the permutation distribution with the shape (T, Nperm, p) if test_statistic_option is True, else None.
            'corr_coef': Correlation Coefficients for the test T, p, q) if method=="correlation or "correlation_com", else None.
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
    if len(idx_data.shape)==2:
        idx_array = get_indices_array(idx_data)
    else:
        idx_array =idx_data.copy()

    # Get input shape information
    n_T, _, n_p,n_q, D_data, R_data = get_input_shape(D_data, R_data)
    #n_q = R_data.shape[-1]
    
# Initialize arrays based on shape of data shape and defined options
    pval, corr_coef, test_statistic_list = initialize_arrays(D_data, R_data, n_p, n_q, n_T, method, Nperm, test_statistic_option)
    for t in tqdm(range(n_T)) if n_T > 1 else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        R_t, D_t = deconfound_Fnc(R_data[t, :],D_data[t, :], confounds)
        if method == "correlation" or method == "correlation_com":
            # Calculate correlation coefficient
            corr_coef[t, :] = get_corr_coef(D_t, R_t)
            test_statistic= None
        # Perform permutation if method not equal to "correlation"
        if method != "correlation":
            # Create test_statistic and pval_perms based on method
            test_statistic, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, D_t)
        
            # Calculate permutation matrix of D_t 
            permute_idx_list = within_subject_across_sessions_permutation(Nperm, D_t, idx_array)
            
            for perm in range(Nperm):
            #for perm in tqdm(range(Nperm)) if n_T == 1 else range(n_T):
                # Perform permutation on R_t
                Rin = R_t[permute_idx_list[:, perm]]
                # Calculate the permutation distribution
                test_statistic = test_statistic_calculations(D_t, Rin, perm, test_statistic, proj, method, method_regression)
            # Caluclate p-values
            pval = get_pval(test_statistic, Nperm, method, t, pval, method_regression)
            if test_statistic_option==True:
                test_statistic_list[t,:] = test_statistic
    pval =np.squeeze(pval) if np.abs(np.sum(pval))>0 else [] 
    corr_coef =np.squeeze(corr_coef) if corr_coef is not None  else []
    test_statistic_list =np.squeeze(test_statistic_list) if test_statistic_list is not None  else []
              
    # Return values
    result = {
        'pval': pval,
        'corr_coef': [] if np.sum(corr_coef)==0 else corr_coef,
        'test_statistic': [] if np.sum(test_statistic_list)==0 else test_statistic_list,
        'test_type': 'across_sessions_within_subject',
        'method': method}
    return result
    
def across_visits(input_data, vpath_data, n_states, method="regression", Nperm=1000, confounds=None, test_statistic_option=False, statistic ="mean", method_regression="RMSE"):
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
        method_regression (str):    The regression method used for the test statistics (default:RMSE).
                                    valid options are "RMSE" (Root mean squared error) or "R2" for explained variance.                         
                                
    Returns:
    ----------  
        result (dict): A dictionary containing the following keys. Depending on the `test_statistic_option` and `method`, it can return the p-values, 
            correlation coefficients, test statistics.
            'pval': p-values for the test (T, p) if method=="Regression", else (T, p, q).
            'test_statistic': Test statistic is the permutation distribution with the shape (T, Nperm, p) if test_statistic_option is True, else None.
            'corr_coef': Correlation Coefficients for the test T, p, q) if method=="correlation or "correlation_com", else None.
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
    pval, corr_coef, test_statistic_list = initialize_arrays(input_data,vpath_data, n_p, n_q,
                                                                            n_T, method, Nperm,
                                                                            test_statistic_option)

    # Print tqdm over n_T if there are more than one timepoint
    for t in tqdm(range(n_T)) if n_T > 1 else range(n_T):
        # Correct for confounds and center data_t
        data_t, _ = deconfound_Fnc(input_data[t, :],None, confounds)
        if method == "correlation" or method == "correlation_com":
            # Calculate correlation coefficient
            corr_coef[t, :] = get_corr_coef(data_t, vpath_array)
            test_statistic= None

        # Perform permutation if method not equal to "correlation"
        if method != "correlation":  
            # Create test_statistic and pval_perms based on method
            if method != "state_pairs":
                ###################### Permutation testing for other tests beside state pairs #################################
                # Create test_statistic and pval_perms based on method
                test_statistic, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, 
                                                                                data_t)
                # Perform permutation testing
                for perm in tqdm(range(Nperm)) if n_T == 1 else range(n_T):
                    # Create vpath_surrogate
                    vpath_surrogate= surrogate_state_time(perm, vpath_array,n_states)
                    if method =="one_vs_rest":
                        for state in range(n_states):
                            test_statistic[perm,state] =calculate_baseline_difference(vpath_surrogate, data_t, state+1, statistic.lower())
                    elif method =="regression":
                        test_statistic = test_statistic_calculations(data_t,vpath_surrogate , perm,
                                                                                test_statistic, proj, method, method_regression)
                    else:
                        # Apply 1 hot encoding
                        vpath_surrogate_onehot = viterbi_path_to_stc(vpath_surrogate,n_states)
                        # Apply t-statistic on the vpath_surrogate
                        test_statistic = test_statistic_calculations(data_t,vpath_surrogate_onehot , perm,
                                                                                    test_statistic, proj, method, method_regression)
                pval = get_pval(test_statistic, Nperm, method, t, pval, method_regression)
            ###################### Permutation testing for state pairs #################################
            elif method =="state_pairs":
                # Run this code if it is "state_pairs"
                # Correct for confounds and center data_t
                data_t, _ = deconfound_Fnc(input_data[t, :],None, confounds)
                
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

    pval =np.squeeze(pval) if np.abs(np.sum(pval))>0 else [] 
    corr_coef =np.squeeze(corr_coef) if corr_coef is not None else []
    test_statistic_list =np.squeeze(test_statistic_list) if test_statistic_list is not None else []
    
    # Return results
    result = {
        
        'pval': pval,
        'corr_coef': corr_coef,
        'test_statistic': test_statistic_list,
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
    """

    # Initialize the arrays based on the selected method and data dimensions
    if  method == "regression":
        pval = np.zeros((n_T, n_q))
        corr_coef = None
        if test_statistic_option==True:
            test_statistic_list = np.zeros((n_T, Nperm, n_q))
        else:
            test_statistic_list= None
            
    elif method == "correlation_com" or method == "correlation" :
        pval = np.zeros((n_T, n_p, n_q))
        corr_coef = pval.copy()
        
        if test_statistic_option==True:    
            test_statistic_list = np.zeros((n_T, Nperm, n_p, n_q))
        else:
            test_statistic_list= None
    elif method == "state_pairs":
        pval = np.zeros((n_T, R_data.shape[-1], R_data.shape[-1]))
        corr_coef = []
        pairwise_comparisons = list(combinations(range(1, R_data.shape[-1] + 1), 2))
        if test_statistic_option==True:    
            test_statistic_list = np.zeros((n_T, Nperm, len(pairwise_comparisons)))
        else:
            test_statistic_list= None
    elif method == "one_vs_rest":
        pval = np.zeros((n_T, n_p, n_q))
        corr_coef = np.zeros((n_T, n_p, n_q))
        if test_statistic_option==True:
            test_statistic_list = np.zeros((n_T, Nperm, n_q))
        else:
            test_statistic_list= None

    return pval, corr_coef, test_statistic_list


def deconfound_Fnc(R_data, D_data, confounds=None):
    """
    Deconfound the variables R_data and D_data for permutation testing.

    Parameters:
    --------------
        R_data (numpy.ndarray): The input data array.
        D_data (numpy.ndarray or None): The second input data array (default: None).
            If None, assumes we are working across visits, and D_data represents the Viterbi path of a sequence.
        confounds (numpy.ndarray or None): The confounds array (default: None).

    Returns:
    ----------  
        numpy.ndarray: Deconfounded R_data array.
        numpy.ndarray: Deconfounded D_data array (returns None if D_data is None).
            If D_data is None, assumes we are working across visits
    """
    
    # Calculate the centered data matrix based on confounds (if provided)
    if confounds is not None:
        # Centering confounds
        confounds = confounds - np.mean(confounds, axis=0)
        # Centering R_data
        R_data = R_data - np.mean(R_data, axis=0)
        # Regressing out confounds from R_data
        R_t = R_data - confounds @ np.linalg.pinv(confounds) @ R_data
        # Check if D_data is provided
        if D_data is not None:
            # Centering D_data
            D_data = D_data - np.mean(D_data, axis=0)
            # Regressing out confounds from D_data
            D_t = D_data - confounds @ np.linalg.pinv(confounds) @ D_data
        else:
            D_t = None
    else:
        # Centering R_data and D_data
        R_t = R_data - np.mean(R_data, axis=0)
        D_t = None if D_data is None else D_data - np.mean(D_data, axis=0)
    
    return R_t, D_t

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
        proj = None
    else:
        # Regression got a N by q matrix 
        test_statistic = np.zeros((Nperm, n_q))
        # Define regularization parameter
        regularization = 0.001
        # Regularized parameter estimation
        regularization_matrix = regularization * np.eye(D_data.shape[1])  # Regularization term for Ridge regression
        # Fit the Ridge regression model
        # The projection matrix is then used to project permuted data matrix (Din) to obtain the regression coefficients (beta)
        proj = np.linalg.inv(D_data.T @ D_data + regularization_matrix) @ D_data.T  # Projection matrix for Ridge regression
    return test_statistic, proj

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

def get_pval(test_statistic, Nperm, method, t, pval, method_regression):
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
        method_regression (str): The regression method used for the test statistics (Default:"RMSE").
                                 valid options are "RMSE" (Root mean squared error) or "R2" for explained variance.  


    Returns:
    ----------  
        pval (numpy.ndarray): Updated updated p-value .

        
    # Ref: https://github.com/OHBA-analysis/HMM-MAR/blob/master/utils/testing/permtest_aux.m
    """
    if method == "regression" or method == "one_vs_rest":
        if method_regression== "RMSE":
            # Count every time there is a smaller estimated RMSE (better fit)
            pval[t, :] = np.sum(test_statistic <= test_statistic[0,:], axis=0) / (Nperm + 1)
        else:
            # Count every time there is a higher estimated R2 (better fit)
            pval[t, :] = np.sum(test_statistic >= test_statistic[0,:], axis=0) / (Nperm + 1)
    elif method == "correlation_com":
        # Count every time there is a higher correlation coefficient
        pval[t, :] = np.sum(test_statistic >= test_statistic[0,:], axis=0) / (Nperm + 1)
    return pval


def get_indices_array(idx_data):
    """
    Generates an indices array based on given data indices.

    Parameters:
    --------------
        idx_data (numpy.ndarray): The data indices array.

    Returns:
    ----------  
        idx_array (numpy.ndarray): The generated indices array.
    """
    # Create a copy of idx_data to avoid modifying the original outside the function
    idx_data_copy = np.copy(idx_data)
    
    # Check if any values in column 1 are equal to any values in column 2
    # If equal remove one value from the second column
    if np.any(np.isin(idx_data_copy[:, 0], idx_data_copy[:, 1])):
        idx_data_copy[:, 1] -= 1
    
    # Get an array of indices based on the given idx_data ranges
    max_value = np.max(idx_data_copy[:, 1])
    idx_array = np.zeros(max_value + 1, dtype=int)
    for count, (start, end) in enumerate(idx_data_copy):
        idx_array[start:end + 1] = count
    return idx_array


def within_session_across_trial_permutation(Nperm, R_t, idx_array, trial_timepoints=None):
    """
    Generates permutation matrix of within-session across-trial data based on given indices.

    Parameters:
    --------------
        Nperm (int): The number of permutations.
        R_t (numpy.ndarray): The preprocessed data array.
        idx_array (numpy.ndarray): The indices array.
        trial_timepoints (int): Number of timepoints for each trial (default: None)

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
            if trial_timepoints is None:
                count = 0
                for i in unique_indices:
                    if i ==0:
                        count =count+R_t[idx_array == unique_indices[i], :].shape[0]
                        permute_idx_list[0:count,perm]=np.random.permutation(np.arange(0,count))
                    else:
                        idx_count=R_t[idx_array == unique_indices[i], :].shape[0]
                        count =count+idx_count
                        permute_idx_list[count-idx_count:count,perm]=np.random.permutation(np.arange(count-idx_count,count))
    
            else:
                # Initialize the array to store permutation indices
                permutation_array = []

                # Iterate over unique session indices
                for count, session_idx in enumerate(unique_indices):
                    # Extract data for the current session
                    session_data = R_t[idx_array == session_idx, :]
                    # Get number of data points for each session
                    num_datapoints = session_data.shape[0]

                    # Calculate the number of trials based on trial_timepoints
                    # This step is required because each session can have a different number of trials
                    num_trials = num_datapoints // trial_timepoints

                    # Generate indices for each trial and repeat them based on trial_timepoints
                    idx_trials = np.repeat(np.arange(num_trials), trial_timepoints)

                    # Count unique indices and their occurrences
                    unique_values, value_counts = np.unique(idx_trials, return_counts=True)

                    # Randomly permute the unique indices
                    unique_values_perm = np.random.permutation(unique_values)

                    # Repeat each unique value according to its count in value_counts
                    permuted_array = np.concatenate([np.repeat(value, count) for value, count in zip(unique_values_perm, value_counts)])

                    # Get positions for each unique trial
                    positions_permute = [np.where(permuted_array == i)[0] for i in unique_values]

                    # Extend the permutation_array with adjusted positions
                    permutation_array.extend(np.concatenate(positions_permute) + num_datapoints * count)
                permute_idx_list[:,perm] =np.array(permutation_array)

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


def within_subject_across_sessions_permutation(Nperm, D_t, idx_array):
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

def test_statistic_calculations(Din, Rin, perm, test_statistic, proj, method, method_regression):
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
        method_regression (str): The regression method used for the test statistics (Default:"RMSE").
                                 valid options are "RMSE" (Root mean squared error) or "R2" for explained variance.  

    Returns:
    ----------  
        test_statistic (numpy.ndarray): Updated test_statistic array.
        pval_perms (numpy.ndarray): Updated pval_perms array.
    """
    
    if method == 'regression':
        # Fit the original model 
        beta = proj @ Rin  # Calculate regression_coefficients (beta)
        # Calculate the root mean squared error
        if method_regression=="RMSE":
            # Calculate the root mean squared error
            test_statistic[perm] = np.sqrt(np.mean((Din @ beta - Rin) ** 2, axis=0))
        else:
            # Calculate the predicted values
            predicted_values = Din @ beta
            # Calculate the total sum of squares (SST)
            sst = np.sum((Rin - np.mean(Rin, axis=0))**2, axis=0)
            # Calculate the residual sum of squares (SSR)
            ssr = np.sum((predicted_values - Rin)**2, axis=0)
            # Calculate R^2
            r_squared = 1 - (ssr / sst)
            # Store the R^2 values in the test_statistic array
            test_statistic[perm] = r_squared
    elif method == "correlation_com":
        # Calculate correlation coefficient matrix
        corr_coef = np.corrcoef(Din, Rin, rowvar=False)
        corr_matrix = corr_coef[:Din.shape[1], Din.shape[1]:]
        # Update test_statistic
        test_statistic[perm, :, :] = np.abs(corr_matrix)
    return test_statistic

def get_corr_coef(Din,Rin):
    # Calculate correlation coefficient matrix
    corr_coef = np.corrcoef(Din, Rin, rowvar=False)
    corr_matrix = corr_coef[:Din.shape[1], Din.shape[1]:]
    return corr_matrix


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


def get_concatenate_sessions(D_sessions, R_sessions, idx_sessions):
    """
    Converts a  3D matrix into a 2D matrix by concatenating timepoints of every trial session into a new design matrix.


    Parameters:
    --------------
        D_sessions (numpy.ndarray): Design matrix for each session.
        R_sessions (numpy.ndarray): R  matrix time for each trial.
        idx_sessions (numpy.ndarray): Indices representing the start and end of trials for each session.

    Returns:
    ----------  
        D_con (numpy.ndarray): Concatenated design matrix.
        R_con (numpy.ndarray): Concatenated R matrix.
        idx_sessions_con (numpy.ndarray): Updated indices after concatenation.
    """
    D_con, R_con, idx_sessions_con = [], [], np.zeros_like(idx_sessions)

    for i, (start_idx, end_idx) in enumerate(idx_sessions):
        # Iterate over trials in each session
        for j in range(start_idx, end_idx):
            # Extend data matrix with selected trials
            D_con.extend(D_sessions[:, j, :])
            # Extend time list for each trial
            R_con.extend([R_sessions[j]] * D_sessions.shape[0])

        # Update end index for the concatenated data matrix
        idx_sessions_con[i, 1] = len(D_con)

        if i < len(idx_sessions) - 1:
            # Update start index for the next session if not the last iteration
            idx_sessions_con[i + 1, 0] = idx_sessions_con[i, 1]

    # Convert lists to numpy arrays
    return np.array(D_con), np.array(R_con), idx_sessions_con


def reconstruct_concatenated_design(D_con,D_sessions=None, n_timepoints=None, n_trials=None, n_channels = None):
    """
    Reconstructs the concatenated design matrix to the original session variables.

    Parameters:
    --------------
        D_con (numpy.ndarray): Concatenated design matrix.
        D_sessions (numpy.ndarray, optional): Original design matrix for each session.
        n_timepoints (int, optional): Number of timepoints per trial.
        n_trials (int, optional): Number of trials per session.
        n_channels (int, optional): Number of channels.

    Returns:
    ----------  
        D_reconstruct (numpy.ndarray): Reconstructed design matrix for each session.
    """
    # Input validation and initialization
    if D_sessions is not None and len([arg for arg in [n_timepoints, n_trials, n_channels] if arg is not None]) == 0:
        if not isinstance(D_sessions, np.ndarray) or D_sessions.ndim != 3:
            raise ValueError("Invalid input: D_sessions must be a 3D numpy array.")
        n_timepoints, n_trials, n_channels = D_sessions.shape
        D_reconstruct = np.zeros_like(D_sessions)
    else:
        if None in [n_timepoints, n_trials, n_channels]:
            raise ValueError("Invalid input: n_timepoints, n_trials, and n_channels must be provided if D_sessions is not provided.")
        D_reconstruct = np.zeros((n_timepoints, n_trials, n_channels))
    
    # Check if the shape of D_con matches the expected shape
    if D_con.shape != (n_timepoints * n_trials, n_channels):
        raise ValueError("Invalid input: D_con does not match the expected shape.")

    # Assign values from D_con to D_reconstruct
    for i in range(n_trials):
        start_idx = i * n_timepoints
        end_idx = (i + 1) * n_timepoints
        D_reconstruct[:, i, :] = D_con[start_idx:end_idx, :]
    return D_reconstruct


########################## PALM  Analysis #######################
# This Python code is a translation of part of the PALM (Permutation Analysis of Linear Models) package, originally developed by Anderson M. Winkler. 
# PALM is a tool for permutation-based statistical analysis.

# To learn more about PALM, please visit the official PALM website:
# http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/PALM

# In this Python translation, our primary focus is on accommodating family structure within your dataset. 
# The techniques employed in PALM for managing exchangeability blocks are detailed in the following publication:

# Title: Multi-level block permutation
# Authors: Winkler AM, Webster MA, Vidaurre D, Nichols TE, Smith SM.
# Source: Neuroimage. 2015;123:253-68 (Open Access)
# DOI: 10.1016/j.neuroimage.2015.05.092

# Translated by:
# Nick Y. Larsen
# CFIN / Aarhus University
# Sep/2023 (first version)

# We would like to acknowledge Anderson M. Winkler and all contributors to the PALM package for their valuable work in the field of permutation analysis.


######################### PART 0 - hcp2block #########################################################
def hcp2block(tmp, blocksfile=None, dz2sib=False, ids=None):
    """
    Convert HCP-style twin data into block structure.

    Parameters:
    --------------
        file (str): Path to the input CSV file containing twin data.
        blocksfile (str, optional): Path to save the resulting blocks as a CSV file.
        dz2sib (bool, optional): If True, handle non-monozygotic twins as siblings. Default is False.
        ids (list or array-like, optional): List of subject IDs to include. Default is None.

    Returns:
    ----------  
        tuple: A tuple containing three elements:
            tab (numpy.ndarray): A modified table of twin data.
            B (numpy.ndarray): Block structure representing relationships between subjects.
            famtype (numpy.ndarray): An array indicating the type of each family.
    """
    # # Load data
    # tmp = pd.read_csv(file)
    
    # Handle missing zygosity
    if 'Zygosity' not in tmp.columns:
        tmp['Zygosity'] = np.where(tmp['ZygosityGT'].isna() | (tmp['ZygosityGT'] == ' ') | tmp['ZygosityGT'].isnull(),
                                    tmp['ZygositySR'], tmp['ZygosityGT'])

    # Select columns of interest
    cols = ['Subject', 'Mother_ID', 'Father_ID', 'Zygosity', 'Age_in_Yrs']
    tab = tmp[cols].copy()
    age = tmp['Age_in_Yrs']

    # Remove subjects with missing data
    tab_nan = (tab.iloc[:, 0:3].isna() | tab['Zygosity'].isna() | tab['Age_in_Yrs'].isna()).any(axis=1)
    # Remove missing data
    tab_empty =(tab== ' ').any(axis=1)
    # [i for i, value in enumerate(tab_empty) if value]
    # clean up table
    tab0 = tab_nan+tab_empty
    idstodel = tab.loc[tab0, 'Subject'].tolist()


    if len(idstodel) > 0:
        print(f"These subjects have data missing and will be removed: {idstodel}")

    # Create new table
    tab = tab[~tab0]
    age = age[~tab0]
    N = tab.shape[0] 

    # Handle non-monozygotic twins
    if dz2sib:
        for n in range(N):
            if tab.iloc[n, 3].lower() in ['notmz', 'dz']:
                tab.iloc[n, 3] = 'NotTwin'

    # Convert zygosity strings to identifiers
    sibtype = np.zeros(N, dtype=int)
    for n in range(N):
        if tab.iloc[n, 3].lower() == 'nottwin':
            sibtype[n] = 10
        elif tab.iloc[n, 3].lower() in ['notmz', 'dz']:
            sibtype[n] = 100
        elif tab.iloc[n, 3].lower() == 'mz':
            sibtype[n] = 1000
    tab = tab.iloc[:, :3]

    # Subselect subjects
    if ids is not None:
        if isinstance(ids[0], bool):
            tab = tab[ids]
            sibtype = sibtype[ids]
        else:
            idx = np.isin(tab[:, 0].astype(int), ids)
            if np.any(~idx):
                print(f"These subjects don't exist in the file and will be removed: {ids[~idx]}")
            ids = ids[idx]
            tabnew, sibtypenew, agenew = [], [], []
            for n in ids:
                idx = tab[:, 0].astype(int) == n
                tabnew.append(tab[idx])
                sibtypenew.append(sibtype[idx])
                agenew.append(age[idx])
            tab = np.array(tabnew)
            sibtype = np.array(sibtypenew)
            age = np.array(agenew)
    N = tab.shape[0]

    # Create family IDs
    famid = np.zeros(N, dtype=int)
    U, inv_idx = np.unique(tab.iloc[:, 1:3], axis=0, return_inverse=True)
    for u in range(U.shape[0]):
        uidx = np.all(tab.iloc[:, 1:3] == U[u], axis=1)
        famid[uidx] = u
        
    # Merge families for parents belonging to multiple families
    par = tab.iloc[:, 1:3]
    for p in par.values.flatten():
        pidx = np.any(par == p, axis=1)
        #print(np.unique(famid[pidx]))
        famids = np.unique(famid[pidx])
        for f in famids:
            famid[famid == f] = famids[0]

    # Label each family type
    F = np.unique(famid)
    famtype = np.zeros(N, dtype=int)
    for f in F:
        fidx = famid == f
        famtype[fidx] = np.sum(sibtype[fidx]) + len(np.unique(tab.iloc[fidx, 1:3]))

    #famtype

    # Correction section, because there need to be more than one person to become a twin pair
    # Handle twins with missing pair data
    # Twins which pair data isn't available should be treated as
    # non-twins, so fix and repeat computing the family types
    idx = ((sibtype == 100) & (famtype >= 100) & (famtype <= 199)) | \
            ((sibtype == 1000) & (famtype >= 1000) & (famtype <= 1999))
    sibtype[idx] = 10
    for f in F:
        fidx = famid == f
        famtype[fidx] = np.sum(sibtype[fidx]) + len(np.unique(tab.iloc[fidx, 1:3]))    
        
    # Append the new info to the table.
    tab = np.column_stack((tab, sibtype,famid, famtype))
    tab
    # Combine columns famid, sibtype, and age into a single array
    #combined_array = np.column_stack((famid, sibtype, age))
    # Use lexsort to obtain the indices that would sort the combined array
    # Sort the data and obtain sorting indices
    idx = np.lexsort((age, sibtype, famid))  # This sorts by famid, then sibtype, and finally age
    idxback = np.argsort(idx)
    tab = tab[idx]
    sibtype = sibtype[idx]
    famid = famid[idx]
    famtype = famtype[idx]
    age = age.iloc[idx]    

    # Create blocks for each family
    B = []
    for f in range(len(F)):
        fidx = famid == F[f]
        ft = famtype[np.where(fidx)[0][0]]
        if ft in np.concatenate((np.arange(12, 100, 10), [23, 202, 2002])):
            B.append(np.column_stack((famid[fidx]+1, sibtype[fidx], tab[fidx, 0])))
        else:
            B.append(np.column_stack((-(famid[fidx]+1), sibtype[fidx], tab[fidx, 0])))
            if ft == 33:
                tabx = tab[fidx, 1:3]
                for s in range(tabx.shape[0]):
                    if (np.sum(tabx[:, 0] == tabx[s, 0]) == 2 and \
                        np.sum(tabx[:, 1] == tabx[s, 1]) == 3) or \
                        (np.sum(tabx[:, 0] == tabx[s, 0]) == 3 and \
                        np.sum(tabx[:, 1] == tabx[s, 1]) == 2):
                        B[-1][s, 1] += 1
            elif ft == 53:
                tabx = tab[fidx, 1:3]
                for s in range(tabx.shape[0]):
                    if (np.sum(tabx[:, 0] == tabx[s, 0]) == 3 and \
                        np.sum(tabx[:, 1] == tabx[s, 1]) == 5) or \
                        (np.sum(tabx[:, 0] == tabx[s, 0]) == 5 and \
                        np.sum(tabx[:, 1] == tabx[s, 1]) == 3):
                        B[-1][s, 1] += 1
        
            elif ft == 234:
                tabx = tab[fidx, 1:3]
                for s in range(tabx.shape[0]):
                    if (np.sum(tabx[:, 0] == tabx[s, 0]) == 1 and \
                        np.sum(tabx[:, 1] == tabx[s, 1]) == 3) or \
                        (np.sum(tabx[:, 0] == tabx[s, 0]) == 3 and \
                        np.sum(tabx[:, 1] == tabx[s, 1]) == 1):
                        B[-1][s, 1] += 1
            elif ft == 54:
                tabx = tab[fidx, 1:3]
                for s in range(tabx.shape[0]):
                    if np.sum(tabx[:, 0] == tabx[s, 0]) == 4 and np.sum(tabx[:, 1] == tabx[s, 1]) == 2:
                        B[-1][s, 1] += 1
                    elif np.sum(tabx[:, 0] == tabx[s, 0]) == 1 and np.sum(tabx[:, 1] == tabx[s, 1]) == 3:
                        B[-1][s, 1] -= 1                          
            
            
            elif ft == 34:
                tabx = tab[fidx, 1:3]
                for s in range(tabx.shape[0]):
                    if (np.sum(tabx[:, 0] == tabx[s, 0]) == 2 and
                        np.sum(tabx[:, 1] == tabx[s, 1]) == 2):
                        B[f][s, 1] += 1
                        famtype[fidx] = 39  
                                
            elif ft == 43:
                tabx = tab[fidx, 1:3]
                k = 0
                for s in range(tabx.shape[0]):
                    if tabx[s, 0] == tabx[0, 0] and tabx[s, 1] == tabx[0, 1]:
                        B[-1][s, 1] += 1
                        k += 1
                if k == 2:
                    famtype[fidx] = 49
                    B[-1][:, 0] = -B[-1][:, 0]
            elif ft == 44:
                tabx = tab[fidx, 1:3]
                for s in range(tabx.shape[0]):
                    if np.sum(tabx[:, 0] == tabx[s, 0]) == 4 and np.sum(tabx[:, 1] == tabx[s, 1]) == 2:
                        B[-1][s, 1] += 1        
                        
            elif ft == 223:
                sibx = sibtype[fidx]
                B[-1][sibx == 10, 1] = -B[-1][sibx == 10, 1]
            elif ft == 302:
                famtype[fidx] = 212
                tmpage = age[fidx]
                if tmpage.iloc[0] == tmpage.iloc[1]:
                    B[-1][2, 1] = 10
                elif tmpage.iloc[0] == tmpage.iloc[2]:
                    B[-1][1, 1] = 10
                elif tmpage.iloc[1] == tmpage.iloc[2]:
                    B[-1][0, 1] = 10
            elif ft == 313 or ft == 314:
                famtype[fidx] = ft - 100 + 10
                if (famtype[fidx] == 223).all():
                    # Identify the elements that are equal to 223 and need changing
                    to_change = (famtype == 223) & fidx
                    # Update the values at the selected indices
                    famtype[fidx] = 229

                tmpage = age[fidx]
                #didx = np.where(B[-1][:, 1] == 100)[0]
                didx = np.where(B[-1][:, 1] == 100)[0]
                if tmpage.iloc[didx[0]] == tmpage.iloc[didx[1]]:
                    B[-1][didx[2], 1] = 10
                elif tmpage.iloc[didx[0]] == tmpage.iloc[didx[2]]:
                    B[-1][didx[1], 1] = 10
                elif tmpage.iloc[didx[1]] == tmpage.iloc[didx[2]]:
                    B[-1][didx[0], 1] = 10  
                    
            # Additional case: ft == 2023
            elif ft == 2023:
                tabx = tab[fidx, 1:3]
                for s in range(tabx.shape[0]):
                    if np.sum(tabx[:, 0] == tabx[s, 0]) == 4 and \
                    (np.sum(tabx[:, 1] == tabx[s, 1]) == 1 or np.sum(tabx[:, 1] == tabx[s, 1]) == 3):
                        famtype[fidx] = 2029
                        if B[f][s, 1] == 10:
                            B[f][s, 1] = -B[f][s, 1]
                            

    B = np.hstack((-np.ones((N, 1),dtype=int), famtype.reshape(-1, 1), np.concatenate(B, axis=0)))

    # Sort back to the original order
    B = B[idxback, :]
    tab = tab[idxback, :]
    tab[:, 5] = B[:, 1]


    blocksfile = "EB.csv"
    if blocksfile is not None and isinstance(blocksfile, str):
        # Save B as a CSV file with integer precision
        np.savetxt(blocksfile, B, delimiter=',', fmt='%d')
        
    # Return tab, B, and famtype
    return tab, B, famtype

######################### PART 1 - Reindix #########################################################
import numpy as np

def renumber(B):
    
    """
    Renumber the elements in a 2D numpy array B, preserving their order within distinct blocks.

    This function renumbers the elements in the input array B based on distinct values in its first column.
    Each distinct value represents a block, and the elements within each block are renumbered sequentially,
    while preserving the relative order of elements within each block.

    Parameters:
    --------------
    B (numpy.ndarray): The 2D input array to be renumbered.

    Returns:
    ----------  
    tuple: A tuple containing:
        - Br (numpy.ndarray): The renumbered array, where elements are renumbered within blocks.
        - addcol (bool): A boolean indicating whether a column was added during renumbering.

    """

    # Extract the first column of the input array B
    B1 = B[:, 0]
    # Find the unique values in B1 and store them in U
    U = np.unique(B1)
    # Create a boolean array to keep track of added columns
    addcolvec = np.zeros_like(U, dtype=bool)
    # Get the number of unique values
    nU = U.shape[0]
    # Create an empty array Br with the same shape as B
    Br = np.zeros_like(B)
    
    # Loop through unique values in B1
    for u in range(nU):
         # Find indices where B1 is equal to the current unique value U[u]
        idx = B1 == U[u]
        # Renumber the corresponding rows in Br based on the index
        Br[idx, 0] = (u + 1) * np.sign(U[u])
        
        # Check if B has more than one column
        if B.shape[1] > 1:
            # Recursively call renumber for the remaining columns and update addcolvec
            Br[idx, 1:], addcolvec[u] = renumber(B[idx, 1:])
        elif np.sum(idx) > 1:
             # If there's only one column and more than one matching row, set addcol to True
            addcol = True
            Br[idx] = -np.abs(B[idx])
        else:
            addcol = False
    # Check if B has more than one column and if any columns were added
    if B.shape[1] > 1:
        addcol = np.any(addcolvec)
    # Return the renumbered array Br and the addcol flag
    return Br, addcol

def palm_reindex(B, meth='fixleaves'):
    """
    Reindex a 2D numpy array using different procedures while preserving block structure.

    This function reorders the elements of a 2D numpy array `B` by applying one of several reindexing methods.
    The primary goal of reindexing is to assign new values to elements in such a way that they are organized
    in a desired order or structure.

    Parameters:
    --------------
    B (numpy.ndarray): The 2D input array to be reindexed.
    meth (str, optional): The reindexing method to be applied. It can take one of the following values:
        - 'fixleaves': This method reindexes the input array by preserving the order of unique values in the
          first column and recursively reindexes the remaining columns. It is well-suited for hierarchical
          data where the first column represents levels or leaves.
        - 'continuous': This method reindexes the input array by assigning new values to elements in a
          continuous, non-overlapping manner within each column. It is useful for continuous data or when
          preserving the order of unique values is not a requirement.
        - 'restart': This method reindexes the input array by restarting the numbering from 1 for each block
          of unique values in the first column. It is suitable for data that naturally breaks into distinct
          segments or blocks.
        - 'mixed': This method combines both the 'fixleaves' and 'continuous' reindexing methods. It reindexes
          the first columns using 'fixleaves' and the remaining columns using 'continuous', creating a mixed
          reindexing scheme.

    Returns:
    ----------  
    numpy.ndarray: The reindexed array, preserving the block structure based on the chosen method.


    Raises:
    ValueError: If the `meth` parameter is not one of the valid reindexing methods.
    """

    # Convert meth to lowercase
    meth = meth.lower()
    
    # Initialize the output array Br with zeros
    Br = np.zeros_like(B)
    
    if meth == 'continuous':
        # Find unique values in the first column of B
        U = np.unique(B[:, 0])
        
        # Renumber the first column based on unique values
        for u in range(U.shape[0]):
            idx = B[:, 0] == U[u]
            Br[idx, 0] = (u + 1) * np.sign(U[u])
        
        # Loop through columns starting from the 2nd column    
        for b in range(1, B.shape[1]):  # From the 2nd column onwards
            Bb = B[:, b]
            Bp = Br[:, b - 1]  # Previous column
            # Find unique values in the previous column
            Up = np.unique(Bp)
            cnt = 1
            
            # Renumber elements within blocks based on unique values
            for up in range(Up.shape[0]):
                idxp = Bp == Up[up]
                U = np.unique(Bb[idxp])
                
                # Renumber elements within the block
                for u in range(U.shape[0]):
                    idx = np.logical_and(Bb == U[u], idxp)
                    Br[idx, b] = cnt * np.sign(U[u])
                    cnt += 1
                    
    elif meth == 'restart':
        # Renumber each block separately, starting from 1
        Br, _ = renumber(B)
        
    elif meth == 'mixed':
        # Mix both 'restart' and 'continuous' methods
        Ba, _ = palm_reindex(B, 'restart')
        Bb, _ = palm_reindex(B, 'continuous')
        Br = np.hstack((Ba[:, :-1], Bb[:, -1:]))
        
    elif meth=="fixleaves":
        # Reindex using 'fixleaves' method as defined in the renumber function

        B1 = B[:, 0]
        U = np.unique(B1)
        addcolvec = np.zeros_like(U, dtype=bool)
        nU = U.shape[0]
        Br = np.zeros_like(B)

        for u in range(nU):
            idx = B1 == U[u]
            Br[idx, 0] = (u + 1) * np.sign(U[u])

            if B.shape[1] > 1:
                Br[idx, 1:], addcolvec[u] = renumber(B[idx, 1:])
            elif np.sum(idx) > 1:
                addcol = True
                Br[idx] = -np.abs(B[idx])
            else:
                addcol = False

        if B.shape[1] > 1:
            addcol = np.any(addcolvec)
        
        if addcol:
            # Add a column of sequential numbers to Br and reindex
            col = np.arange(1, Br.shape[0] + 1).reshape(-1, 1)
            Br = np.hstack((Br, col))
            Br, _ = renumber(Br)
            
    else:
        # Raise a ValueError for an unknown method
        raise ValueError(f'Unknown method: {meth}')
    # Return the reindexed array Br
    return Br


######################### PART 2 - PALMTREE #########################################################


def palm_permtree(Ptree, nP, CMC=False, maxP=None):
    """
    Generate permutations of a given palm tree structure.

    This function generates permutations of a palm tree structure represented by Ptree. Permutations are created by
    shuffling the branches of the palm tree. The number of permutations is controlled by the 'nP' parameter.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure to be permuted.
    nP (int): The number of permutations to generate.
    CMC (bool, optional): Whether to use Conditional Monte Carlo (CMC) method for permutation.
                          Defaults to False.
    maxP (int, optional): The maximum number of permutations allowed. If not provided, it is calculated automatically.

    Returns:
    ----------  
    numpy.ndarray: An array representing the permutations. Each row corresponds to a permutation, with the first
                   column always representing the identity permutation.

    Note:
    - If 'CMC' is False and 'nP' is greater than 'maxP' / 2, a warning message is displayed, as it may take a
      considerable amount of time to find non-repeated permutations.
    - The function utilizes the 'pickperm' and 'randomperm' helper functions for the permutation process.
    """
    
    if nP == 1 and not maxP:
        # Calculate the maximum number of permutations if not provided
        maxP = palm_maxshuf(Ptree, 'perms')
        if nP > maxP:
            nP = maxP  # The cap is only imposed if maxP isn't supplied
    
    
    # Permutation #1 is no permutation, regardless.
    P = pickperm(Ptree, np.array([], dtype=int))
    P = np.hstack((P.reshape(-1,1), np.zeros((P.shape[0], nP - 1), dtype=int)))
    
    
    # Generate all other permutations up to nP
    if nP == 1:
        pass
    elif CMC or nP > maxP:
        for p in range(2, nP + 1):
            Ptree_perm = copy.deepcopy(Ptree)
            Ptree_perm = randomperm(Ptree_perm)
            P[:, p - 1] = pickperm(Ptree_perm, [])
    else:
        if nP > maxP / 2:
            # Inform the user about the potentially long runtime
            print(f'The maximum number of permutations ({maxP}) is not much larger than\n'
                  f'the number you chose to run ({nP}). This means it may take a while (from\n'
                  f'a few seconds to several minutes) to find non-repeated permutations.\n'
                  'Consider instead running exhaustively all possible permutations. It may be faster.')
        for p in range(1, nP):
            whiletest = True
            while whiletest:
                Ptree_perm = copy.deepcopy(Ptree)
                Ptree_perm = randomperm(Ptree_perm)
                P[:, p] = pickperm(Ptree_perm, [])
                
                whiletest = np.any(np.all(P[:, :p] == P[:, p][:, np.newaxis], axis=0))
    
    # The grouping into branches screws up the original order, which
    # can be restored by noting that the 1st permutation is always
    # the identity, so with indices 1:N. This same variable idx can
    # be used to likewise fix the order of sign-flips (separate func).
    idx = np.argsort(P[:, 0])
    P = P[idx, :]
    
    return P

def pickperm(Ptree, P):
    """
    Extract a permutation from a palm tree structure.

    This function extracts a permutation from a given palm tree structure. It does not perform the permutation
    but returns the indices representing the already permuted tree.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure.
    P (numpy.ndarray): The current state of the permutation.

    Returns:
    ----------  
    numpy.ndarray: An array of indices representing the permutation of the palm tree structure.
    """
    # Check if Ptree is a list and has three elements, then recursively call pickperm on the third element
    if isinstance(Ptree,list):
        if len(Ptree) == 3:
            P = pickperm(Ptree[2],P)
    # Check if the shape of Ptree is (N, 3), where N is the number of branches
    elif Ptree.shape[1] ==3:
        nU = Ptree.shape[0]
        # Loop through each branch
        for u in range(nU):
            # Recursively call pickperm on the third element of the branch
            P = pickperm(Ptree[u][2],P)
    # Check if the shape of Ptree is (N, 1)
    elif Ptree.shape[1] ==1:
        nU = Ptree.shape[0]
        # Loop through each branch
        for u in range(nU):
            # Concatenate the first element of the branch (a submatrix) to P
            P = np.concatenate((P, Ptree[u][0]), axis=None)
    return P

def randomperm(Ptree_perm):
    """
    Create a random permutation of a palm tree structure.

    This function generates a random permutation of a given palm tree structure by shuffling its branches.

    Parameters:
    --------------
    Ptree_perm (list or numpy.ndarray): The palm tree structure to be permuted.

    Returns:
    ----------  
    list: The randomly permuted palm tree structure.
    """
    # Check if Ptree_perm is a list and has three elements, then recursively call randomperm on the third element
    if isinstance(Ptree_perm,list):
        if len(Ptree_perm) == 3:
            Ptree_perm = randomperm(Ptree_perm[2])
            
        
    # Get the number of branches in Ptree_perm
    nU = Ptree_perm.shape[0]
    # Loop through each branch
    for u in range(nU):
        # Check if the first element of the branch is a single value and not NaN
        if is_single_value(Ptree_perm[u][0]):
            if not np.isnan(Ptree_perm[u][0]):
                tmp = 1
                # Shuffle the first element of the branch
                np.random.shuffle(Ptree_perm[u][0])
                # Check if tmp is not equal to the first element of the branch
                if np.any(tmp != Ptree_perm[u][0][0]):
                     # Rearrange the third element of the branch based on the shuffled indices
                    Ptree_perm[u][2][Ptree_perm[u][0][:, 2].astype(int) - 1, :] 
        # Check if the first element of the branch is a list with three elements            
        elif isinstance(Ptree_perm[u][0],list) and len(Ptree_perm[u][0])==3:
            tmp = 1
            # Shuffle the first element of the branch
            np.random.shuffle(Ptree_perm[u][0])
            # Check if tmp is not equal to the first element of the branch
            if np.any(tmp != Ptree_perm[u][0][0]):
                # Rearrange the third element of the branch based on the shuffled indices
                Ptree_perm[u][2][Ptree_perm[u][0][:, 2].astype(int) - 1, :]
            
        else:
            tmp = np.arange(1,len(Ptree_perm[u][0][:,0])+1,dtype=int)
            # Shuffle the first element of the branch
            np.random.shuffle(Ptree_perm[u][0])
            
            # Check if tmp is not equal to the first element of the branch
            if np.any(tmp != Ptree_perm[u][0][:, 0]):
                # Rearrange the third element of the branch based on the shuffled indices
                Ptree_perm[u][2] =Ptree_perm[u][2][Ptree_perm[u][0][:, 2].astype(int) - 1, :]      
            
        # Make sure the next isn't the last level.
        if Ptree_perm[u][2].shape[1] > 1:
            # Recursively call randomperm on the third element of the branch
            Ptree_perm[u][2] = randomperm(Ptree_perm[u][2])  
            
    return Ptree_perm


######################### PART 3.1 - Permute PTREE #########################################################
#### Permutation functions
import numpy as np

def palm_maxshuf(Ptree, stype='perms', uselog=False):
    """
    Calculate the maximum number of shufflings (permutations or sign-flips) for a given palm tree structure.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure.
    stype (str, optional): The type of shuffling to calculate ('perms' for permutations by default).
    uselog (bool, optional): A flag indicating whether to calculate using logarithmic values (defaults to False).

    Returns:
    ----------  
    int: The maximum number of shufflings (permutations or sign-flips) based on the specified criteria.
    """
    
    # Calculate the maximum number of shufflings based on user-defined options
    if uselog:
        if stype == 'perms':
            maxb = lmaxpermnode(Ptree, 0)
    
    else:
        if stype == 'perms':
            maxb = maxpermnode(Ptree, 1)
    return maxb

def maxpermnode(Ptree, np):
    """
    Calculate the maximum number of permutations within a palm tree node.

    This function recursively calculates the maximum number of permutations within a palm tree node.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure.
    np (int): The current number of permutations (initialized to 1).

    Returns:
    ----------  
    int: The maximum number of permutations within the node.
    """
    for u in range(len(Ptree)):
        n_p = n_p * seq2np(Ptree[u][0][:, 0])
        if len(Ptree[u][2][0]) > 1:
            n_p = maxpermnode(Ptree[u][2], np)
    return n_p

def seq2np(S):
    """
    Calculate the number of permutations for a given sequence.

    This function calculates the number of permutations for a given sequence.

    Parameters:
    --------------
    S (numpy.ndarray): The input sequence.

    Returns:
    ----------  
    int: The number of permutations for the sequence.
    """
    U, cnt = np.unique(S, return_counts=True)
    n_p = np.math.factorial(len(S)) / np.prod(np.math.factorial(cnt))
    return n_p

def maxflipnode(Ptree, ns):
    """
    Calculate the maximum number of sign-flips within a palm tree node.

    This function recursively calculates the maximum number of sign-flips within a palm tree node.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure.
    ns (int): The current number of sign-flips (initialized to 1).

    Returns:
    ----------  
    int: The maximum number of sign-flips within the node.
    """
    for u in range(len(Ptree)):
        if len(Ptree[u][2][0]) > 1:
            ns = maxflipnode(Ptree[u][2], ns)
        ns = ns * (2 ** len(Ptree[u][1]))
    return ns

def lmaxpermnode(Ptree, n_p):
    """
    Calculate the logarithm of the maximum number of permutations within a palm tree node.

    This function calculates the logarithm of the maximum number of permutations within a palm tree node.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure.
    n_p (int): The current logarithm of permutations (initialized to 0).

    Returns:
    ----------  
    int: The logarithm of the maximum number of permutations within the node.
    """
    if isinstance(Ptree,list):
        n_p = n_p + lseq2np(Ptree[0])
        if Ptree[2].shape[1] > 1:
            n_p = lmaxpermnode(Ptree[2], n_p)
    else:
        for u in range(Ptree.shape[0]):
            if isinstance(Ptree[u][0],list):
                n_p = n_p + lseq2np(Ptree[u][0][0])
                if Ptree[u][2].shape[1] > 1:
                    #n_p = lmaxpermnode(Ptree[u][2][0][2], n_p)
                    n_p = lmaxpermnode(Ptree[u][2], n_p)
            elif is_single_value(Ptree[u][0]):
                n_p = n_p + lseq2np(Ptree[u][0])
                
                if len(Ptree[u]) > 2 and Ptree[u][2].shape[1] > 1:
                    n_p = lmaxpermnode(Ptree[u][2], n_p) 
            else:     
                n_p = n_p + lseq2np(Ptree[u][0][:,0])
                
                if len(Ptree[u]) > 2 and Ptree[u][2].shape[1] > 1:
                    n_p = lmaxpermnode(Ptree[u][2], n_p) 
    
    return n_p

def lseq2np(S):
    """
    Calculate the logarithm of the number of permutations for a given sequence.

    This function calculates the logarithm of the number of permutations for a given sequence.

    Parameters:
    --------------
    S (numpy.ndarray): The input sequence.

    Returns:
    ----------  
    int: The logarithm of the number of permutations for the sequence.
    """
    if is_single_value(S):
        nS = 1
        if np.isnan(S):
            U = np.nan
            cnt = 0
        else:
            U, cnt = np.unique(S, return_counts=True)
            
    else: 
        nS = len(S)
        U, cnt = np.unique(S, return_counts=True)
        
    #lfac=palm_factorial(nS)
    lfac=palm_factorial()
    n_p = lfac[nS] - np.sum(lfac[cnt])
    return n_p

def lmaxflipnode(Ptree, ns):
    """
    Calculate the logarithm of the maximum number of sign-flips within a palm tree node.

    This function calculates the logarithm of the maximum number of sign-flips within a palm tree node.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure.
    ns (int): The current logarithm of sign-flips (initialized to 0).

    Returns:
    ----------  
    int: The logarithm of the maximum number of sign-flips within the node.
    """
    for u in range(len(Ptree)):
        if len(Ptree[u][2][0]) > 1:
            ns = lmaxflipnode(Ptree[u][2], ns)
        ns = ns + len(Ptree[u][1])
    return ns

def is_single_value(variable):
    """
    Check if an array contains a singlevalue.

    This function checks if an array contains a singlevalue.

    Parameters:
    --------------
    arr (numpy.ndarray or list): The array to be checked.

    Returns:
    ----------  
    bool: True if the array contains a single value, False otherwise.
    """
    return isinstance(variable, (int, float, complex))

def palm_factorial(N=101):
    """
    Calculate logarithmically scaled factorials up to a given number.

    This function precomputes logarithmically scaled factorials up to a specified number.

    Parameters:
    --------------
    N (int, optional): The maximum number for which to precompute factorials (defaults to 101).

    Returns:
    ----------  
    numpy.ndarray: An array of precomputed logarithmically scaled factorials.
    """
    if N == 1:
        N = 101
    # Initialize the lf array with zeros
    lf = np.zeros(N+1)

    # Calculate log(factorial) values
    for n in range(1, N+1):
        lf[n] = np.log(n) + lf[n-1]

    return lf

######################### PART 3.2 - PALMTREE #########################################################
#### Permute PALM tree



def palm_permtree(Ptree, nP, CMC=False, maxP=None):
    """
    Generate permutations of a given palm tree structure.

    This function generates permutations of a palm tree structure represented by Ptree. Permutations are created by
    shuffling the branches of the palm tree. The number of permutations is controlled by the 'nP' parameter.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure to be permuted.
    nP (int): The number of permutations to generate.
    CMC (bool, optional): Whether to use Conditional Monte Carlo (CMC) method for permutation.
                          Defaults to False.
    maxP (int, optional): The maximum number of permutations allowed. If not provided, it is calculated automatically.

    Returns:
    ----------  
    numpy.ndarray: An array representing the permutations. Each row corresponds to a permutation, with the first
                   column always representing the identity permutation.

    Note:
    - If 'CMC' is False and 'nP' is greater than 'maxP' / 2, a warning message is displayed, as it may take a
      considerable amount of time to find non-repeated permutations.
    - The function utilizes the 'pickperm' and 'randomperm' helper functions for the permutation process.
    """
    
    if nP == 1 and not maxP:
        # Calculate the maximum number of permutations if not provided
        maxP = palm_maxshuf(Ptree, 'perms')
        if nP > maxP:
            nP = maxP  # The cap is only imposed if maxP isn't supplied
    
    
    # Permutation #1 is no permutation, regardless.
    P = pickperm(Ptree, np.array([], dtype=int))
    P = np.hstack((P.reshape(-1,1), np.zeros((P.shape[0], nP - 1), dtype=int)))
    
    
    # Generate all other permutations up to nP
    if nP == 1:
        pass
    elif CMC or nP > maxP:
        for p in range(2, nP + 1):
            Ptree_perm = copy.deepcopy(Ptree)
            Ptree_perm = randomperm(Ptree_perm)
            P[:, p - 1] = pickperm(Ptree_perm, [])
    else:
        if nP > maxP / 2:
            # Inform the user about the potentially long runtime
            print(f'The maximum number of permutations ({maxP}) is not much larger than\n'
                  f'the number you chose to run ({nP}). This means it may take a while (from\n'
                  f'a few seconds to several minutes) to find non-repeated permutations.\n'
                  'Consider instead running exhaustively all possible permutations. It may be faster.')
        for p in range(1, nP):
            whiletest = True
            while whiletest:
                Ptree_perm = copy.deepcopy(Ptree)
                Ptree_perm = randomperm(Ptree_perm)
                P[:, p] = pickperm(Ptree_perm, [])
                
                whiletest = np.any(np.all(P[:, :p] == P[:, p][:, np.newaxis], axis=0))
    
    # The grouping into branches screws up the original order, which
    # can be restored by noting that the 1st permutation is always
    # the identity, so with indices 1:N. This same variable idx can
    # be used to likewise fix the order of sign-flips (separate func).
    idx = np.argsort(P[:, 0])
    P = P[idx, :]
    
    return P

def pickperm(Ptree, P):
    """
    Extract a permutation from a palm tree structure.

    This function extracts a permutation from a given palm tree structure. It does not perform the permutation
    but returns the indices representing the already permuted tree.

    Parameters:
    --------------
    Ptree (list or numpy.ndarray): The palm tree structure.
    P (numpy.ndarray): The current state of the permutation.

    Returns:
    ----------  
    numpy.ndarray: An array of indices representing the permutation of the palm tree structure.
    """
    # Check if Ptree is a list and has three elements, then recursively call pickperm on the third element
    if isinstance(Ptree,list):
        if len(Ptree) == 3:
            P = pickperm(Ptree[2],P)
    # Check if the shape of Ptree is (N, 3), where N is the number of branches
    elif Ptree.shape[1] ==3:
        nU = Ptree.shape[0]
        # Loop through each branch
        for u in range(nU):
            # Recursively call pickperm on the third element of the branch
            P = pickperm(Ptree[u][2],P)
    # Check if the shape of Ptree is (N, 1)
    elif Ptree.shape[1] ==1:
        nU = Ptree.shape[0]
        # Loop through each branch
        for u in range(nU):
            # Concatenate the first element of the branch (a submatrix) to P
            P = np.concatenate((P, Ptree[u][0]), axis=None)
    return P

def randomperm(Ptree_perm):
    """
    Create a random permutation of a palm tree structure.

    This function generates a random permutation of a given palm tree structure by shuffling its branches.

    Parameters:
    --------------
    Ptree_perm (list or numpy.ndarray): The palm tree structure to be permuted.

    Returns:
    ----------  
    list: The randomly permuted palm tree structure.
    """
    # Check if Ptree_perm is a list and has three elements, then recursively call randomperm on the third element
    if isinstance(Ptree_perm,list):
        if len(Ptree_perm) == 3:
            Ptree_perm = randomperm(Ptree_perm[2])
            
        
    # Get the number of branches in Ptree_perm
    nU = Ptree_perm.shape[0]
    # Loop through each branch
    for u in range(nU):
        # Check if the first element of the branch is a single value and not NaN
        if is_single_value(Ptree_perm[u][0]):
            if not np.isnan(Ptree_perm[u][0]):
                tmp = 1
                # Shuffle the first element of the branch
                np.random.shuffle(Ptree_perm[u][0])
                # Check if tmp is not equal to the first element of the branch
                if np.any(tmp != Ptree_perm[u][0][0]):
                     # Rearrange the third element of the branch based on the shuffled indices
                    Ptree_perm[u][2][Ptree_perm[u][0][:, 2].astype(int) - 1, :] 
        # Check if the first element of the branch is a list with three elements            
        elif isinstance(Ptree_perm[u][0],list) and len(Ptree_perm[u][0])==3:
            tmp = 1
            # Shuffle the first element of the branch
            np.random.shuffle(Ptree_perm[u][0])
            # Check if tmp is not equal to the first element of the branch
            if np.any(tmp != Ptree_perm[u][0][0]):
                # Rearrange the third element of the branch based on the shuffled indices
                Ptree_perm[u][2][Ptree_perm[u][0][:, 2].astype(int) - 1, :]
            
        else:
            tmp = np.arange(1,len(Ptree_perm[u][0][:,0])+1,dtype=int)
            # Shuffle the first element of the branch
            np.random.shuffle(Ptree_perm[u][0])
            
            # Check if tmp is not equal to the first element of the branch
            if np.any(tmp != Ptree_perm[u][0][:, 0]):
                # Rearrange the third element of the branch based on the shuffled indices
                Ptree_perm[u][2] =Ptree_perm[u][2][Ptree_perm[u][0][:, 2].astype(int) - 1, :]      
            
        # Make sure the next isn't the last level.
        if Ptree_perm[u][2].shape[1] > 1:
            # Recursively call randomperm on the third element of the branch
            Ptree_perm[u][2] = randomperm(Ptree_perm[u][2])  
            
    return Ptree_perm


######################### PART 3.3 - Permute PTREE #########################################################
###### Function that integrates the above functions in part 3
import warnings
import numpy as np


def palm_shuftree(Ptree,nP,CMC= False,EE = True):
    """
    Generate a set of shufflings (permutations or sign-flips) for a given palm tree structure.

    Parameters:
    --------------
    Ptree (list): The palm tree structure.
    nP (int): The number of permutations to generate.

    CMC (bool, optional): A flag indicating whether to use the Conditional Monte Carlo method (CMC).
                          Defaults to False.
    EE (bool, optional): A flag indicating whether to assume exchangeable errors, which allows permutation.
                        Defaults to True.

    Returns:
    ----------  
    list: A list containing the generated shufflings (permutations).
    """ 
    
    # Maximum number of shufflings (perms, sign-flips, or both)
    maxP = 1
    maxS = 1
    if EE:
        lmaxP = palm_maxshuf(Ptree, 'perms', True)
        maxP = np.exp(lmaxP)

        if np.isinf(maxP):
            print('Number of possible permutations is exp({}).'.format(lmaxP))
        else:
            print('Number of possible permutations is {}.'.format(maxP))


    maxB = maxP * maxS

    # String for the screen output below
    whatshuf = 'permutations only'
    whatshuf2 = 'perms'

    # Generate the Pset and Sset
    Pset = []
    Sset = []

    if nP == 0 or nP >= maxB:
        # Run exhaustively if the user requests too many permutations.
        # Note that here CMC is irrelevant.
        print('Generating {} shufflings ({}).'.format(maxB, whatshuf))
        if EE:
            Pset = palm_permtree(Ptree, int(round(maxP)) if maxP != np.inf else maxP, [], int(round(maxP)) if maxP != np.inf else maxP)


    elif nP < maxB:
        # Or use a subset of possible permutations.
        print('Generating {} shufflings ({}).'.format(nP, whatshuf))
        if EE:
            if nP >= maxP:
                Pset = palm_permtree(Ptree, int(round(maxP)) if maxP != np.inf else maxP, CMC, int(round(maxP)) if maxP != np.inf else maxP)
                
            else:
                Pset = palm_permtree(Ptree, nP, CMC, int(round(maxP)) if maxP != np.inf else maxP)

    return Pset
######################### PART 4 - quick_perm #########################################################
def palm_quickperms(EB, M=None, nP=1000, CMC=False, EE=True):
    """
    Generate a set of permutations for a given input matrix using palm methods.

    Parameters:
    --------------
    EB (numpy.ndarray): Block structure representing relationships between subjects.
    M (numpy.ndarray, optional): The matrix of attributes, which is not typically required.
                                Defaults to None.
    nP (int): The number of permutations to generate.
    CMC (bool, optional): A flag indicating whether to use the Conditional Monte Carlo method (CMC).
                          Defaults to False.
    EE (bool, optional): A flag indicating whether to assume exchangeable errors, which allows permutation.
                        Defaults to True.

    Returns:
    ----------  
    list: A list containing the generated permutations.
    """
    
    # Reindex the input matrix for palm methods with 'fixleaves'
    EB2 = palm_reindex(EB, 'fixleaves')
    
    # Generate a palm tree structure from the reindexed matrix
    Ptree = palm_tree(EB2)
    
    # Generate a set of shufflings (permutations) based on the palm tree structure
    Pset = palm_shuftree(Ptree, nP, CMC, EE)
    # Need to change the number so the index startes from 0
    # Pset = Pset-1
    return Pset

######################### Helper functions #########################################################

def palm_maxshuf(Ptree, stype='perms', uselog=False):
    """
    Calculate the maximum number of shufflings (permutations or sign-flips) for a given palm tree structure.

    Parameters:
    Ptree (list or numpy.ndarray): The palm tree structure.
    stype (str, optional): The type of shuffling to calculate ('perms' for permutations by default).
    uselog (bool, optional): A flag indicating whether to calculate using logarithmic values (defaults to False).

    Returns:
    int: The maximum number of shufflings (permutations or sign-flips) based on the specified criteria.
    """
    
    # Calculate the maximum number of shufflings based on user-defined options
    if uselog:
        if stype == 'perms':
            maxb = lmaxpermnode(Ptree, 0)
    
    else:
        if stype == 'perms':
            maxb = maxpermnode(Ptree, 1)
    return maxb

def maxpermnode(Ptree, np):
    """
    Calculate the maximum number of permutations within a palm tree node.

    This function recursively calculates the maximum number of permutations within a palm tree node.

    Parameters:
    Ptree (list or numpy.ndarray): The palm tree structure.
    np (int): The current number of permutations (initialized to 1).

    Returns:
    int: The maximum number of permutations within the node.
    """
    for u in range(len(Ptree)):
        n_p = n_p * seq2np(Ptree[u][0][:, 0])
        if len(Ptree[u][2][0]) > 1:
            n_p = maxpermnode(Ptree[u][2], np)
    return n_p

def seq2np(S):
    """
    Calculate the number of permutations for a given sequence.

    This function calculates the number of permutations for a given sequence.

    Parameters:
    S (numpy.ndarray): The input sequence.

    Returns:
    int: The number of permutations for the sequence.
    """
    U, cnt = np.unique(S, return_counts=True)
    n_p = np.math.factorial(len(S)) / np.prod(np.math.factorial(cnt))
    return n_p

def maxflipnode(Ptree, ns):
    """
    Calculate the maximum number of sign-flips within a palm tree node.

    This function recursively calculates the maximum number of sign-flips within a palm tree node.

    Parameters:
    Ptree (list or numpy.ndarray): The palm tree structure.
    ns (int): The current number of sign-flips (initialized to 1).

    Returns:
    int: The maximum number of sign-flips within the node.
    """
    for u in range(len(Ptree)):
        if len(Ptree[u][2][0]) > 1:
            ns = maxflipnode(Ptree[u][2], ns)
        ns = ns * (2 ** len(Ptree[u][1]))
    return ns

def lmaxpermnode(Ptree, n_p):
    """
    Calculate the logarithm of the maximum number of permutations within a palm tree node.

    This function calculates the logarithm of the maximum number of permutations within a palm tree node.

    Parameters:
    Ptree (list or numpy.ndarray): The palm tree structure.
    n_p (int): The current logarithm of permutations (initialized to 0).

    Returns:
    int: The logarithm of the maximum number of permutations within the node.
    """
    if isinstance(Ptree,list):
        n_p = n_p + lseq2np(Ptree[0])
        if Ptree[2].shape[1] > 1:
            n_p = lmaxpermnode(Ptree[2], n_p)
    else:
        for u in range(Ptree.shape[0]):
            if isinstance(Ptree[u][0],list):
                n_p = n_p + lseq2np(Ptree[u][0][0])
                if Ptree[u][2].shape[1] > 1:
                    #n_p = lmaxpermnode(Ptree[u][2][0][2], n_p)
                    n_p = lmaxpermnode(Ptree[u][2], n_p)
            elif is_single_value(Ptree[u][0]):
                n_p = n_p + lseq2np(Ptree[u][0])
                
                if len(Ptree[u]) > 2 and Ptree[u][2].shape[1] > 1:
                    n_p = lmaxpermnode(Ptree[u][2], n_p) 
            else:     
                n_p = n_p + lseq2np(Ptree[u][0][:,0])
                
                if len(Ptree[u]) > 2 and Ptree[u][2].shape[1] > 1:
                    n_p = lmaxpermnode(Ptree[u][2], n_p) 
    
    return n_p

def lseq2np(S):
    """
    Calculate the logarithm of the number of permutations for a given sequence.

    This function calculates the logarithm of the number of permutations for a given sequence.

    Parameters:
    S (numpy.ndarray): The input sequence.

    Returns:
    int: The logarithm of the number of permutations for the sequence.
    """
    if is_single_value(S):
        nS = 1
        if np.isnan(S):
            U = np.nan
            cnt = 0
        else:
            U, cnt = np.unique(S, return_counts=True)
            
    else: 
        nS = len(S)
        U, cnt = np.unique(S, return_counts=True)
        
    #lfac=palm_factorial(nS)
    lfac=palm_factorial()
    n_p = lfac[nS] - np.sum(lfac[cnt])
    return n_p

def lmaxflipnode(Ptree, ns):
    """
    Calculate the logarithm of the maximum number of sign-flips within a palm tree node.

    This function calculates the logarithm of the maximum number of sign-flips within a palm tree node.

    Parameters:
    Ptree (list or numpy.ndarray): The palm tree structure.
    ns (int): The current logarithm of sign-flips (initialized to 0).

    Returns:
    int: The logarithm of the maximum number of sign-flips within the node.
    """
    for u in range(len(Ptree)):
        if len(Ptree[u][2][0]) > 1:
            ns = lmaxflipnode(Ptree[u][2], ns)
        ns = ns + len(Ptree[u][1])
    return ns

def is_single_value(variable):
    """
    Check if an array contains a singlevalue.

    This function checks if an array contains a singlevalue.

    Parameters:
    arr (numpy.ndarray or list): The array to be checked.

    Returns:
    bool: True if the array contains a single value, False otherwise.
    """
    return isinstance(variable, (int, float, complex))

def palm_factorial(N=101):
    """
    Calculate logarithmically scaled factorials up to a given number.

    This function precomputes logarithmically scaled factorials up to a specified number.

    Parameters:
    N (int, optional): The maximum number for which to precompute factorials (defaults to 101).

    Returns:
    numpy.ndarray: An array of precomputed logarithmically scaled factorials.
    """
    if N == 1:
        N = 101
    # Initialize the lf array with zeros
    lf = np.zeros(N+1)

    # Calculate log(factorial) values
    for n in range(1, N+1):
        lf[n] = np.log(n) + lf[n-1]

    return lf

import numpy as np

def renumber(B):
    
    """
    Renumber the elements in a 2D numpy array B, preserving their order within distinct blocks.

    This function renumbers the elements in the input array B based on distinct values in its first column.
    Each distinct value represents a block, and the elements within each block are renumbered sequentially,
    while preserving the relative order of elements within each block.

    Parameters:
    B (numpy.ndarray): The 2D input array to be renumbered.

    Returns:
    tuple: A tuple containing:
        - Br (numpy.ndarray): The renumbered array, where elements are renumbered within blocks.
        - addcol (bool): A boolean indicating whether a column was added during renumbering.

    """

    # Extract the first column of the input array B
    B1 = B[:, 0]
    # Find the unique values in B1 and store them in U
    U = np.unique(B1)
    # Create a boolean array to keep track of added columns
    addcolvec = np.zeros_like(U, dtype=bool)
    # Get the number of unique values
    nU = U.shape[0]
    # Create an empty array Br with the same shape as B
    Br = np.zeros_like(B)
    
    # Loop through unique values in B1
    for u in range(nU):
         # Find indices where B1 is equal to the current unique value U[u]
        idx = B1 == U[u]
        # Renumber the corresponding rows in Br based on the index
        Br[idx, 0] = (u + 1) * np.sign(U[u])
        
        # Check if B has more than one column
        if B.shape[1] > 1:
            # Recursively call renumber for the remaining columns and update addcolvec
            Br[idx, 1:], addcolvec[u] = renumber(B[idx, 1:])
        elif np.sum(idx) > 1:
             # If there's only one column and more than one matching row, set addcol to True
            addcol = True
            Br[idx] = -np.abs(B[idx])
        else:
            addcol = False
    # Check if B has more than one column and if any columns were added
    if B.shape[1] > 1:
        addcol = np.any(addcolvec)
    # Return the renumbered array Br and the addcol flag
    return Br, addcol

def palm_reindex(B, meth='fixleaves'):
    """
    Reindex a 2D numpy array using different procedures while preserving block structure.

    This function reorders the elements of a 2D numpy array `B` by applying one of several reindexing methods.
    The primary goal of reindexing is to assign new values to elements in such a way that they are organized
    in a desired order or structure.

    Parameters:
    B (numpy.ndarray): The 2D input array to be reindexed.
    meth (str, optional): The reindexing method to be applied. It can take one of the following values:
        - 'fixleaves': This method reindexes the input array by preserving the order of unique values in the
          first column and recursively reindexes the remaining columns. It is well-suited for hierarchical
          data where the first column represents levels or leaves.
        - 'continuous': This method reindexes the input array by assigning new values to elements in a
          continuous, non-overlapping manner within each column. It is useful for continuous data or when
          preserving the order of unique values is not a requirement.
        - 'restart': This method reindexes the input array by restarting the numbering from 1 for each block
          of unique values in the first column. It is suitable for data that naturally breaks into distinct
          segments or blocks.
        - 'mixed': This method combines both the 'fixleaves' and 'continuous' reindexing methods. It reindexes
          the first columns using 'fixleaves' and the remaining columns using 'continuous', creating a mixed
          reindexing scheme.

    Returns:
    numpy.ndarray: The reindexed array, preserving the block structure based on the chosen method.


    Raises:
    ValueError: If the `meth` parameter is not one of the valid reindexing methods.
    """

    # Convert meth to lowercase
    meth = meth.lower()
    
    # Initialize the output array Br with zeros
    Br = np.zeros_like(B)
    
    if meth == 'continuous':
        # Find unique values in the first column of B
        U = np.unique(B[:, 0])
        
        # Renumber the first column based on unique values
        for u in range(U.shape[0]):
            idx = B[:, 0] == U[u]
            Br[idx, 0] = (u + 1) * np.sign(U[u])
        
        # Loop through columns starting from the 2nd column    
        for b in range(1, B.shape[1]):  # From the 2nd column onwards
            Bb = B[:, b]
            Bp = Br[:, b - 1]  # Previous column
            # Find unique values in the previous column
            Up = np.unique(Bp)
            cnt = 1
            
            # Renumber elements within blocks based on unique values
            for up in range(Up.shape[0]):
                idxp = Bp == Up[up]
                U = np.unique(Bb[idxp])
                
                # Renumber elements within the block
                for u in range(U.shape[0]):
                    idx = np.logical_and(Bb == U[u], idxp)
                    Br[idx, b] = cnt * np.sign(U[u])
                    cnt += 1
                    
    elif meth == 'restart':
        # Renumber each block separately, starting from 1
        Br, _ = renumber(B)
        
    elif meth == 'mixed':
        # Mix both 'restart' and 'continuous' methods
        Ba, _ = palm_reindex(B, 'restart')
        Bb, _ = palm_reindex(B, 'continuous')
        Br = np.hstack((Ba[:, :-1], Bb[:, -1:]))
        
    elif meth=="fixleaves":
        # Reindex using 'fixleaves' method as defined in the renumber function

        B1 = B[:, 0]
        U = np.unique(B1)
        addcolvec = np.zeros_like(U, dtype=bool)
        nU = U.shape[0]
        Br = np.zeros_like(B)

        for u in range(nU):
            idx = B1 == U[u]
            Br[idx, 0] = (u + 1) * np.sign(U[u])

            if B.shape[1] > 1:
                Br[idx, 1:], addcolvec[u] = renumber(B[idx, 1:])
            elif np.sum(idx) > 1:
                addcol = True
                Br[idx] = -np.abs(B[idx])
            else:
                addcol = False

        if B.shape[1] > 1:
            addcol = np.any(addcolvec)
        
        if addcol:
            # Add a column of sequential numbers to Br and reindex
            col = np.arange(1, Br.shape[0] + 1).reshape(-1, 1)
            Br = np.hstack((Br, col))
            Br, _ = renumber(Br)
            
    else:
        # Raise a ValueError for an unknown method
        raise ValueError(f'Unknown method: {meth}')
    # Return the reindexed array Br
    return Br


import numpy as np

def palm_tree(B, M=None):
    """
    Construct a palm tree structure from an input matrix B and an optional design-matrix M.

    The palm tree represents a hierarchical structure where each node can have three branches:
    - The left branch contains data elements.
    - The middle branch represents special features (if any).
    - The right branch contains nested structures.

    Parameters:
    B (numpy.ndarray): The input matrix where each row represents the Multi-level block definitions of the PALM tree.
    M (numpy.ndarray, optional): An optional Design-matrix that associates each node in B with additional data.
                                 Defaults to None.

    Returns:
    list: A list containing three elements:
        - Ptree[0] (numpy.ndarray or list): The left branch of the palm tree, containing data elements.
        - Ptree[1] (numpy.ndarray, list, or empty list): The middle branch of the palm tree, representing
                                                       special features (if any).
        - Ptree[2] (numpy.ndarray or list): The right branch of the palm tree, containing nested structures.
    """

    # If M is not provided, create a default M matrix with sequential values
    if M is None:
        M = np.arange(1, B.shape[0] + 1).reshape(-1, 1)
    # Check if the number of rows in B and M match, raise an error if not
    elif B.shape[0] != M.shape[0]:
        raise ValueError("The two inputs must have the same number of rows.")

    # Make some initial sanity checks
    O = np.arange(1, M.shape[0] + 1).reshape(-1, 1)

    # Determine if the entire block is positive
    wholeblock = B[0, 0] > 0

    # Initialize a list to store the palm tree structure
    Ptree = [[] for _ in range(3)]

    # Recursively build the palm tree structure
    Ptree[0], Ptree[2] = maketree(B[:, 1:], M, O, wholeblock, wholeblock)

    # If the block is a whole block, set the middle branch to zeros, otherwise, set it to an empty list
    if wholeblock:
        Ptree[1] = np.zeros(Ptree[2].shape[0], dtype=bool)
    else:
        Ptree[1] = []

    # Return the palm tree structure
    return Ptree

def maketree(B, M, O, wholeblock, nosf):
    """
    Recursively construct a palm tree structure from input matrices.

    This function builds a palm tree structure by recursively processing input matrices representing
    nodes in the palm tree.

    Parameters:
    B (numpy.ndarray): The input matrix where each row represents a node in the palm tree (Block definitions).
    M (numpy.ndarray): The corresponding Design-matrix, which associates nodes in B with additional data.
    O (numpy.ndarray): Observation indices
    wholeblock (bool): A boolean indicating if the entire block is positive based on the first element of B.
    nosf (bool): A boolean indicating if there are no signflip this level


    Returns:
    tuple: A tuple containing:
        - S (numpy.ndarray or float): The palm tree structure for this branch.
        - Ptree (numpy.ndarray or list): The palm tree structure
    """
    
    # Extract the first column of the input matrix B
    B1 = B[:, 0]

    # Find unique values in the first column of B
    U = np.unique(B1)

    # Get the number of unique values
    nU = len(U)

    # Initialize the Ptree array based on the number of columns in B
    if B.shape[1] > 1:
        Ptree = np.empty((nU, 3), dtype=object)
    else:
        Ptree = np.empty((nU, 1), dtype=object)

    # Loop through unique values in the first column of B
    for u in range(nU):
        # Find indices where the first column matches the current unique value U[u]
        idx = B1 == U[u]

        if B.shape[1] > 1:
            # Determine if the entire block is positive for this branch
            wholeblockb = B[np.where(idx)[0][0], 0] > 0

            # Recursively build left and right branches
            Ptree[u][0], Ptree[u][2] = maketree(B[idx, 1:], M[idx], O[idx], wholeblockb, wholeblockb or nosf)

            # Initialize the middle branch as an empty list
            Ptree[u][1] = []

            # Check if there are no special features
            if nosf:
                Ptree[u][1] = []
            # Check if the right branch has more than one column
            elif Ptree[u][2].shape[1] > 1:
                if isinstance(Ptree[u][0][0], np.ndarray):
                    if M.ndim == 0:
                        if np.isnan(Ptree[u][0][0]):
                            Ptree[u][1] = []
                        else:
                            Ptree[u][1] = np.zeros(Ptree[u][2].shape[0], dtype=int)
                    else:
                        if np.isnan(Ptree[u][0][0][0]):
                            Ptree[u][1] = []
                        else:
                            Ptree[u][1] = np.zeros(Ptree[u][2].shape[0], dtype=int)
                else:
                    if np.isnan(Ptree[u][0][0]):
                        Ptree[u][1] = []
                    else:
                        Ptree[u][1] = np.zeros(Ptree[u][2].shape[0], dtype=int)
        else:
            # Set the first column of this branch to O[idx]
            Ptree[u][0] = O[idx]

    if wholeblock and nU > 1:
        # Sort the combined array based on the first column
        combined_array = np.column_stack((B1, M))
        sorted_indices = np.argsort(combined_array[:, 0])
        B1M = combined_array[sorted_indices]

        # Use lexsort to sort by both columns
        sorted_indices = np.lexsort((B1M[:, 1], B1M[:, 0]))
        B1M_sorted = B1M[sorted_indices]
        Ms = B1M_sorted[:, 1:]
        Msre = Ms.reshape(nU, int(Ms.size / nU))

        # Get unique rows and their indices
        _, S = np.unique(Msre, axis=0, return_inverse=True)

        # Put in ascending order and (un)shuffle the branches accordingly
        idx = np.argsort(S)
        S = np.column_stack((S[idx], np.arange(0, S.shape[0]), np.arange(0, S.shape[0]))) + 1
        Ptree = Ptree[idx, :]

    elif wholeblock and nU == 1:
        # If it's a whole block with a single unique value, set S to [1, 1, 1]
        S = [1, 1, 1]
    else:
        # If not a whole block, set S to NaN
        S = np.nan

    return S, Ptree
