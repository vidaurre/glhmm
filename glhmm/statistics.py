"""
Permutation testing from Gaussian Linear Hidden Markov Model
@author: Nick Y. Larsen 2023
"""

import numpy as np
import pandas as pd
import random
import copy
from tqdm import tqdm
from glhmm.palm_functions import *


def test_across_subjects(D_data, R_data, method="regression", Nperm=0, confounds = None, dict_family = None, test_statistic_option=False, FWER_correction=False):
    """
    This function performs statistical tests between a independent variable (`D_data`) and the dependent-variable (`R_data`) using permutation testing.
    The permutation testing is performed across across different subjects and it is possible to take family structure into account.
    This procedure is particularly valuable for investigating the differences between subjects in one's study. 
    
    Three options are available to customize the statistical analysis to a particular research questions:
        - 'regression': Perform permutation testing using regression analysis.
        - 'correlation': Conduct permutation testing with correlation analysis.
        - 'cca': Apply permutation testing using canonical correlation analysis.
        
    Parameters:
    --------------
    D_data (numpy.ndarray): Input data array of shape that can be either a 2D array or a 3D array.
                            For 2D, the data is represented as a (n, p) matrix, where n represents 
                            the number of subjects, and p represents the number of predictors.
                            For 3D array, it has a shape (T, n, q), where the first dimension 
                            represents timepoints, the second dimension represents the number of subjects, 
                            and the third dimension represents features. 
                            For 3D, permutation testing is performed per timepoint for each subject.              
    R_data (numpy.ndarray): The dependent variable can be either a 2D array or a 3D array. 
                            For 2D array, it has a shape of (n, q), where n represents 
                            the number of subjects, and q represents the outcome of the dependent variable.
                            For 3D array, it has a shape (T, n, q), where the first dimension 
                            represents timepoints, the second dimension represents the number of subjects, 
                            and the third dimension represents a dependent variable.   
                            For 3D, permutation testing is performed per timepoint for each subject.                 
    method (str, optional): The statistical method to be used for the permutation test. Valid options are
                            "regression", "correlation", or "cca". (default: "regression").      
                            Note: "cca" stands for Canonical Correlation Analysis                                        
    Nperm (int): Number of permutations to perform (default: 1000).                       
    confounds (numpy.ndarray or None, optional): 
                            The confounding variables to be regressed out from the input data (D_data).
                            If provided, the regression analysis is performed to remove the confounding effects. 
                            (default: None)     
    dict_family (dict): 
                            Dictionary containing family structure information.                          
                                - file_location (str): The file location of the family structure data in CSV format.
                                - M (numpy.ndarray, optional): The matrix of attributes, which is not typically required.
                                                          Defaults to None.
                                - CMC (bool, optional): A flag indicating whether to use the Conditional Monte Carlo method (CMC).
                                              Defaults to False.
                                - EE (bool, optional): A flag indicating whether to assume exchangeable errors, which allows permutation.
                                              Defaults to True. Other options are not available.            
    test_statistic_option (bool, optional): 
                            If True, the function will return the test statistic for each permutation.
                            (default: False) 
    FWER_correction (bool, optional): 
                            Specify whether to perform family-wise error rate (FWER) correction for multiple comparisons using the MaxT method(default: False).                     
                                
    Returns:
    ----------  
    result (dict): A dictionary containing the following keys. Depending on the `test_statistic_option` and `method`, it can return the p-values, 
        correlation coefficients, test statistics.
        'pval': P-values for the test with shapes based on the method:
            - method=="Regression": (T, p)
            - method=="correlation": (T, p, q)
            - method=="cca": (T, 1)
        'test_statistic': Test statistic is the permutation distribution if `test_statistic_option` is True, else None.
            - method=="Regression": (T, Nperm, p)
            - method=="correlation": (T, Nperm, p, q)
            - method=="cca": (T, Nperm, 1)
        'corr_coef': Correlation coefficients for the test with shape (T, p, q) if method=="correlation", else None.
        'test_type': the type of test, which is the name of the function
        'method': the method used for analysis Valid options are
                "regression", "correlation", or "cca", "one_vs_rest" and "state_pairs" (default: "regression").
        'Nperm' :The number of permutations that has been performed.    
                  
    Note:
    The function automatically determines whether permutation testing is performed per timepoint for each subject or
    for the whole data based on the dimensionality of `D_data`.
    The function assumes that the number of rows in `D_data` and `R_data` are equal
    """
    # Have to run the permutation test function 1 time at least once
    if Nperm==0:
        Nperm+=1
        
    # Check validity of method and data_type
    valid_methods = ["regression", "correlation", "cca"]
    validate_condition(method in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    # Get the shapes of the data
    n_T, _, n_p, n_q, D_data, R_data = get_input_shape(D_data, R_data)
    # Note for convension we wrote (T, p, q) => (n_T, n_p, n_q)
    
    # Crate the family structure by looking at the dictionary 
    if dict_family is not None:
        # process dictionary of family structure
        dict_mfam=process_family_structure(dict_family, Nperm) 
        
    # Initialize arrays based on shape of data shape and defined options
    pval, corr_coef, test_statistic_list = initialize_arrays(R_data, n_p, n_q, n_T, method, Nperm, test_statistic_option)

    for t in tqdm(range(n_T)) if n_T > 1 else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        D_t, R_t = deconfound_values(D_data[t, :],R_data[t, :], confounds)
        
        # Removing rows that contain nan-values
        D_t, R_t= remove_nan_values(D_t, R_t, t, n_T) ### can be optmized
        
        if method == "correlation":
            # Calculate correlation coefficient
            corr_coef[t, :] = get_corr_coef(D_t, R_t)
            test_statistic= None
        # Create test_statistic based on method
        test_statistic, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, D_t)

        if dict_family is None:
            # Get indices for permutation
            permutation_matrix = permutation_matrix_across_subjects(Nperm, R_t)
            
        else:
            # Call function "__palm_quickperms" from glhmm.palm_functions
            permutation_matrix = __palm_quickperms(dict_mfam["EB"], M=dict_mfam["M"], nP=dict_mfam["nP"], 
                                            CMC=dict_mfam["CMC"], EE=dict_mfam["EE"])
            # Need to convert the index so it starts from 0
            permutation_matrix = permutation_matrix-1
            
        for perm in tqdm(range(Nperm)) if n_T == 1 else range(Nperm):
            # Perform permutation on R_t
            Rin = R_t[permutation_matrix[:, perm]]
            # Calculate the permutation distribution
            test_statistic = test_statistic_calculations(D_t, Rin, perm, test_statistic, proj, method)
        # Calculate p-values
        pval = get_pval(test_statistic, Nperm, method, t, pval, FWER_correction) if Nperm>1 else 0
        
        # Output test statistic if it is set to True can be hard for memory otherwise
        if test_statistic_option==True:
            test_statistic_list[t,:] = test_statistic
        elif method !="correlation":
            test_statistic_list[t,:] = test_statistic[0,:]
    pval =np.squeeze(pval) if np.abs(np.sum(pval))>0 else [] 
    corr_coef =np.squeeze(corr_coef) if corr_coef is not None else []
    test_statistic_list =np.squeeze(test_statistic_list) if test_statistic_list is not None else []
    Nperm = 0 if Nperm==1 else Nperm
    
    # Return results
    result = {
        'pval': pval,
        'corr_coef': corr_coef,
        'test_statistic': test_statistic_list,
        'test_type': 'test_across_subjects',
        'method': method,
        'FWER_correction':FWER_correction,
        'Nperm': Nperm}
    return result



def test_across_trials_within_session(D_data, R_data, idx_data, method="regression", Nperm=0, confounds=None, trial_timepoints=None,test_statistic_option=False, FWER_correction=False):
    """
    This function performs statistical tests between a independent variable (`D_data`) and the dependent-variable (`R_data`) using permutation testing.
    The permutation testing is performed across different trials within a session using permutation testing
    This procedure is particularly valuable for investigating the differences between trials in one or more sessions.  
    An example could be if we want to test if any learning is happening during a session that might speed up times.
    
    Three options are available to customize the statistical analysis to a particular research questions:
        - 'regression': Perform permutation testing using regression analysis.
        - 'correlation': Conduct permutation testing with correlation analysis.
        - 'cca': Apply permutation testing using canonical correlation analysis.
             
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
                                "regression", "correlation", or "cca". (default: "regression").
                                Note: "cca" stands for Canonical Correlation Analysis    
        Nperm (int): Number of permutations to perform (default: 1000). 
        confounds (numpy.ndarray or None, optional): 
                                The confounding variables to be regressed out from the input data (D_data).
                                If provided, the regression analysis is performed to remove the confounding effects. 
                                (default: None):    
        trial_timepoints (int): Number of timepoints for each trial (default: None)                                                          
        test_statistic_option (bool, optional): 
                                If True, the function will return the test statistic for each permutation.
                                (default: False) 
        FWER_correction (bool, optional): 
                                Specify whether to perform family-wise error rate (FWER) correction for multiple comparisons using the MaxT method(default: False).                        
                                                      
    Returns:
    ----------  
        result (dict): A dictionary containing the following keys. Depending on the `test_statistic_option` and `method`, it can return the p-values, 
            correlation coefficients, test statistics.
            'pval': P-values for the test with shapes based on the method:
                - method=="Regression": (T, p)
                - method=="correlation": (T, p, q)
                - method=="cca": (T, 1)
            'test_statistic': Test statistic is the permutation distribution if `test_statistic_option` is True, else None.
                - method=="Regression": (T, Nperm, p)
                - method=="correlation": (T, Nperm, p, q)
                - method=="cca": (T, Nperm, 1)
            'corr_coef': Correlation coefficients for the test with shape (T, p, q) if method=="correlation", else None.
            'test_type': the type of test, which is the name of the function
            'method': the method used for analysis Valid options are
                    "regression", "correlation", or "cca", "one_vs_rest" and "state_pairs" (default: "regression").
            'Nperm' :The number of permutations that has been performed.

    Note:
        The function automatically determines whether permutation testing is performed per timepoint for each subject or
        for the whole data based on the dimensionality of `D_data`.
        The function assumes that the number of rows in `D_data` and `R_data` are equal
    """
    # Have to run the permutation test function 1 time at least once
    if Nperm==0:
        Nperm+=1 
    # Check validity of method
    valid_methods = ["regression", "correlation", "cca"]
    validate_condition(method in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))

    # Get input shape information
    n_T, _, n_p, n_q, D_data, R_data = get_input_shape(D_data, R_data)
    n_q = R_data.shape[-1]
    
    # Get indices for permutation
    if len(idx_data.shape)==2:
        idx_array = get_indices_array(idx_data)
    else:
        idx_array =idx_data.copy()        

    # Initialize arrays based on shape of data shape and defined options
    pval, corr_coef, test_statistic_list = initialize_arrays(R_data, n_p, n_q, n_T, method, Nperm, test_statistic_option)


    for t in tqdm(range(n_T)) if n_T > 1 else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        D_t, R_t = deconfound_values(D_data[t, :],R_data[t, :], confounds)
        
        # Removing rows that contain nan-values
        D_t, R_t= remove_nan_values(D_t, R_t, t, n_T) ### can be optmized
        
        if method == "correlation":
            # Calculate correlation coefficient
            corr_coef[t, :] = get_corr_coef(D_t,R_t)
        
        # Create test_statistic and pval_perms based on method
        test_statistic, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, D_t)
    
            
        # Calculate permutation matrix of D_t 
        permutation_matrix = permutation_matrix_across_trials_within_session(Nperm,R_t, idx_array,trial_timepoints)
                
        for perm in range(Nperm):
        #for perm in tqdm(range(Nperm)) if n_T == 1 else range(n_T):
            # Perform permutation on R_t
            Rin = R_t[permutation_matrix[:, perm]]

            # Calculate the permutation distribution
            test_statistic = test_statistic_calculations(D_t, Rin, perm, test_statistic, proj, method)
        # Calculate p-values
        pval = get_pval(test_statistic, Nperm, method, t, pval, FWER_correction) if Nperm>1 else 0
        if test_statistic_option==True:
            test_statistic_list[t,:] = test_statistic
    pval =np.squeeze(pval) if np.abs(np.sum(pval))>0 else [] 
    corr_coef =np.squeeze(corr_coef) if corr_coef is not None  else []
    test_statistic_list =np.squeeze(test_statistic_list) if test_statistic_list is not None else []
    Nperm = 0 if Nperm==1 else Nperm
    # Return results
    result = {
        'pval': pval,
        'corr_coef': corr_coef,
        'test_statistic': test_statistic_list,
        'test_type': 'test_across_trials_within_session',
        'method': method,
        'Nperm': Nperm}
    
    return result

def test_across_sessions_within_subject(D_data, R_data, idx_data, method="regression", Nperm=0, confounds=None,test_statistic_option=False,FWER_correction=False):
    """
    This function performs statistical tests between a independent variable (`D_data`) and the dependent-variable (`R_data`) using permutation testing. 
    The permutation testing is performed across sessions within the same subject, while keeping the trial order the same.
    This procedure is particularly valuable for investigating the effects of long-term treatments or monitoring changes in brain responses across sessions over time.

    Three options are available to customize the statistical analysis to a particular research questions:
        - 'regression': Perform permutation testing using regression analysis.
        - 'correlation': Conduct permutation testing with correlation analysis.
        - 'cca': Apply permutation testing using canonical correlation analysis.
           
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
                                "regression", "correlation", or "cca". (default: "regression").
                                Note: "cca" stands for Canonical Correlation Analysis    
        Nperm (int): Number of permutations to perform (default: 1000).                
        confounds (numpy.ndarray or None, optional): 
                                The confounding variables to be regressed out from the input data (D_data).
                                If provided, the regression analysis is performed to remove the confounding effects. 
                                (default: None):                                                              
        test_statistic_option (bool, optional): 
                                If True, the function will return the test statistic for each permutation.
                                (default: False) 
        FWER_correction (bool, optional): 
                                Specify whether to perform family-wise error rate (FWER) correction for multiple comparisons using the MaxT method(default: False).
                                
    Returns:
    ----------  
        result (dict): A dictionary containing the following keys. Depending on the `test_statistic_option` and `method`, it can return the p-values, 
            correlation coefficients, test statistics.
            'pval': P-values for the test with shapes based on the method:
                - method=="Regression": (T, p)
                - method=="correlation": (T, p, q)
                - method=="cca": (T, 1)
            'test_statistic': Test statistic is the permutation distribution if `test_statistic_option` is True, else None.
                - method=="Regression": (T, Nperm, p)
                - method=="correlation": (T, Nperm, p, q)
                - method=="cca": (T, Nperm, 1)
            'corr_coef': Correlation coefficients for the test with shape (T, p, q) if method=="correlation", else None.
            'test_type': the type of test, which is the name of the function
            'method': the method used for analysis Valid options are
                    "regression", "correlation", or "cca", "one_vs_rest" and "state_pairs" (default: "regression").
            'Nperm' :The number of permutations that has been performed.
                  
    Note:
        The function automatically determines whether permutation testing is performed per timepoint for each subject or
        for the whole data based on the dimensionality of `D_data`.
        The function assumes that the number of rows in `D_data` and `R_data` are equal

    """ 
    # Have to run the permutation test function 1 time at least once
    if Nperm==0:
        Nperm+=1
    # Check validity of method
    valid_methods = ["regression", "correlation", "cca"]
    validate_condition(method in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    # Get indices for permutation
    if len(idx_data.shape)==2:
        idx_array = get_indices_array(idx_data)
    else:
        idx_array =idx_data.copy()

    # Get input shape information
    n_T, _, n_p,n_q, D_data, R_data = get_input_shape(D_data, R_data)
    #n_q = R_data.shape[-1]
    
# Initialize arrays based on shape of data shape and defined options
    pval, corr_coef, test_statistic_list = initialize_arrays(R_data, n_p, n_q, n_T, method, Nperm, test_statistic_option)
    for t in tqdm(range(n_T)) if n_T > 1 else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        D_t, R_t = deconfound_values(D_data[t, :],R_data[t, :], confounds)
        
        # Removing rows that contain nan-values
        D_t, R_t= remove_nan_values(D_t, R_t, t, n_T) ### can be optmized
        
        if method == "correlation":
            # Calculate correlation coefficient
            corr_coef[t, :] = get_corr_coef(D_t, R_t)

        # Create test_statistic and pval_perms based on method
        test_statistic, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, D_t)

        # Calculate permutation matrix of D_t 
        permutation_matrix = permutation_matrix_within_subject_across_sessions(Nperm, D_t, idx_array)
        
        for perm in range(Nperm):
        #for perm in tqdm(range(Nperm)) if n_T == 1 else range(n_T):
            # Perform permutation on R_t
            Rin = R_t[permutation_matrix[:, perm]]
            # Calculate the permutation distribution
            test_statistic = test_statistic_calculations(D_t, Rin, perm, test_statistic, proj, method)
        # Caluclate p-values
        pval = get_pval(test_statistic, Nperm, method, t, pval, FWER_correction) if Nperm>1 else 0
        if test_statistic_option==True:
            test_statistic_list[t,:] = test_statistic
    pval =np.squeeze(pval) if np.abs(np.sum(pval))>0 else [] 
    corr_coef =np.squeeze(corr_coef) if corr_coef is not None  else []
    test_statistic_list =np.squeeze(test_statistic_list) if test_statistic_list is not None  else []
    Nperm = 0 if Nperm==1 else Nperm          
    # Return values
    result = {
        'pval': pval,
        'corr_coef': [] if np.sum(corr_coef)==0 else corr_coef,
        'test_statistic': [] if np.sum(test_statistic_list)==0 else test_statistic_list,
        'test_type': 'test_across_sessions_within_subject',
        'method': method,
        'FWER_correction':FWER_correction,
        'Nperm': Nperm}
    return result

def test_across_visits(input_data, vpath_data, n_states, method="regression", Nperm=0, confounds=None, test_statistic_option=False, pairwise_statistic ="mean",FWER_correction=False):
    from itertools import combinations
    """
    Perform permutation testing within a session for continuous data.
    This function performs statistical tests, such as regression, correlation, 
    canonical correlation analysis (cca), one-vs-rest, and state pairs, 
    between a dependent variable (`input_data`) and a hidden state path (`vpath_data`) using permutation testing. 


    Parameters:
    --------------            
        input_data (numpy.ndarray): Dependent variable with shape (n, q), where n is the number of samples (n_timepoints x n_trials), 
                                    and q represents dependent/target variables.  
        vpath_data (numpy.ndarray): The hidden state path data of the continuous measurements represented as a (n, p) matrix. 
                                    It could be a 2D matrix where each row represents a trials over a period of time and
                                    each column represents a state variable and gives the shape ((n_timepoints X n_trials), n_states). 
                                    If it is a 1D array of of shape ((n_timepoints X n_trials),) where each row value represent a giving state.                                 
        n_states (int):             The number of hidden states in the hidden state path data.
        method (str, optional):     Statistical method for the permutation test. Valid options are 
                                    "regression", "correlation", "cca", "one_vs_rest" or "state_pairs". 
                                    Note: "cca" stands for Canonical Correlation Analysis.   
        Nperm (int):                Number of permutations to perform (default: 0). 
        test_statistic_option (bool, optional): 
                                    If True, the function will return the test statistic for each permutation.
                                    (default: False) 
        pairwise_statistic (str, optional)  
                                    The chosen statistic when applying methods "one_vs_rest" or "state_pairs". 
                                    Valid options are "mean" or "median" (default: "mean").
        FWER_correction (bool, optional): 
                                    Specify whether to perform family-wise error rate (FWER) correction for multiple comparisons using the MaxT method(default: False).
                     
    Returns:
    ----------  
        result (dict): A dictionary containing the following keys. Depending on the `test_statistic_option` and `method`, it can return the p-values, 
            correlation coefficients, test statistics.
            'pval': P-values for the test with shapes based on the method:
                - method=="Regression": (T, p)
                - method=="correlation": (T, p, q)
                - method=="cca": (T, 1)
            'test_statistic': Test statistic is the permutation distribution if `test_statistic_option` is True, else None.
                - method=="Regression": (T, Nperm, p)
                - method=="correlation": (T, Nperm, p, q)
                - method=="cca": (T, Nperm, 1)
            'corr_coef': Correlation coefficients for the test with shape (T, p, q) if method=="correlation", else None.
            'test_type': the type of test, which is the name of the function
            'method': the method used for analysis Valid options are
                    "regression", "correlation", or "cca", "one_vs_rest" and "state_pairs" (default: "regression").
            'Nperm' :The number of permutations that has been performed.
                
    Note:
        The function assumes that the number of rows in `vpath_data` and `Y_data` are equal
    """
    # Have to run the permutation test function 1 time at least once
    if Nperm==0:
        Nperm+=1
    # Check validity of method
    valid_methods = ["regression", "correlation", "cca", "one_vs_rest", "state_pairs"]
    validate_condition(method in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    valid_statistic = ["mean", "median"]
    validate_condition(pairwise_statistic.lower() in valid_statistic, "Invalid option specified for 'statistic'. Must be one of: " + ', '.join(valid_statistic))
    
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
        data_t, _ = deconfound_values(input_data[t, :],None, confounds)
        
        # Removing rows that contain nan-values
        data_t, vpath_array= remove_nan_values(data_t, vpath_array, t, n_T) ### can be optmized
        
        if method == "correlation":
            # Calculate correlation coefficient
            corr_coef[t, :] = get_corr_coef(data_t, vpath_data[t,:,:])

        # Perform permutation test
        # Create test_statistic and pval_perms based on method
        if method != "state_pairs":
            ###################### Permutation testing for other tests beside state pairs #################################
            # Create test_statistic and pval_perms based on method
            test_statistic, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, 
                                                                            data_t)
            # Perform permutation testing
            for perm in tqdm(range(Nperm)) if n_T == 1 else range(n_T):
                # Redo vpath_surrogate calculation if the number of states are not the same (out of 1000 permutations it happens maybe 1-2 times with this demo dataset)
                while True:
                    # Create vpath_surrogate
                    vpath_surrogate = surrogate_state_time(perm, vpath_array, n_states)
                    if len(np.unique(vpath_surrogate)) == n_states:
                        break  # Exit the loop if the condition is satisfied
                if method =="one_vs_rest":
                    for state in range(n_states):
                        test_statistic[perm,state] =calculate_baseline_difference(vpath_surrogate, data_t, state+1, pairwise_statistic.lower())
                elif method =="regression":
                    test_statistic = test_statistic_calculations(data_t,vpath_surrogate , perm,
                                                                            test_statistic, proj, method)
                else:
                    # Apply 1 hot encoding
                    vpath_surrogate_onehot = viterbi_path_to_stc(vpath_surrogate,n_states)
                    # Apply t-statistic on the vpath_surrogate
                    test_statistic = test_statistic_calculations(data_t,vpath_surrogate_onehot , perm,
                                                                                test_statistic, proj, method)
            pval = get_pval(test_statistic, Nperm, method, t, pval, FWER_correction) if Nperm>1 else 0
        ###################### Permutation testing for state pairs #################################
        elif method =="state_pairs":
            # Run this code if it is "state_pairs"
            # Correct for confounds and center data_t
            data_t, _ = deconfound_values(input_data[t, :],None, confounds)
            
            # Generates all unique combinations of length 2 
            pairwise_comparisons = list(combinations(range(1, n_states + 1), 2))
            test_statistic = np.zeros((Nperm, len(pairwise_comparisons)))
            pval = np.zeros((n_states, n_states))
            # Iterate over pairwise state comparisons
            for idx, (state_1, state_2) in tqdm(enumerate(pairwise_comparisons), total=len(pairwise_comparisons), desc="Pairwise comparisons"):    
                # Generate surrogate state-time data and calculate differences for each permutation
                for perm in range(Nperm):
                    # Redo vpath_surrogate calculation if the number of states are not the same (out of 1000 permutations it happens maybe 1-2 times with this demo dataset)
                    while True:
                        # Create vpath_surrogate
                        vpath_surrogate = surrogate_state_time(perm, vpath_array, n_states)
                        if len(np.unique(vpath_surrogate)) == n_states:
                            break  # Exit the loop if the condition is satisfied
                    test_statistic[perm,idx] = calculate_statepair_difference(vpath_surrogate, data_t, state_1, state_2, pairwise_statistic)
                
                if Nperm>1:
                    p_val= np.sum(test_statistic[:,idx] >= test_statistic[0,idx], axis=0) / (Nperm + 1)
                    pval[state_1-1, state_2-1] = p_val
                    pval[state_2-1, state_1-1] = 1 - p_val
            corr_coef =[]
                
        if test_statistic_option:
            test_statistic_list[t, :] = test_statistic

    pval =np.squeeze(pval) if np.abs(np.sum(pval))>0 else [] 
    corr_coef =np.squeeze(corr_coef) if corr_coef is not None else []
    test_statistic_list =np.squeeze(test_statistic_list) if test_statistic_list is not None else []
    Nperm = 0 if Nperm==1 else Nperm
    # Return results
    result = {
        
        'pval': pval,
        'corr_coef': corr_coef,
        'test_statistic': test_statistic_list,
        'test_type': 'test_across_visits',
        'method': method,
        'FWER_correction':FWER_correction,
        'Nperm': Nperm} 
    return result

def remove_nan_values(D_data, R_data, t, n_T):
    """
    Remove rows with NaN values from input data arrays.

    Parameters
    ----------
    D_data : numpy.ndarray
        Input data array containing features.
    R_data : numpy.ndarray
        Input data array containing response values.
    t      : int
        Timepoint of the data
    n_T    : int
        Total number of timepoint of the data
    Returns
    -------
    D_data : numpy.ndarray
        Cleaned feature data (D_data) with NaN values removed.  
    R_data : numpy.ndarray
        Cleaned response data (R_data) with NaN values removed.
    """
    FLAG = 0
    if R_data.ndim == 1:
        FLAG = 1
        R_data = R_data.reshape(-1,1) 
       
    # Check for NaN values and remove corresponding rows
    nan_mask = np.isnan(D_data).any(axis=1) | np.isnan(R_data).any(axis=1)
    # Get indices or rows that have been removed
    removed_indices = np.where(nan_mask)[0]

    D_data = D_data[~nan_mask]
    R_data = R_data[~nan_mask]
    # Only print this 1 time
    # Check if the array is empty
    if n_T==1 and np.any(removed_indices): 
        print("Rows with NaN values have been removed:")
        print("Removed Indices:", removed_indices)
    # Check if the array is empty
    elif np.any(removed_indices):
        print(f"Rows with NaN values have been removed at timepoint {t}:")
        print("Removed Indices:", removed_indices)
    if FLAG ==1:
        R_data =R_data.flatten()
    return D_data, R_data

def validate_condition(condition, error_message):
    """
    Validates a given condition and raises a ValueError with the specified error message if the condition is not met.

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

def process_family_structure(dict_family, Nperm):
    """
    Process a dictionary containing family structure information.

    Parameters:
    --------------
        dict_family (dict): Dictionary containing family structure information.
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
    
    # dict_family: dictionary of family structure
    # Nperm: number of permutations

    default_values = {
        'file_location' : 'None',
        'M': 'None',
        'CMC': 'False',
        'EE': 'False',
        'nP': Nperm
    }
    dict_mfam =dict_family.copy()

    # Validate and load family structure data
    if 'file_location' not in dict_mfam:
        raise ValueError("The 'file_location' variable must be defined in dict_family.")
    
    # Convert the DataFrame to a matrix
    EB = pd.read_csv(dict_mfam['file_location'], header=None).to_numpy()
    
    # Check for invalid keys in dict_family
    invalid_keys = set(dict_mfam.keys()) - set(default_values.keys())
    if not invalid_keys== set():
        valid_keys = ['M', 'CMC', 'EE']
        validate_condition(
            invalid_keys in valid_keys, "Invalid keys in dict_family: Must be one of: " + ', '.join(valid_keys)
        )
    
    # Set default values for M, CMC, and EE
    del dict_mfam['file_location']
    dict_mfam['EB'] = EB
    dict_mfam['nP'] = Nperm
    dict_mfam.setdefault('M', default_values['M'])
    dict_mfam.setdefault('CMC', default_values['CMC'])
    dict_mfam.setdefault('EE', default_values['EE'])
    
    return dict_mfam

def initialize_arrays(R_data, n_p, n_q, n_T, method, Nperm, test_statistic_option):
    from itertools import combinations
    """
    Initializes arrays for permutation testing.

    Parameters:
    --------------
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
        corr_coef (numpy array): Correlation coefficient for the test (n_T, n_p, n_q) if method="correlation", else None.
        test_statistic_list (numpy array): Test statistic values (n_T, Nperm, n_p) or (n_T, Nperm, n_p, n_q) if method="correlation" , else None.
    """

    # Initialize the arrays based on the selected method and data dimensions
    if  method == "regression":
        pval = np.zeros((n_T, n_q))
        corr_coef = None
        if test_statistic_option==True:
            test_statistic_list = np.zeros((n_T, Nperm, n_q))
        else:
            test_statistic_list= np.zeros((n_T, 1, n_q))
    elif  method == "cca":
        pval = np.zeros((n_T, 1))
        corr_coef = None
        if test_statistic_option==True:
            test_statistic_list = np.zeros((n_T, Nperm, 1))
        else:
            test_statistic_list= np.zeros((n_T, 1, 1))        
    elif method == "correlation" :
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
            test_statistic_list= np.zeros((n_T, 1, len(pairwise_comparisons)))
    elif method == "one_vs_rest":
        pval = np.zeros((n_T, n_p, n_q))
        corr_coef = []
        if test_statistic_option==True:
            test_statistic_list = np.zeros((n_T, Nperm, n_q))
        else:
            test_statistic_list= np.zeros((n_T, 1, n_q))

    return pval, corr_coef, test_statistic_list


def deconfound_values(D_data, R_data, confounds=None):
    """
    Deconfound the variables R_data and D_data for permutation testing.

    Parameters:
    --------------
        D_data  (numpy.ndarray): The input data array.
        R_data (numpy.ndarray or None): The second input data array (default: None).
            If None, assumes we are working across visits, and R_data represents the Viterbi path of a sequence.
        confounds (numpy.ndarray or None): The confounds array (default: None).

    Returns:
    ----------  
        numpy.ndarray: Deconfounded D_data  array.
        numpy.ndarray: Deconfounded R_data array (returns None if R_data is None).
            If R_data is None, assumes we are working across visits
    """
    
    # Calculate the centered data matrix based on confounds (if provided)
    if confounds is not None:
         # Centering confounds
        confounds = confounds - np.nanmean(confounds, axis=0)
        # Centering R_data
        D_data = D_data - np.nanmean(D_data, axis=0)
        # Regressing out confounds from R_data
        D_t = D_data - confounds @ np.linalg.pinv(confounds) @ D_data
        # Check if D_data is provided
        if R_data is not None:
                    # Regressing out confounds from D_data
            R_t = R_data - confounds @ np.linalg.pinv(confounds) @ R_data
        else:
            R_t = None # Centering D_data
            R_data = R_data - np.nanmean(R_data, axis=0)
              
    else:
        # Centering D_data and R_data
        D_t = D_data - np.nanmean(D_data, axis=0)
        R_t = None if R_data is None else R_data - np.nanmean(R_data, axis=0)
    
    return D_t, R_t

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
    # Define projection matrix
    proj = None
    # Initialize the permutation matrices based on the selected method
    if method in {"correlation"}:
        # Initialize test statistic output matrix based on the selected method
        test_statistic = np.zeros((Nperm, n_p, n_q))
        proj = None
    elif method =="cca":
        # Initialize test statistic output matrix based on the selected method
        test_statistic = np.zeros((Nperm, 1))
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

def permutation_matrix_across_subjects(Nperm, D_t):
    """
    Generates a normal permutation matrix with the assumption that each index is independent across subjects. 

    Parameters:
    --------------
        Nperm (int): The number of permutations.
        D_t (numpy.ndarray): The preprocessed data array.
        
    Returns:
    ----------  
        permutation_matrix (numpy.ndarray): Permutation matrix of subjects it got a shape (n_ST, Nperm)
    """
    permutation_matrix = np.zeros((D_t.shape[0],Nperm), dtype=int)
    for perm in range(Nperm):
        if perm == 0:
            permutation_matrix[:,perm] = np.arange(D_t.shape[0])
        else:
            permutation_matrix[:,perm] = np.random.permutation(D_t.shape[0])
    return permutation_matrix

def get_pval(test_statistic, Nperm, method, t, pval, FWER_correction):
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

    Returns:
    ----------  
        pval (numpy.ndarray): Updated updated p-value .

        
    # Ref: https://github.com/OHBA-analysis/HMM-MAR/blob/master/utils/testing/permtest_aux.m
    """
    if method == "regression" or method == "one_vs_rest":
        if FWER_correction:
            # Perform family wise permutation correction
            # Define the number of columns and rows
            nCols = test_statistic[0,:].shape[-1]
            nRows = len(test_statistic)
            # Get the maximum explained variance for each column
            max_test_statistic =np.tile(np.max(test_statistic, axis=1), (1, nCols)).reshape(nCols, nRows).T
            # Count every time there is a higher estimated R2 (better fit)
            pval[t, :] = np.sum(max_test_statistic>= test_statistic[0,:], axis=0) / (Nperm + 1)
        else:
            # Count every time there is a higher estimated R2 (better fit)
            pval[t, :] = np.sum(test_statistic >= test_statistic[0,:], axis=0) / (Nperm + 1)
        
    elif method == "correlation" or method =="cca":
        if FWER_correction:
            # Perform family wise permutation correction
            # Define the number of columns and rows
            nCols = test_statistic[0,:].shape[-1]
            nRows = len(test_statistic)
            # Get the maximum explained variance for each column
            max_test_statistic =np.tile(np.max(test_statistic, axis=1), (1, nCols)).reshape(nCols, nRows).T
            # Count every time there is a higher correlation coefficient
            pval[t, :] = np.sum(max_test_statistic>= test_statistic[0,:], axis=0) / (Nperm + 1)
        else:    
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


def permutation_matrix_across_trials_within_session(Nperm, R_t, idx_array, trial_timepoints=None):
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
        permutation_matrix (numpy.ndarray): Permutation matrix of subjects it got a shape (n_ST, Nperm)
    """
    # Perform within-session between-trial permutation based on the given indices
    # Createing the permutation matrix
    permutation_matrix = np.zeros((R_t.shape[0], Nperm), dtype=int)
    for perm in range(Nperm):
        if perm == 0:
            permutation_matrix[:,perm] = np.arange(R_t.shape[0])
        else:
            unique_indices = np.unique(idx_array)
            if trial_timepoints is None:
                count = 0
                for i in unique_indices:
                    if i ==0:
                        count =count+R_t[idx_array == unique_indices[i], :].shape[0]
                        permutation_matrix[0:count,perm]=np.random.permutation(np.arange(0,count))
                    else:
                        idx_count=R_t[idx_array == unique_indices[i], :].shape[0]
                        count =count+idx_count
                        permutation_matrix[count-idx_count:count,perm]=np.random.permutation(np.arange(count-idx_count,count))
    
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
                permutation_matrix[:,perm] =np.array(permutation_array)

    return permutation_matrix

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


def permutation_matrix_within_subject_across_sessions(Nperm, D_t, idx_array):
    """
    Generates permutation matrix of within-session across-session data based on given indices.

    Parameters:
    --------------
        Nperm (int): The number of permutations.
        D_t (numpy.ndarray): The preprocessed data array.
        idx_array (numpy.ndarray): The indices array.


    Returns:
    ----------  
        permutation_matrix (numpy.ndarray): The within-session continuos indices array.
    """
    permutation_matrix = np.zeros((D_t.shape[0],Nperm), dtype=int)
    for perm in range(Nperm):
        if perm == 0:
            permutation_matrix[:,perm] = np.arange(D_t.shape[0])
        else:
            idx_array_perm = permute_subject_trial_idx(idx_array)
            unique_indices = np.unique(idx_array_perm)
            positions_permute = [np.where(np.array(idx_array_perm) == i)[0] for i in unique_indices]
            permutation_matrix[:,perm] = np.concatenate(positions_permute,axis=0)
    return permutation_matrix


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

def calculate_baseline_difference(vpath_array, R_data, state, pairwise_statistic):
    """
    Calculate the difference between the specified statistics of a state and all other states combined.

    Parameters:
    --------------
        vpath_data (numpy.ndarray): The Viterbi path as of integer values that range from 1 to n_states.
        R_data (numpy.ndarray):     The dependent-variable associated with each state.
        state(numpy.ndarray):       the state for which the difference is calculated.
        pairwise_statistic (str)             The chosen statistic to be calculated. Valid options are "mean" or "median".

    Returns:
    ----------  
        difference (float)            the calculated difference between the specified state and all other states combined.
    """
    if pairwise_statistic == 'median':
        state_R_data = np.median(R_data[vpath_array == state])
        other_R_data = np.median(R_data[vpath_array != state])
    elif pairwise_statistic == 'mean':
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

def test_statistic_calculations(Din, Rin, perm, test_statistic, proj, method):
    from sklearn.cross_decomposition import CCA
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
        # Fit the original model 
        beta = proj @ Rin  # Calculate regression_coefficients (beta)
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
        
    elif method == "correlation":
        # Calculate correlation coefficient matrix
        corr_coef = np.corrcoef(Din, Rin, rowvar=False)
        corr_matrix = corr_coef[:Din.shape[1], Din.shape[1]:]
        # Update test_statistic
        test_statistic[perm, :, :] = np.abs(corr_matrix)
    elif method =="cca":
        # Create CCA object with 1 component
        cca = CCA(n_components=1)
        # Fit the CCA model to your data
        cca.fit(Din, Rin)
        # Transform the input data using the learned CCA model
        X_c, Y_c = cca.transform(Din, Rin)
        # Calcualte the correlation coefficients between X_c and Y_c
        corr_coef = np.corrcoef(X_c, Y_c, rowvar=False)[0, 1]
        # Update test_statistic
        test_statistic[perm] = np.abs(corr_coef)
        
    return test_statistic

def get_corr_coef(Din,Rin):
    # Calculate correlation coefficient matrix
    corr_coef = np.corrcoef(Din, Rin, rowvar=False)
    corr_matrix = corr_coef[:Din.shape[1], Din.shape[1]:]
    return corr_matrix


def pval_correction(pval, method='fdr_bh', alpha = 0.05):
    from statsmodels.stats import multitest as smt
    """
    Adjusts p-values for multiple testing

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
    rejected, p_values_corrected, _, _ = smt.multipletests(non_nan_values.flatten(), alpha=0.05, method=method, returnsorted=False)

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


def __palm_quickperms(EB, M=None, nP=1000, CMC=False, EE=True):
    # Call palm_quickperms from palm_functions
    return palm_quickperms(EB, M, nP, CMC, EE)
