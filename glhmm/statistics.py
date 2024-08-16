"""
Permutation testing from Gaussian Linear Hidden Markov Model
@author: Nick Y. Larsen 2023
"""

import numpy as np
import pandas as pd
import random
import math
import warnings
from tqdm import tqdm
from glhmm.palm_functions import *
from statsmodels.stats import multitest as smt
from sklearn.cross_decomposition import CCA
from collections import Counter
from skimage.measure import label, regionprops
from scipy.stats import ttest_ind, f_oneway, pearsonr, f, norm
from itertools import combinations
from sklearn.model_selection import train_test_split


def test_across_subjects(D_data, R_data, idx_data=None, method="multivariate", Nperm=0, confounds = None, 
                         dict_family = None,  within_group =False, between_groups=False,  verbose = True, test_statistics_option=False, 
                         FWER_correction=False, identify_categories=False, category_lim=10, 
                         test_combination=False):
    """
    Perform permutation testing across subjects. Family structure can be taken into account by inputting "dict_family".
    Three options are available to customize the statistical analysis to a particular research questions:
        - "multivariate": Perform permutation testing using regression analysis.
        - "univariate": Conduct permutation testing with correlation analysis.
        - "cca": Apply permutation testing using canonical correlation analysis.
        
    Parameters:
    --------------
    D_data (numpy.ndarray): 
        Input data array of shape that can be either a 2D array or a 3D array.
        For 2D, the data is represented as a (n, p) matrix, where n represents 
        the number of subjects, and p represents the number of predictors.
        For 3D array, it has a shape (T, n, p), where the first dimension 
        represents timepoints, the second dimension represents the number of subjects, 
        and the third dimension represents features. 
        For 3D, permutation testing is performed per timepoint for each subject.              
    R_data (numpy.ndarray): 
        The dependent variable can be either a 2D array or a 3D array. 
        For 2D array, it has a shape of (n, q), where n represents 
        the number of subjects, and q represents the outcome of the dependent variable.
        For 3D array, it has a shape (T, n, q), where the first dimension 
        represents timepoints, the second dimension represents the number of subjects, 
        and the third dimension represents a dependent variable.   
        For 3D, permutation testing is performed per timepoint for each subject.     
    idx_data (numpy.ndarray), default=None: 
        An array containing the indices for each group. The array can be either 1D or 2D:
        For a 1D array, a sequence of integers where each integer labels the group number. For example, [1, 1, 1, 1, 2, 2, 2, ..., N, N, N, N, N, N, N, N].
        For a 2D array, each row represents the start and end indices for the trials in a given session, with the format [[start1, end1], [start2, end2], ..., [startN, endN]].              
    method (str, optional), default="multivariate": 
        The statistical method to be used for the permutation test. Valid options are
        "multivariate", "univariate", or "cca".       
        Note: "cca" stands for Canonical Correlation Analysis                                        
    Nperm (int), default=0: 
        Number of permutations to perform.                       
    confounds (numpy.ndarray or None, optional), default=None: 
        The confounding variables to be regressed out from the input data (D_data).
        If provided, the regression analysis is performed to remove the confounding effects.    
    dict_family (dict): 
        Dictionary containing family structure information.                          
        - file_location (str): The file location of the family structure data in CSV format.
        - M (numpy.ndarray, optional): The matrix of attributes, which is not typically required.
                                    Defaults to None.
        - CMC (bool, optional), default=False: 
            A flag indicating whether to use the Conditional Monte Carlo method (CMC).
        - EE (bool, optional), default=True: A flag indicating whether to assume exchangeable errors, which allows permutation.
    verbose (bool, optional): 
        If True, display progress messages. If False, suppress progress messages.    
    within_group (bool, optional), default=False: 
        If True, the function will perfrom within group permutation      
    between_groups (bool, optional), default=False: 
        If True, the function will perfrom between group permutation                                                  
    test_statistics_option (bool, optional), default=False: 
        If True, the function will return the test statistics for each permutation.
    FWER_correction (bool, optional), default=False: 
        Specify whether to perform family-wise error rate (FWER) correction using the MaxT method.
        Note: FWER_correction is not necessary if pval_correction is applied later for multiple comparison p-value correction.                  
    identify_categories: bool or list or numpy.ndarray, optional, default=False
        If True, automatically identify categorical columns. If list or ndarray, use the provided list of column indices.    
    category_lim : int or None, optional, default=10
        Maximum allowed number of categories for F-test. Acts as a safety measure for columns 
        with integer values, like age, which may be mistakenly identified as multiple categories.        
    test_combination (bool or str, optional), default=False:
            Calculates geometric means of p-values using permutation testing.
            This is useful for summarizing statistical significance across multiple tests in experimental conditions.
            Valid options are False, True, "across_rows", or "across_columns".
            When method="multivariate":
                - True (bool): Returns a single geometric mean value (1 by 1).
            When method="univariate":
                - True (bool): Returns a single geometric mean value (1 by 1).
                - "across_rows" (str): Calculates geometric means aggregated along each row, returns mean value for each column (1 by p).
                - "across_columns" (str): Calculates geometric means aggregated along each column, returns mean value for each row (1 by q).                           
    Returns:
    ----------  
    result (dict): 
        A dictionary containing the following keys. Depending on the `test_statistics_option` and `method`, it can return the p-values, 
        correlation coefficients, test statistics.
        'pval': P-values for the test with shapes based on the method:
            - method=="multivariate": (T, p)
            - method=="univariate": (T, p, q)
            - method=="cca": (T, 1)
        'test_statistics': test statistics is the permutation distribution if `test_statistics_option` is True, else None.
            - method=="multivariate": (T, Nperm, p)
            - method=="univariate": (T, Nperm, p, q)
            - method=="cca": (T, Nperm, 1)
        'base_statistics': base statistics values of a giving test
        'test_type': the type of test, which is the name of the function
        'method': the method used for analysis Valid options are "multivariate", "univariate", or "cca".
        'max_correction': Specifies if FWER has been applied using MaxT, can either output True or False.  
        'performed_tests': A dictionary that marks the columns in the test_statistics or p-value matrix corresponding to the (q dimension) where t-tests or F-tests have been performed.
        'Nperm' :The number of permutations that has been performed.   
    """
    # Initialize variables
    test_type =  'test_across_subjects'
    
    # Ensure Nperm is at least 1
    if Nperm <= 1:
        Nperm = 1
        # Set flag for identifying categories without permutation
        identify_categories = True
        if method == 'cca' and Nperm == 1:
            raise ValueError("CCA does not support parametric statistics. The number of permutations ('Nperm') cannot be set to 1. "
                            "Please increase the number of permutations to perform a valid analysis.")

            
    
    # Check validity of method and data_type
    valid_methods = ["multivariate", "univariate", "cca"]
    validate_condition(method.lower() in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    # Check validity of method
    valid_test_combination = [False, True, "across_columns", "across_rows"]
    validate_condition(test_combination in valid_test_combination, "Invalid option specified for 'test_combination'. Must be one of: " + ', '.join(map(str, valid_test_combination)))
    
    if method=="multivariate" and test_combination in valid_test_combination[-2:]:
            raise ValueError("method is set to 'multivariate' and 'test_combination' is set to 'across_columns' or 'across_rows"
                         "If you want to perform 'test_combination' while doing 'multivariate' then please set 'test_combination' to 'True'.")

    if FWER_correction and test_combination in [True, "across_columns", "across_rows"]:
       # Raise an exception and stop function execution
        raise ValueError("'FWER_correction' is set to True and 'test_combination' is either True, 'columns', or 'rows'. "
                         "Please set 'FWER_correction' to False if you want to apply 'test_combination' or set 'test_combination' to False if you want to run 'FWER_correction'.")
        
    if (within_group or between_groups) and idx_data is None:
        # Raise an exception and stop function execution
        raise ValueError("Cannot perform within-group or between-group permutation without 'idx_data'. "
                        "Please provide 'idx_data' to define group boundaries for the permutation.")
  
    if not within_group and not between_groups and idx_data is not None:
        # Raise an exception and stop function execution
        raise ValueError("When providing 'idx_data' to define group boundaries, either 'within_group' or 'between_groups' must be set to True. "
                        "If neither is set to True, do not provide 'idx_data' and perform permutation testing across subjects instead.")
    
    # Get indices for permutation
    if idx_data is None:
        idx_array = None
    elif len(idx_data.shape)==2:
        idx_array = get_indices_array(idx_data)
    else:
        idx_array =idx_data.copy()  
    # Check if indices are unique
    _, idx_count =np.unique(idx_array, return_counts=True)  
    unique_diff =np.diff(idx_count)  
    if np.sum(unique_diff) !=0 and between_groups==True:
        # Raise an exception and stop function execution
        raise ValueError("The size of each group need to have a equal size to perform between group permutation"
                         "Please provide adjust your 'idx_data' or") 
    del unique_diff, idx_count    
    
    # Get the shapes of the data
    n_T, _, n_p, n_q, D_data, R_data = get_input_shape(D_data, R_data, verbose)
    # Note for convension we wrote (T, p, q) => (n_T, n_p, n_q)
    
    # Identify categorical columns in R_data
    category_columns = identify_coloumns_for_t_and_f_tests(R_data, method, Nperm, identify_categories, category_lim) if method=="univariate" or method =="multivariate" else {'t_test_cols': [], 'f_anova_cols': []}

    if category_columns["t_test_cols"]!=[] or category_columns["f_anova_cols"]!=[]:
        if FWER_correction and (len(category_columns.get('t_test_cols')) != R_data.shape[-1] or len(category_columns.get('f_anova_cols')) != R_data.shape[-1]):
            print("Warning: Cannot perform FWER_correction with different test statisticss.\nConsider to set identify_categories=False")
            raise ValueError("Cannot perform FWER_correction")  

    # Crate the family structure by looking at the dictionary 
    if dict_family is not None:
        # process dictionary of family structure
        dict_mfam=process_family_structure(dict_family, Nperm) 
        
    
    # Initialize arrays based on shape of data shape and defined options
    pval, base_statistics, test_statistics_list = initialize_arrays(R_data, n_p, n_q, n_T, method, Nperm, test_statistics_option, test_combination)
    

    for t in tqdm(range(n_T)) if n_T > 1 & verbose ==True else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        D_t, R_t = deconfound_values(D_data[t, :],R_data[t, :], confounds)
        
        # Removing rows that contain nan-values
        if method == "multivariate" or method == "cca":
            # Removing rows that contain nan-values
            D_t, R_t, _ = remove_nan_values(D_t, R_t, method)
        
        # Create test_statistics based on method
        test_statistics, reg_pinv = initialize_permutation_matrices(method, Nperm, n_p, n_q, D_t, test_combination, category_columns=category_columns)

        if dict_family is not None and idx_array is None:
            # Call function "__palm_quickperms" from glhmm.palm_functions
            permutation_matrix = __palm_quickperms(dict_mfam["EB"], M=dict_mfam["M"], nP=dict_mfam["nP"], 
                                                CMC=dict_mfam["CMC"], EE=dict_mfam["EE"])
            # Convert the index so it starts from 0
            permutation_matrix -= 1
            
        elif idx_array is None:
            # Get indices for permutation across subjects
            permutation_matrix = permutation_matrix_across_subjects(Nperm, R_t)
            
        elif idx_array is not None:
            if within_group:
                if between_groups:
                    # Permutation within and between groups
                    permutation_matrix = permutation_matrix_within_and_between_groups(Nperm, R_t, idx_array)
                else:
                    # Permutation within trials across sessions (within groups only)
                    permutation_matrix = permutation_matrix_across_trials_within_session(Nperm, R_t, idx_array)
            elif between_groups:
                # Permutation across sessions (between groups only)
                permutation_matrix = permutation_matrix_within_subject_across_sessions(Nperm, R_t, idx_array)

            
        for perm in tqdm(range(Nperm)) if n_T == 1 & verbose==True else range(Nperm):
            # Perform permutation on R_t
            Rin = R_t[permutation_matrix[:, perm]]
            # Calculate the permutation distribution
            test_statistics, bstat, pval_matrix = test_statistics_calculations(D_t, Rin, perm, test_statistics, reg_pinv, method, category_columns,test_combination)
            base_statistics[t, :] = bstat if perm == 0 and bstat is not None else base_statistics[t, :]
            pval[t, :] = pval_matrix if perm == 0 and pval_matrix is not None else pval[t, :]
        if Nperm>1:
            # Calculate p-values
            pval = get_pval(test_statistics, Nperm, method, t, pval, FWER_correction)
        
        # Output test statistics if it is set to True can be hard for memory otherwise
        if test_statistics_option==True:
            test_statistics_list[t,:] = test_statistics
            
    pval =np.squeeze(pval) # if np.abs(np.nansum(pval))>0 else np.nan 
    base_statistics =np.squeeze(base_statistics) if base_statistics is not None else [] 
    test_statistics_list =np.squeeze(test_statistics_list) if test_statistics_list is not None else []
    Nperm = 0 if Nperm==1 else Nperm
    
    if np.sum(np.isnan(pval)) > 0 and verbose:
        print("Warning: Permutation testing resulted in p-values that are NaN.")
        print("This could indicate an issue with the input data, such as:")
        print("  - More predictors than datapoints.")
        print("  - One or more features having identical values (no variability), making the F-statistic undefined.")
        print("Please review and clean your data before proceeding.")
    
    # Return results
    result = {
        'pval': pval,
        'base_statistics': base_statistics,
        'test_statistics': test_statistics_list,
        'test_type': test_type,
        'method': method,
        'test_combination': test_combination,
        'max_correction':FWER_correction,
        'performed_tests': category_columns,
        'Nperm': Nperm}
    return result



def test_across_trials_within_session(D_data, R_data, idx_data, method="multivariate", Nperm=0, confounds=None, 
                                      trial_timepoints=None,verbose=True, test_statistics_option=False, 
                                      FWER_correction=False, identify_categories=False, category_lim=10, 
                                      test_combination=False):
    """
    Perform permutation testing across different trials within a session. 
    An example could be if we want to test if any learning is happening during a session that might speed up times.
    
    Three options are available to customize the statistical analysis to a particular research questions:
        - 'multivariate': Perform permutation testing using regression analysis.
        - 'correlation': Conduct permutation testing with correlation analysis.
        - 'cca': Apply permutation testing using canonical correlation analysis.
             
    Parameters:
    --------------
    D_data (numpy.ndarray): 
        Input data array of shape that can be either a 2D array or a 3D array.
        For 2D array, it got a shape of (n, p), where n represent 
        the number of trials, and p represents the number of predictors (e.g., brain region)
        For a 3D array,it got a shape (T, n, p), where the first dimension 
        represents timepoints, the second dimension represents the number of trials, 
        and the third dimension represents features/predictors. 
        In the latter case, permutation testing is performed per timepoint for each subject.              
    R_data (numpy.ndarray): 
        The dependent-variable can be either a 2D array or a 3D array. 
        For 2D array, it got a shape of (n, q), where n represent 
        the number of trials, and q represents the outcome/dependent variable
        For a 3D array,it got a shape (T, n, q), where the first dimension 
        represents timepoints, the second dimension represents the number of trials, 
        and the third dimension represents a dependent variable                    
    idx_data (numpy.ndarray): 
        An array containing the indices for each session. The array can be either 1D or 2D:
        For a 1D array, a sequence of integers where each integer labels the session number. For example, [1, 1, 1, 1, 2, 2, 2, ..., N, N, N, N, N, N, N, N].
        For a 2D array, each row represents the start and end indices for the trials in a given session, with the format [[start1, end1], [start2, end2], ..., [startN, endN]].  
    method (str, optional), default="multivariate": 
        The statistical method to be used for the permutation test. Valid options are
        "multivariate", "univariate", or "cca".
        Note: "cca" stands for Canonical Correlation Analysis    
    Nperm (int), default=0: 
        Number of permutations to perform. 
    confounds (numpy.ndarray or None, optional), default=None: 
        The confounding variables to be regressed out from the input data (D_data).
        If provided, the regression analysis is performed to remove the confounding effects.    
    trial_timepoints (int), default=None: 
        Number of timepoints for each trial.  
    verbose (bool, optional), default=True: 
        If True, display progress messages. If False, suppress progress messages.                                                       
    test_statistics_option (bool, optional), default=False: 
        If True, the function will return the test statistics for each permutation.
    FWER_correction (bool, optional), default= False: 
        Specify whether to perform family-wise error rate (FWER) correction for multiple comparisons using the MaxT method.
        Note: FWER_correction is not necessary if pval_correction is applied later for multiple comparison p-value correction.                        
    identify_categories, default=False: 
        bool or list or numpy.ndarray, optional.
        If True, automatically identify categorical columns. If list or ndarray, use the provided list of column indices.    
    category_lim : int or None, optional, default=None
        Maximum allowed number of categories for F-test. Acts as a safety measure for columns 
        with integer values, like age, which may be mistakenly identified as multiple categories.     
    test_combination (bool or str, optional), default=False:
            Calculates geometric means of p-values using permutation testing.
            This is useful for summarizing statistical significance across multiple tests in experimental conditions.
            Valid options are False, True, "across_rows", or "across_columns".
            When method="multivariate":
                - True (bool): Returns a single geometric mean value (1 by 1).
            When method="univariate":
                - True (bool): Returns a single geometric mean value (1 by 1).
                - "across_rows" (str): Calculates geometric means aggregated along each row, returns mean value for each column (1 by p).
                - "across_columns" (str): Calculates geometric means aggregated along each column, returns mean value for each row (1 by q).                                         
    
    multivariate
    
    Returns:
    ----------  
    result (dict): A dictionary containing the following keys. Depending on the `test_statistics_option` and `method`, it can return the p-values, 
        correlation coefficients, test statistics.
        'pval': P-values for the test with shapes based on the method:
            - method=="multivariate": (T, p)
            - method=="univariate": (T, p, q)
            - method=="cca": (T, 1)
        'test_statistics': test statistics is the permutation distribution if `test_statistics_option` is True, else None.
            - method=="multivariate": (T, Nperm, p)
            - method=="univariate": (T, Nperm, p, q)
            - method=="cca": (T, Nperm, 1)
        'base_statistics': base statistics of a given test
        'test_type': the type of test, which is the name of the function
        'method': the method used for analysis Valid options are:
                "multivariate", "univariate", or "cca".
        'max_correction': Specifies if FWER has been applied using MaxT, can either output True or False.
        'Nperm' :The number of permutations that has been performed.   
    """
    # Initialize variable
    category_columns = []   
    test_type =  'test_across_trials'
    
    # Ensure Nperm is at least 1
    if Nperm <= 1:
        Nperm = 1
        # Set flag for identifying categories without permutation
        identify_categories = True
        if method == 'cca' and Nperm == 1:
            raise ValueError("CCA does not support parametric statistics. The number of permutations ('Nperm') cannot be set to 1. "
                            "Please increase the number of permutations to perform a valid analysis.")

    # Check validity of method and data_type
    valid_methods = ["multivariate", "univariate", "cca"]
    validate_condition(method.lower() in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    # Check validity of method
    valid_test_combination = [False, True, "across_columns", "across_rows"]
    validate_condition(test_combination in valid_test_combination, "Invalid option specified for 'test_combination'. Must be one of: " + ', '.join(map(str, valid_test_combination)))
    
    if method=="multivariate" and test_combination in valid_test_combination[-2:]:
            raise ValueError("method is set to 'multivariate' and 'test_combination' is set to 'across_columns' or 'across_rows"
                         "If you want to perform 'test_combination' while doing 'multivariate' then please set 'test_combination' to 'True'.")


    if FWER_correction and test_combination in [True, "across_columns", "across_rows"]:
       # Raise an exception and stop function execution
        raise ValueError("'FWER_correction' is set to True and 'test_combination' is either True, 'columns', or 'rows'. "
                         "Please set 'FWER_correction' to False if you want to apply 'test_combination' or set 'test_combination' to False if you want to run 'FWER_correction'.")

    # Get input shape information
    n_T, _, n_p, n_q, D_data, R_data = get_input_shape(D_data, R_data, verbose)

    # Identify categorical columns in R_data
    category_columns = identify_coloumns_for_t_and_f_tests(R_data, method, Nperm, identify_categories, category_lim) if method=="univariate" or method =="multivariate" else {'t_test_cols': [], 'f_anova_cols': []}
    
    if category_columns["t_test_cols"]!=[] or category_columns["f_anova_cols"]!=[]:
        if FWER_correction and (len(category_columns.get('t_test_cols')) != R_data.shape[-1] or len(category_columns.get('f_anova_cols')) != R_data.shape[-1]):
            print("Warning: Cannot perform FWER_correction with different test statisticss.\nConsider to set identify_categories=False")
            raise ValueError("Cannot perform FWER_correction")
    
    # Get indices for permutation
    if len(idx_data.shape)==2:
        idx_array = get_indices_array(idx_data)
    else:
        idx_array =idx_data.copy()        

    # Initialize arrays based on shape of data shape and defined options
    pval, base_statistics, test_statistics_list = initialize_arrays(R_data, n_p, n_q, n_T, method, Nperm, test_statistics_option, test_combination)

    for t in tqdm(range(n_T)) if n_T > 1 & verbose ==True else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        D_t, R_t = deconfound_values(D_data[t, :],R_data[t, :], confounds)
        
        # Removing rows that contain nan-values
        if method == "multivariate" or method == "cca":
            D_t, R_t, nan_mask = remove_nan_values(D_t, R_t, method)
            if np.sum(nan_mask)>0:
                # Update indices
                idx_array = idx_array.copy()[~nan_mask]
                idx_data_in=get_indices_update_nan(idx_data_in,~nan_mask)
                
        # Create test_statistics and the regularized pseudo-inverse of D_data
        test_statistics, reg_pinv = initialize_permutation_matrices(method, Nperm, n_p, n_q, D_t, test_combination, category_columns=category_columns)
    

        # Calculate permutation matrix of D_t 
        permutation_matrix = permutation_matrix_across_trials_within_session(Nperm,R_t, idx_array,trial_timepoints)
                
        for perm in range(Nperm):
        #for perm in tqdm(range(Nperm)) if n_T == 1 else range(n_T):
            # Perform permutation on R_t
            Rin = R_t[permutation_matrix[:, perm]]
            # Calculate the permutation distribution
            test_statistics, bstat, pval_matrix = test_statistics_calculations(D_t, Rin, perm, test_statistics, reg_pinv, method, category_columns,test_combination)
            base_statistics[t, :] = bstat if perm == 0 and bstat is not None else base_statistics[t, :]
            pval[t, :] = pval_matrix if perm == 0 and pval_matrix is not None else pval[t, :]
            if Nperm>1:
                # Calculate p-values
                pval = get_pval(test_statistics, Nperm, method, t, pval, FWER_correction)
            
            # Output test statistics if it is set to True can be hard for memory otherwise
            if test_statistics_option==True:
                test_statistics_list[t,:] = test_statistics
            
    pval =np.squeeze(pval) # if np.abs(np.nansum(pval))>0 else np.nan
    base_statistics =np.squeeze(base_statistics) if base_statistics is not None  else []
    test_statistics_list =np.squeeze(test_statistics_list) if test_statistics_list is not None else []
    Nperm = 0 if Nperm==1 else Nperm
    
    if np.sum(np.isnan(pval))>0 & verbose:
        print("Warning: Permutation testing resulted in p-values equal to NaN.")
        print("This may indicate an issue with the input data. Please review your data.")
        
    # Return results
    result = {
        'pval': pval,
        'base_statistics': base_statistics,
        'test_statistics': test_statistics_list,
        'test_type': test_type,
        'method': method,
        'test_combination': test_combination,
        'max_correction':FWER_correction,
        'performed_tests': category_columns,
        'Nperm': Nperm}
    
    return result

def test_across_sessions_within_subject(D_data, R_data, idx_data, method="multivariate", Nperm=0, confounds=None, 
                                        verbose = True, test_statistics_option=False, FWER_correction=False, 
                                        test_combination=False, identify_categories=False, 
                                        category_lim=10):
    """
    Perform permutation testing across sessions within the same subject, while keeping the trial order the same.
    This procedure is particularly valuable for investigating the effects of long-term treatments or monitoring changes in brain responses across sessions over time.
    Three options are available to customize the statistical analysis to a particular research questions:
        - 'multivariate': Perform permutation testing using regression analysis.
        - 'correlation': Conduct permutation testing with correlation analysis.
        - 'cca': Apply permutation testing using canonical correlation analysis.
           
    Parameters:
    --------------
    D_data (numpy.ndarray): 
        Input data array of shape that can be either a 2D array or a 3D array.
        For 2D array, it got a shape of (n, p), where n represent 
        the number of trials of the dataset, and each column represents a feature (e.g., brain region). 
        For a 3D array,it got a shape (T, n, p), where the first dimension 
        represents timepoints, the second dimension represents the number of trials, 
        and the third dimension represents features/predictors.             
    R_data (numpy.ndarray): 
        The dependent-variable can be either a 2D array or a 3D array. 
        For 2D array, it got a shape of (n, q), where n represent 
        the number of trials, and q represents the outcome/dependent variable
        For a 3D array,it got a shape (T, n, q), where the first dimension 
        represents timepoints, the second dimension represents the number of trials, 
        and the third dimension represents a dependent variable                   
    idx_data (numpy.ndarray): 
        An array containing the indices for each session. The array can be either 1D or 2D:
        For a 1D array, a sequence of integers where each integer labels the session number. For example, [1, 1, 1, 1, 2, 2, 2, ..., N, N, N, N, N, N, N, N].
        For a 2D array, each row represents the start and end indices for the trials in a given session, with the format [[start1, end1], [start2, end2], ..., [startN, endN]].  
    method (str, optional), default="multivariate": 
        The statistical method to be used for the permutation test. Valid options are
        "multivariate", "univariate", or "cca".
        Note: "cca" stands for Canonical Correlation Analysis    
    Nperm (int), default=0: 
        Number of permutations to perform.                
    confounds (numpy.ndarray or None, optional): 
        The confounding variables to be regressed out from the input data (D_data).
        If provided, the regression analysis is performed to remove the confounding effects. (default: None):                           
    verbose (bool, optional), default=False: 
        If True, display progress messages and prints. If False, suppress messages.                                                             
    test_statistics_option (bool, optional), default=False: 
        If True, the function will return the test statistics for each permutation.
    FWER_correction (bool, optional), default=False: 
        Specify whether to perform family-wise error rate (FWER) correction for multiple comparisons using the MaxT method.
        Note: FWER_correction is not necessary if pval_correction is applied later for multiple comparison p-value correction.
    test_combination (bool or str, optional), default=False:
            Calculates geometric means of p-values using permutation testing.
            This is useful for summarizing statistical significance across multiple tests in experimental conditions.
            Valid options are False, True, "across_rows", or "across_columns".
            When method="multivariate":
                - True (bool): Returns a single geometric mean value (1 by 1).
            When method="univariate":
                - True (bool): Returns a single geometric mean value (1 by 1).
                - "across_rows" (str): Calculates geometric means aggregated along each row, returns mean value for each column (1 by p).
                - "across_columns" (str): Calculates geometric means aggregated along each column, returns mean value for each row (1 by q).     
    category_lim : int or None, optional, default=10
        Maximum allowed number of categories for F-test. Acts as a safety measure for columns 
        with integer values, like age, which may be mistakenly identified as multiple categories.                              
    
    Returns:
    ----------  
    result (dict): 
        A dictionary containing the following keys. Depending on the `test_statistics_option` and `method`, it can return the p-values, 
        correlation coefficients, test statistics.
        'pval': P-values for the test with shapes based on the method:
            - method=="multivariate": (T, p)
            - method=="univariate": (T, p, q)
            - method=="cca": (T, 1)
        'test_statistics': test statistics is the permutation distribution if `test_statistics_option` is True, else None.
            - method=="multivariate": (T, Nperm, p)
            - method=="univariate": (T, Nperm, p, q)
            - method=="cca": (T, Nperm, 1)
        'base_statistics': base statistics of a given test
        'test_type': the type of test, which is the name of the function
        'method': the method used for analysis Valid options are
                "multivariate", "univariate", or "cca" (default: "multivariate").
        'max_correction': Specifies if FWER has been applied using MaxT, can either output True or False.
        'Nperm' :The number of permutations that has been performed.
                  
    """ 
    # Initialize variable
    category_columns = []  
    test_type = 'test_across_sessions'  
    permute_beta = True # For across session test we are permuting the beta coefficients for each session
     
    # Ensure Nperm is at least 1
    if Nperm <= 1:
        Nperm = 1
        # Set flag for identifying categories without permutation
        identify_categories = True
        if method == 'cca' and Nperm == 1:
            raise ValueError("CCA does not support parametric statistics. The number of permutations ('Nperm') cannot be set to 1. "
                            "Please increase the number of permutations to perform a valid analysis.")

     
    # Check validity of method and data_type
    valid_methods = ["multivariate", "univariate", "cca"]
    validate_condition(method.lower() in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    # Check validity of method
    valid_test_combination = [False, True, "across_columns", "across_rows"]
    validate_condition(test_combination in valid_test_combination, "Invalid option specified for 'test_combination'. Must be one of: " + ', '.join(map(str, valid_test_combination)))
    
    if method=="multivariate" and test_combination in valid_test_combination[-2:]:
            raise ValueError("method is set to 'multivariate' and 'test_combination' is set to 'across_columns' or 'across_rows"
                         "If you want to perform 'test_combination' while doing 'multivariate' then please set 'test_combination' to 'True'.")

    if FWER_correction and test_combination in [True, "across_columns", "across_rows"]:
       # Raise an exception and stop function execution
        raise ValueError("'FWER_correction' is set to True and 'test_combination' is either True, 'columns', or 'rows'. "
                         "Please set 'FWER_correction' to False if you want to apply 'test_combination' or set 'test_combination' to False if you want to run 'FWER_correction'.")
        
    # Get indices of the sessions
    idx_array = get_indices_array(idx_data)

    # Calculate the maximum number of permutations
    max_permutations = math.factorial(len(np.unique(idx_array)))
    # Using scientific notation format with string formatting
    exp_notation = "{:.2e}".format(max_permutations)
    if Nperm > max_permutations:
        warnings.warn(f"Maximum number of permutations with {len(np.unique(idx_array))} sessions is: {exp_notation}. "
                    "Reduce the number of permutations to the maximum number of permutations to run the test properly.")
    if verbose:
        print(f"Maximum number of permutations with {len(np.unique(idx_array))} sessions is: {exp_notation}")
    
    # Get input shape information
    n_T, _, n_p, n_q, D_data, R_data = get_input_shape(D_data, R_data, verbose)
    
    # Identify categorical columns in R_data
    category_columns = identify_coloumns_for_t_and_f_tests(R_data, method, Nperm, identify_categories, category_lim) if method=="univariate" or method =="multivariate" else {'t_test_cols': [], 'f_anova_cols': []}

    # Initialize arrays based on shape of data shape and defined options
    pval, base_statistics, test_statistics_list = initialize_arrays(R_data, n_p, n_q, n_T, method, Nperm, test_statistics_option, test_combination)

    for t in tqdm(range(n_T)) if n_T > 1 & verbose==True else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        D_t, R_t = deconfound_values(D_data[t, :],R_data[t, :], confounds)
        
        # Removing rows that contain nan-values
        if method == "multivariate" or method == "cca":
            D_t, R_t,_ = remove_nan_values(D_t, R_t, method)

        # Create test_statistics and pval_perms based on method
        test_statistics, reg_pinv = initialize_permutation_matrices(method, Nperm, n_p, n_q, D_t, test_combination, permute_beta, category_columns)
        
        # Calculate the beta coefficient of each session
        beta, test_indices_list = calculate_beta_session(reg_pinv, R_t, idx_data, permute_beta, category_lim)
        # Old code
        #permutation_matrix = permutation_matrix_within_subject_across_sessions(Nperm, D_t, idx_array)

        for perm in range(Nperm):
            # Calculate the permutation distribution
            test_indices =[np.concatenate(indices, axis=0)  for indices in test_indices_list]
            idx_data_in_test =get_indices_from_list(test_indices_list[0])
            test_statistics, bstat, pval_matrix = test_statistics_calculations(D_t, R_t, perm, test_statistics, reg_pinv, method, category_columns, test_combination, idx_data_in_test, permute_beta, beta, test_indices)
            base_statistics[t, :] = bstat if perm == 0 and bstat is not None else base_statistics[t, :]
            pval[t, :] = pval_matrix if perm == 0 and pval_matrix is not None else pval[t, :]
            if Nperm>1:
                # Calculate p-values
                pval = get_pval(test_statistics, Nperm, method, t, pval, FWER_correction)
            
            # Output test statistics if it is set to True can be hard for memory otherwise
            if test_statistics_option==True:
                test_statistics_list[t,:] = test_statistics
            
    pval =np.squeeze(pval) # if np.abs(np.nansum(pval))>0 else np.nan 
    base_statistics =np.squeeze(base_statistics) if base_statistics is not None  else []
    test_statistics_list =np.squeeze(test_statistics_list) if test_statistics_list is not None  else []
    Nperm = 0 if Nperm==1 else Nperm    
    if np.isscalar(pval) is True:
        print("Exporting the base statistics only")
    elif np.sum(np.isnan(pval))>0 & verbose:
        print("Warning: Permutation testing resulted in p-values equal to NaN.")
        print("This may indicate an issue with the input data. Please review your data.")
              
    # Return results
    result = {
        'pval': pval,
        'base_statistics': base_statistics,
        'test_statistics': test_statistics_list,
        'test_type': test_type,
        'method': method,
        'test_combination': test_combination,
        'max_correction':FWER_correction,
        'performed_tests': category_columns,
        'Nperm': Nperm}
    return result

def test_across_visits(input_data, vpath_data, n_states, method="multivariate", Nperm=0, verbose = True, 
                       confounds=None, test_statistics_option=False, pairwise_statistic ="mean",
                       FWER_correction=False, category_lim=10, identify_categories = False, 
                       vpath_surrogates=None):
    """
    Perform permutation testing across Viterbi path for continuous data.
    
    Parameters:
    --------------            
    input_data (numpy.ndarray): 
        Dependent variable with shape (n, q), where n is the number of samples (n_timepoints x n_trials), 
        and q represents dependent/target variables.  
    vpath_data (numpy.ndarray): 
        The hidden state path data of the continuous measurements represented as a (n, p) matrix. 
        It could be a 2D matrix where each row represents a trials over a period of time and
        each column represents a state variable and gives the shape ((n_timepoints X n_trials), n_states). 
        If it is a 1D array of of shape ((n_timepoints X n_trials),) where each row value represent a giving state.                                 
    n_states (int):             The number of hidden states in the hidden state path data.
    method (str, optional), default="multivariate":     
        Statistical method for the permutation test. Valid options are 
        "multivariate", "univariate", "cca", "one_vs_rest" or "state_pairs". 
        Note: "cca" stands for Canonical Correlation Analysis.   
    Nperm (int), default=0:                
        Number of permutations to perform. 
    verbose (bool, optional): 
        If True, display progress messages. If False, suppress progress messages.
    test_statistics_option (bool, optional), default=False: 
        If True, the function will return the test statistics for each permutation.
    pairwise_statistic (str, optional), default="mean".  
        The chosen statistic when applying methods "one_vs_rest" or "state_pairs". 
        Valid options are "mean" or "median".
    FWER_correction (bool, optional), default=False: 
        Specify whether to perform family-wise error rate (FWER) correction for multiple comparisons using the MaxT method.
        Note: FWER_correction is not necessary if pval_correction is applied later for multiple comparison p-value correction.
    category_lim : int or None, optional, default=None
        Maximum allowed number of categories for F-test. Acts as a safety measure for columns 
        with integer values, like age, which may be mistakenly identified as multiple categories.                   
    
    Returns:
    ----------  
    result (dict): A dictionary containing the following keys. Depending on the `test_statistics_option` and `method`, it can return the p-values, 
        correlation coefficients, test statistics.
        'pval': P-values for the test with shapes based on the method:
            - method=="multivariate": (T, p)
            - method=="univariate": (T, p, q)
            - method=="cca": (T, 1)
            - method=="one_vs_rest": (T, p, 1)
            - method=="state_pairs": (T, p, p)
        'test_statistics': test statistics is the permutation distribution if `test_statistics_option` is True, else None.
            - method=="multivariate": (T, Nperm, p)
            - method=="univariate": (T, Nperm, p, q)
            - method=="cca": (T, Nperm, 1)
            - method=="one_vs_rest": (T, Nperm, p)
            - method=="state_pairs": (T, Nperm, 1)
        'base_statistics': base statistics of a given test
        'test_type': the type of test, which is the name of the function
        'method': the method used for analysis Valid options are
                "multivariate", "univariate", or "cca", "one_vs_rest" and "state_pairs" (default: "multivariate").
        'max_correction': Specifies if FWER has been applied using MaxT, can either output True or False.
        'Nperm' :The number of permutations that has been performed.
    """
    # Initialize variables
    test_type = 'test_across_visits'
      
    # Ensure Nperm is at least 1
    if Nperm <= 1:
        Nperm = 1
        # Set flag for identifying categories without permutation
        identify_categories = True
        if method == 'cca' or method =='one_vs_rest' or method =='state_pairs' and Nperm == 1:
            raise ValueError("'cca', 'one_vs_rest' and 'state_pairs' does not support parametric statistics. The number of permutations ('Nperm') cannot be set to 1. "
                            "Please increase the number of permutations to perform a valid analysis.")

         
    # Check if the Viterbi path is correctly constructed
    if vpath_check_2D(vpath_data) == False:
        raise ValueError(
            "'vpath_data' is not correctly formatted. Ensure that the Viterbi path is correctly positioned within the target matrix (R). "
            "The data should be one-hot encoded if it's 2D, or an array of integers if it's 1D. "
            "Please verify your input to the 'test_across_visits' function."
        )
   
    # Check validity of method
    valid_methods = ["multivariate", "univariate", "cca", "one_vs_rest", "state_pairs"]
    validate_condition(method.lower() in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    valid_statistic = ["mean", "median"]
    validate_condition(pairwise_statistic.lower() in valid_statistic, "Invalid option specified for 'statistic'. Must be one of: " + ', '.join(valid_statistic))
    
    # Convert vpath from matrix to vector
    vpath_array=generate_vpath_1D(vpath_data)
    if vpath_data.ndim == 1:
        vpath_bin = np.zeros((len(vpath_data), len(np.unique(vpath_data))))
        vpath_bin[np.arange(len(vpath_data)), vpath_data - 1] = 1
        vpath_data = vpath_bin.copy()


    # Get input shape information
    n_T, _, n_p, n_q, input_data, vpath_data= get_input_shape(input_data, vpath_data, verbose)  

    # Identify categorical columns in R_data
    category_columns = identify_coloumns_for_t_and_f_tests(vpath_data, method, Nperm, identify_categories, category_lim)
    
    # Identify categorical columns
    if category_columns["t_test_cols"]!=[] or category_columns["f_anova_cols"]!=[]:
        if FWER_correction and (len(category_columns.get('t_test_cols')) != vpath_data.shape[-1] or len(category_columns.get('f_anova_cols')) != vpath_data.shape[-1]):
            print("Warning: Cannot perform FWER_correction with different test statisticss.\nConsider to set identify_categories=False")
            raise ValueError("Cannot perform FWER_correction")    
   
    # Initialize arrays based on shape of data shape and defined options
    pval, base_statistics, test_statistics_list = initialize_arrays(vpath_data, n_p, n_q,
                                                                            n_T, method, Nperm,
                                                                            test_statistics_option)


    # Print tqdm over n_T if there are more than one timepoint
    for t in tqdm(range(n_T)) if n_T > 1 & verbose==True else range(n_T):
        # Correct for confounds and center data_t
        data_t, _ = deconfound_values(input_data[t, :],None, confounds)
        
        # Removing rows that contain nan-values
        if method == "multivariate" or method == "cca":
            data_t, vpath_array, _ = remove_nan_values(data_t, vpath_array, method)
        
        if method != "state_pairs":
            ###################### Permutation testing for other tests beside state pairs #################################
            # Create test_statistics and pval_perms based on method
            test_statistics, reg_pinv = initialize_permutation_matrices(method, Nperm, n_p, n_q, data_t, category_columns=category_columns)
            # Perform permutation testing
            for perm in tqdm(range(Nperm)) if n_T == 1 & verbose==True else range(n_T):
                # Redo vpath_surrogate calculation if the number of states are not the same (out of 1000 permutations it happens maybe 1-2 times with this demo dataset)
                
                if vpath_surrogates is None:
                    while True:
                        # Create vpath_surrogate
                        vpath_surrogate = surrogate_state_time(perm, vpath_array, n_states)
                        if len(np.unique(vpath_surrogate)) == n_states:
                            break  # Exit the loop if the condition is satisfied
                else:
                    vpath_surrogate = vpath_surrogates[:,perm].astype(int)
                        
                if method =="one_vs_rest":
                    for state in range(1, n_states+1):
                        test_statistics[perm,state -1] =calculate_baseline_difference(vpath_surrogate, data_t, state, pairwise_statistic.lower(), category_columns)
    
                        
                elif method =="multivariate":
                    # Make vpath to a binary matrix
                    vpath_surrogate_binary = np.zeros((len(vpath_surrogate), len(np.unique(vpath_surrogate))))
                    # Set the appropriate positions to 1
                    vpath_surrogate_binary[np.arange(len(vpath_surrogate)), vpath_surrogate - 1] = 1
                    test_statistics, bstat, pval_matrix = test_statistics_calculations(data_t,vpath_surrogate_binary , perm,
                                                                            test_statistics, reg_pinv, method, category_columns)
                    base_statistics[t, :] = bstat if perm == 0 and bstat is not None else base_statistics[t, :]
                    pval[t, :] = pval_matrix if perm == 0 and pval_matrix is not None else pval[t, :]
                else:
                    # Univariate test
                    # Apply 1 hot encoding
                    vpath_surrogate_onehot = viterbi_path_to_stc(vpath_surrogate,n_states)
                    # Apply t-statistic on the vpath_surrogate
                    test_statistics, bstat, pval_matrix = test_statistics_calculations(data_t, vpath_surrogate_onehot, perm, 
                                                                          test_statistics, reg_pinv, method, category_columns)
                    base_statistics[t, :] = bstat if perm == 0 and bstat is not None else base_statistics[t, :]
                    pval[t, :] = pval_matrix if perm == 0 and pval_matrix is not None else pval[t, :]
                    
            if Nperm>1:
                # Calculate p-values
                pval = get_pval(test_statistics, Nperm, method, t, pval, FWER_correction)
        ###################### Permutation testing for state pairs #################################
        elif method =="state_pairs":
            # Run this code if it is "state_pairs"
            # Correct for confounds and center data_t
            data_t, _ = deconfound_values(input_data[t, :],None, confounds)
            
            # Generates all unique combinations of length 2 
            pairwise_comparisons = list(combinations(range(1, n_states + 1), 2))
            test_statistics = np.zeros((Nperm, len(pairwise_comparisons)))
            pval = np.zeros((n_states, n_states))
            # Iterate over pairwise state comparisons
            
            for idx, (state_1, state_2) in tqdm(enumerate(pairwise_comparisons), total=len(pairwise_comparisons), desc="Pairwise comparisons") if verbose ==True else enumerate(pairwise_comparisons):    
                # Generate surrogate state-time data and calculate differences for each permutation
                for perm in range(Nperm):
                    
                    if vpath_surrogates is None:
                        while True:
                            # Create vpath_surrogate
                            vpath_surrogate = surrogate_state_time(perm, vpath_array, n_states)
                            if len(np.unique(vpath_surrogate)) == n_states:
                                break  # Exit the loop if the condition is satisfied
                    else:
                        vpath_surrogate = vpath_surrogates[:,perm].astype(int)
                        
                    test_statistics[perm,idx] = calculate_statepair_difference(vpath_surrogate, data_t, state_1, 
                                                                               state_2, pairwise_statistic)
                
                if Nperm>1:
                    p_val= np.sum(test_statistics[:,idx] >= test_statistics[0,idx], axis=0) / (Nperm + 1)
                    pval[state_1-1, state_2-1] = p_val
                    pval[state_2-1, state_1-1] = 1 - p_val
            # Fill numbers in base statistics
            if  np.sum(base_statistics[t, :])==0:
                base_statistics[t, :] =test_statistics[0,:]
                
        if test_statistics_option:
            test_statistics_list[t, :] = test_statistics

    pval =np.squeeze(pval) # if np.abs(np.nansum(pval))>0 else np.nan 
    base_statistics =np.squeeze(base_statistics) if base_statistics is not None else []
    test_statistics_list =np.squeeze(test_statistics_list) if test_statistics_list is not None else []
    Nperm = 0 if Nperm==1 else Nperm
    
    if np.sum(np.isnan(pval))>0 & verbose:
        print("Warning: Permutation testing resulted in p-values equal to NaN.")
        print("This may indicate an issue with the input data. Please review your data.")
        
    # Return results
    result = {
        'pval': pval,
        'base_statistics': base_statistics,
        'test_statistics': test_statistics_list,
        'test_type': test_type,
        'method': method,
        'max_correction':FWER_correction,
        'performed_tests': category_columns,
        'Nperm': Nperm}
    return result

def remove_nan_values(D_data, R_data, method):
    """
    Remove rows with NaN values from input data arrays.

    Parameters:
    -----------
    D_data (numpy.ndarray)
        Input data array containing features.
    R_data (numpy.ndarray): 
        Input data array containing response values.
    method (str, optional), default="multivariate":     
        Statistical method for the permutation test. Valid options are 
        "multivariate", "univariate", "cca", "one_vs_rest" or "state_pairs". 
        Note: "cca" stands for Canonical Correlation Analysis.   
    
    Returns:
    ---------
    D_data (numpy.ndarray): 
        Cleaned feature data (D_data) with NaN values removed.  
    R_data (numpy.ndarray): 
        Cleaned response data (R_data) with NaN values removed.
    nan_mask(bool)
        Array that mask the position of the NaN values with True and False for non-nan values
    """
    FLAG = 0
    # removed_indices = None
    
    if R_data.ndim == 1:
        FLAG = 1
        R_data = R_data.reshape(-1,1) 
    if method == "multivariate":
        # When applying "multivariate" we need to remove rows for our D_data, as we cannot use it as a predictor for
        # Check for NaN values and remove corresponding rows
        nan_mask = np.isnan(D_data).any(axis=1)
        # nan_mask = np.isnan(D_data).any(axis=1)
        # Get indices or rows that have been removed
        # removed_indices = np.where(nan_mask)[0]

        D_data = D_data[~nan_mask]
        R_data = R_data[~nan_mask]
    elif method== "cca":
        # When applying cca we need to remove rows at both D_data and R_data
        # Check for NaN values and remove corresponding rows
        nan_mask = np.isnan(D_data).any(axis=1) | np.isnan(R_data).any(axis=1)
        # Remove nan indices
        D_data = D_data[~nan_mask]
        R_data = R_data[~nan_mask]
    if FLAG ==1:
        # Flatten R_data
        R_data =R_data.flatten()
    return D_data, R_data, nan_mask

def validate_condition(condition, error_message):
    """
    Validates a given condition and raises a ValueError with the specified error message if the condition is not met.

    Parameters:
    --------------
    condition (bool): 
        The condition to check.
    error_message (str): 
        The error message to raise if the condition is not met.
    """
    # Check if a condition is False and raise a ValueError with the given error message
    if not condition:
        raise ValueError(error_message)


def get_input_shape(D_data, R_data, verbose):
    """
    Computes the input shape parameters for permutation testing.

    Parameters:
    --------------
    D_data (numpy.ndarray): 
        The input data array.
    R_data (numpy.ndarray): 
        The dependent variable.
    verbose (bool): 
        If True, display progress messages. If False, suppress progress messages.

    Returns:
    ----------  
    n_T (int): 
        The number of timepoints.
    n_ST (int): 
        The number of subjects or trials.
    n_p (int): 
        The number of features.
    D_data (numpy.ndarray): 
        The updated input data array.
    R_data (numpy.ndarray): 
        The updated dependent variable.
    """
    # Get the input shape of the data and perform necessary expansions if needed
    if R_data.ndim == 1:
        R_data = np.expand_dims(R_data, axis=1)
    
    if len(D_data.shape) == 1:
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
        if verbose:
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
        EB (numpy.ndarray): 
            Block structure representing relationships between subjects.
        M (numpy.ndarray, optional), default=None: 
            The matrix of attributes, which is not typically required.
        nP (int): 
            The number of permutations to generate.
        CMC (bool, optional), default=False: 
            A flag indicating whether to use the Conditional Monte Carlo method (CMC).
        EE (bool, optional), default=True: 
            A flag indicating whether to assume exchangeable errors, which allows permutation.
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

def initialize_arrays(R_data, n_p, n_q, n_T, method, Nperm, test_statistics_option, test_combination=False):
    """
    Initializes arrays for permutation testing.

    Parameters:
    --------------
    R_data (numpy.ndarray): 
        The dependent variable
    n_p (int): 
        The number of features.
    n_q (int): 
        The number of predictions.
    n_T (int): 
        The number of timepoints.
    method (str): 
        The method to use for permutation testing.
    Nperm (int): 
        Number of permutations.
    test_statistics_option (bool): 
        If True, return the test statistics values.
    test_combination (str), default=False:       
        Specifies the combination method.
        Valid options: "True", "across_columns", "across_rows".

    Returns:
    ----------  
    pval (numpy array): 
        p-values for the test (n_T, n_p) if test_statistics_option is False, else None.
    base_statistics (numpy array): 
        base statistics of a given test
    test_statistics_list (numpy array): 
        test statistics values (n_T, Nperm, n_p) or (n_T, Nperm, n_p, n_q) if method="univariate" , else None.
    """
    
    # Initialize the arrays based on the selected method and data dimensions
    if  method == "multivariate":
        if test_combination in [True, "across_columns", "across_rows"]: 
            pval = np.zeros((n_T, 1))
            if test_statistics_option==True:
                test_statistics_list = np.zeros((n_T, Nperm, 1))
            else:
                test_statistics_list= None
            base_statistics= np.zeros((n_T, 1, 1))
        else:
            pval = np.zeros((n_T, n_q))
        
            if test_statistics_option==True:
                test_statistics_list = np.zeros((n_T, Nperm, n_q))
            else:
                test_statistics_list= None
            base_statistics= np.zeros((n_T, 1, n_q))
        
    elif  method == "cca":
        pval = np.zeros((n_T, 1))
        if test_statistics_option==True:
            test_statistics_list = np.zeros((n_T, Nperm, 1))
        else:
            test_statistics_list= None
        base_statistics= np.zeros((n_T, 1, 1))        
    elif method == "univariate" :  
        if test_combination in [True, "across_columns", "across_rows"]: 
            pval_shape = (n_T, 1) if test_combination == True else (n_T, n_q) if test_combination == "across_columns" else (n_T, n_p)
            pval = np.zeros(pval_shape)
            base_statistics = pval.copy()
            if test_statistics_option:
                test_statistics_list_shape = (n_T, Nperm, 1) if test_combination == True else (n_T, Nperm, n_q) if test_combination == "across_columns" else (n_T, Nperm, n_p)
                test_statistics_list = np.zeros(test_statistics_list_shape)
            else:
                test_statistics_list = None
        else:    
            pval = np.zeros((n_T, n_p, n_q))
            base_statistics = pval.copy()
            if test_statistics_option==True:    
                test_statistics_list = np.zeros((n_T, Nperm, n_p, n_q))
            else:
                test_statistics_list= None
    elif method == "state_pairs":
        pval = np.zeros((n_T, R_data.shape[-1], R_data.shape[-1]))
        pairwise_comparisons = list(combinations(range(1, R_data.shape[-1] + 1), 2))
        if test_statistics_option==True:    
            test_statistics_list = np.zeros((n_T, Nperm, len(pairwise_comparisons)))
        else:
            test_statistics_list= None
        base_statistics= np.zeros((n_T, 1, len(pairwise_comparisons)))
    elif method == "one_vs_rest":
        pval = np.zeros((n_T, 1, n_q))
        if test_statistics_option==True:
            test_statistics_list = np.zeros((n_T, Nperm, n_q))
        else:
            test_statistics_list= None
        base_statistics= np.zeros((n_T, 1, n_q))

    return pval, base_statistics, test_statistics_list

def expand_variable_permute_beta(base_statistics,test_statistics_list,idx_array, method):
    """
    Expand the base statistics and test statistics for permutation testing.

    Parameters:
    -----------
    base_statistics (numpy.ndarray): 
        The base statistics array.
    test_statistics_list (numpy.ndarray): 
        The list of test statistics arrays.
    idx_array (numpy.ndarray):
        The array containing indices.

    method (str):
        The method used for expansion. Options: "multivariate", other.

    Returns:
    --------
    base_statistics (numpy.ndarray): 
        The expanded base statistics array.

    test_statistics_list (numpy.ndarray):
        The expanded list of test statistics arrays.
    """
    num_sessions =len(np.unique(idx_array))
    if method == "multivariate":
        # Add a new axis with at the second position
        base_statistics = np.tile(base_statistics, (1, num_sessions, 1))  # Expand the second dimension 
        # Add a new axis with size 5 at the second position
        test_statistics_list = np.expand_dims(test_statistics_list, axis=1)  
        test_statistics_list = np.tile(test_statistics_list, (1, num_sessions, 1, 1))  
    else:
        # Add a new axis with size 5 at the second position
        base_statistics = np.expand_dims(base_statistics, axis=1)  
        base_statistics = np.tile(base_statistics, (1, num_sessions, 1, 1))  
        # Add a new axis with size 5 at the second position
        test_statistics_list = np.expand_dims(test_statistics_list, axis=1)  
        test_statistics_list = np.tile(test_statistics_list, (1, num_sessions, 1, 1)) 
    return base_statistics,test_statistics_list

def deconfound_values(D_data, R_data, confounds=None):
    """
    Deconfound the variables R_data and D_data for permutation testing.

    Parameters:
    --------------
    D_data  (numpy.ndarray): 
        The input data array.
    R_data (numpy.ndarray or None): 
        The second input data array (default: None).
        If None, assumes we are working across visits, and R_data represents the Viterbi path of a sequence.
    confounds (numpy.ndarray or None): 
        The confounds array (default: None).

    Returns:
    ----------  
    D_data (numpy.ndarray): 
        Deconfounded D_data  array.
    R_data (numpy.ndarray): 
        Deconfounded R_data array (returns None if R_data is None).
        If R_data is None, assumes we are working across visits
    """
    
    
    # Calculate the centered D-matrix based on confounds (if provided)
    if confounds is not None:
         # Centering confounds
        confounds = confounds - np.nanmean(confounds, axis=0)
        # Centering D_data
        D_data = D_data - np.nanmean(D_data, axis=0)
        # Regressing out confounds from R_data
        D_t = D_data - confounds @ np.linalg.pinv(confounds) @ D_data
        # Check if D_data is provided
        if R_data is not None:
            # Create an NaN-matrix
            R_t= np.empty((R_data.shape))
            R_t[:] = np.nan
            # Regressing out confounds from D_data
            R_data = R_data - np.nanmean(R_data, axis=0)
            q = R_data.shape[-1]

            # Detect if there are ny nan values
            if np.isnan(R_data).any():
                for i in range(q):
                    R_column = np.expand_dims(R_data[:, i],axis=1)
                    valid_indices = np.all(~np.isnan(R_column), axis=1)
                    true_indices = np.where(valid_indices)[0]
                    # detect nan values per column
                    # Set the elements at true_indices to one
                    R_t[true_indices,i] = np.squeeze(R_column[valid_indices] - confounds[valid_indices] @ np.linalg.pinv(confounds[valid_indices]) @ R_column[valid_indices])
            else:
                # Perform matrix operation if there are no nan values
                R_t = R_data - confounds @ np.linalg.pinv(confounds) @ R_data
        else:
            R_t = None # Centering D_data
            R_data = R_data - np.nanmean(R_data, axis=0)
              
    else:
        # Centering D_data and R_data
        D_t = D_data - np.nanmean(D_data, axis=0)
        R_t = None if R_data is None else R_data - np.nanmean(R_data, axis=0)
    
    return D_t, R_t

def initialize_permutation_matrices(method, Nperm, n_p, n_q, D_data, test_combination=False, permute_beta=False, category_columns=None):
    """
    Initializes the permutation matrices and prepare the regularized pseudo-inverse of D_data.

    Parameters:
    --------------
    method (str): 
        The method to use for permutation testing.
    Nperm (int): 
        The number of permutations.
    n_p (int): 
        The number of features.
    n_q (int): 
        The number of predictions.
    D_data (numpy.ndarray): 
        The independent variable.
    test_combination (str), default=False:       
        Specifies the combination method.
        Valid options: "True", "across_columns", "across_rows".
    permute_beta (bool, optional), default=False: 
        A flag indicating whether to permute beta coefficients.


    Returns:
    ----------  
    test_statistics (numpy.ndarray): 
        The permutation array.
    pval_perms (numpy.ndarray): 
        The p-value permutation array.
    reg_pinv (numpy.ndarray or None): 
        The regularized pseudo-inverse of D_data.
    """
    # Define regularized pseudo-inverse
    reg_pinv = None
    # Initialize the permutation matrices based on the selected method
    if method in {"univariate"}:
        if test_combination in [True, "across_columns", "across_rows"]: 
            test_statistics_shape = (Nperm, 1) if test_combination == True else (Nperm, n_q) if test_combination == "across_columns" else (Nperm, n_p)
            test_statistics = np.zeros(test_statistics_shape)
        else:
            # Initialize test statistics output matrix based on the selected method
            test_statistics = np.zeros((Nperm, n_p, n_q))
        if permute_beta or category_columns['f_reg_cols']!=[]:
            # Set the regularization parameter
            regularization = 0.001
            # Create a regularization matrix (identity matrix scaled by the regularization parameter)
            regularization_matrix = regularization * np.eye(D_data.shape[1])  # Regularization term for Ridge regression
            # Compute the regularized pseudo-inverse of D_data
            reg_pinv = np.linalg.inv(D_data.T @ D_data + regularization_matrix) @ D_data.T 
    elif method =="cca":
        # Initialize test statistics output matrix based on the selected method
        test_statistics = np.zeros((Nperm, 1))
    else:
        if test_combination in [True, "across_columns", "across_rows"]:
            test_statistics = np.zeros((Nperm, 1))
        else:
            # Regression got a N by q matrix 
            test_statistics = np.zeros((Nperm, n_q))

        # Define regularization parameter
        regularization_parameter = 0.001
        # Create a regularization matrix (identity matrix scaled by the regularization parameter)
        regularization_matrix = regularization_parameter * np.eye(D_data.shape[1]) 
        # Compute the regularized pseudo-inverse
        reg_pinv = np.linalg.inv(D_data.T @ D_data + regularization_matrix) @ D_data.T  
        
    return test_statistics, np.array(reg_pinv)

def permutation_matrix_across_subjects(Nperm, D_t):
    """
    Generates a normal permutation matrix with the assumption that each index is independent across subjects. 

    Parameters:
    --------------
    Nperm (int): 
        The number of permutations.
    D_t (numpy.ndarray): 
        D-matrix at timepoint 't'
        
    Returns:
    ----------  
    permutation_matrix (numpy.ndarray): 
        Permutation matrix of subjects it got a shape (n_ST, Nperm)
    """
    permutation_matrix = np.zeros((D_t.shape[0],Nperm), dtype=int)
    for perm in range(Nperm):
        if perm == 0:
            permutation_matrix[:,perm] = np.arange(D_t.shape[0])
        else:
            permutation_matrix[:,perm] = np.random.permutation(D_t.shape[0])
    return permutation_matrix

def get_pval(test_statistics, Nperm, method, t, pval, FWER_correction=False):
    """
    Computes p-values from the test statistics.
    # Ref: https://github.com/OHBA-analysis/HMM-MAR/blob/master/utils/testing/permtest_aux.m

    Parameters:
    --------------
    test_statistics (numpy.ndarray): 
        The permutation array.
    pval_perms (numpy.ndarray): 
        The p-value permutation array.
    Nperm (int): 
        The number of permutations.
    method (str): 
        The method used for permutation testing.
    t (int): 
        The timepoint index.
    pval (numpy.ndarray): 
        The p-value array.


    Returns:
    ----------  
    pval (numpy.ndarray): 
        Updated updated p-value .
    """
    if method == "multivariate" or method == "one_vs_rest":
        if FWER_correction:
            # Perform family wise permutation correction
            # Define the number of columns and rows
            nCols = test_statistics[0,:].shape[-1]
            nRows = len(test_statistics)
            # Get the maximum explained variance for each column
            max_test_statistics =np.tile(np.max(test_statistics[1:,:], axis=1), (1, nCols)).reshape(nCols, nRows-1).T
            # Count every time there is a higher estimated R2 (better fit)
            pval[t, :] = np.sum(max_test_statistics>= test_statistics[0,:], axis=0) / (Nperm + 1)
        else:
            # Count every time there is a higher estimated R2 (better fit)
            pval[t, :] = np.nansum(test_statistics >= test_statistics[0,:], axis=0) / (Nperm + 1)
        
    elif method == "univariate" or method =="cca":
        if FWER_correction:
            # Perform family wise permutation correction
            # Define the number of columns and rows
            nCols = test_statistics.shape[-1]
            nRows = test_statistics.shape[1]

            nPerm = test_statistics.shape[0]
            # Calculate the MaxT statistics for each permutation (excluding the observed)
            maxT_statistics = np.max(np.abs(test_statistics[1:, :, :]), axis=(1, 2))

            # Calculate the observed test statistics
            observed_test_stats = np.abs(test_statistics[0])

            # Use broadcasting to compare and count
            observed_expanded = observed_test_stats[np.newaxis, :, :]  
            maxT_expanded = maxT_statistics[:, np.newaxis, np.newaxis]  

            # Count how many times maxT_statistics exceed or equal each observed statistic
            pval[t, :] = np.sum(maxT_expanded >= observed_expanded, axis=0)  / (Nperm)

            #pval[t, :] = np.nansum(max_test_statistics>= test_statistics[0,:], axis=0) / (Nperm + 1)
        else:    
            # Count every time there is a higher correlation coefficient
            pval[t, :] = np.nansum(test_statistics >= test_statistics[0,:], axis=0) / (Nperm + 1)
    
    # Convert 0 values to NaN
    # pval[pval == 0] = np.nan
    
    return pval


def permutation_matrix_across_trials_within_session(Nperm, R_t, idx_array, trial_timepoints=None):
    """
    Generates permutation matrix of within-session across-trial data based on given indices.

    Parameters:
    --------------
    Nperm (int): 
        The number of permutations.
    R_t (numpy.ndarray): 
        The preprocessed data array.
    idx_array (numpy.ndarray): 
        The indices array.
    trial_timepoints (int): 
        Number of timepoints for each trial (default: None)

    Returns:
    ----------  
    permutation_matrix (numpy.ndarray): 
        Permutation matrix of subjects it got a shape (n_ST, Nperm)
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
    idx_array (numpy.ndarray): 
        Input array to be permuted.
    
    Returns:
    ----------  
    permuted_array (numpy.ndarray):
        Permuted matrix based on unique values.
    """
    # Get unique values and their counts
    unique_values, value_counts = np.unique(idx_array, return_counts=True)

    # Permute the unique values
    permuted_unique_values = np.random.permutation(unique_values)

    # Repeat each unique value according to its original count
    permuted_array = np.repeat(permuted_unique_values, value_counts)

    return permuted_array


def permutation_matrix_within_subject_across_sessions(Nperm, R_t, idx_array):
    """
    Generates permutation matrix of within-session across-session data based on given indices.

    Parameters:
    --------------
    Nperm (int): 
        The number of permutations.
    R_t (numpy.ndarray): 
        The preprocessed data array.
    idx_array (numpy.ndarray): 
        The indices array.


    Returns:
    ----------  
    permutation_matrix (numpy.ndarray): 
        The within-session continuos indices array.
    """
    permutation_matrix = np.zeros((R_t.shape[0],Nperm), dtype=int)
    for perm in range(Nperm):
        if perm == 0:
            permutation_matrix[:,perm] = np.arange(R_t.shape[0])
        else:
            idx_array_perm = permute_subject_trial_idx(idx_array)
            unique_indices = np.unique(idx_array_perm)
            positions_permute = [np.where(np.array(idx_array_perm) == i)[0] for i in unique_indices]
            permutation_matrix[:,perm] = np.concatenate(positions_permute,axis=0)
    return permutation_matrix

def permutation_matrix_within_and_between_groups(Nperm, R_t, idx_array):
    """
    Generates a permutation matrix with permutations within and between groups.
    
    Parameters:
    --------------
    Nperm (int): 
        The number of permutations.
    R_t (numpy.ndarray): 
        The R_t array used for generating the within-group permutation matrix.
    idx_array (numpy.ndarray): 
        The indices array that groups the R_t (e.g., session or subject identifiers).
    
    Returns:
    ----------
    permutation_matrix (numpy.ndarray): 
        The matrix with both within- and between-group permutations.
    """
    
    # Generate the within-group permutation matrix
    permutation_matrix_within_group = permutation_matrix_across_trials_within_session(
        Nperm, R_t, idx_array)
    
    # Initialize the permutation matrix for within and between groups
    permutation_within_and_between_group = np.zeros_like(permutation_matrix_within_group)
    
    for perm in range(Nperm):
        if perm == 0:
            # The first column is just the identity permutation (no change)
            permutation_within_and_between_group[:, perm] = np.arange(R_t.shape[0])
        else:
            # Permute the idx_array to shuffle the groupings
            idx_array_perm = permute_subject_trial_idx(idx_array)
            unique_indices = np.unique(idx_array_perm)
            
            # Find the positions of each unique group after permutation
            positions_permute = [np.where(np.array(idx_array_perm) == i)[0] for i in unique_indices]
            perm_array = np.concatenate(positions_permute, axis=0)
            
            # Apply the within-group permutation to the newly permuted positions
            permutation_within_and_between_group[:, perm] = permutation_matrix_within_group[:, perm][perm_array]
    
    return permutation_within_and_between_group

def generate_vpath_1D(vpath):
    """
    Convert a 2D array representing a matrix with one non-zero element in each row
    into a 1D array where each element is the column index of the non-zero element.

    Parameters:
    ------------
    vpath(numpy.ndarray):       
        A 2D array where each row has only one non-zero element. 
        Or a 1D array where each row represents a sate number

    Returns:
    ------------
    vpath_array(numpy.ndarray): 
        A 1D array containing the column indices of the non-zero elements.
        If the input array is already 1D, it returns a copy of the input array.

    """
    if np.ndim(vpath) == 2:
        vpath_array = np.nonzero(vpath)[1] + 1
    else:
        if np.min(vpath)==0:
            # Then it is already a vector
            vpath_array = vpath.copy()+1
        else:
            # Then it is already a vector
            vpath_array = vpath.copy()

    return vpath_array



def surrogate_state_time(perm, viterbi_path,n_states):
    """
    Generates surrogate state-time matrix based on a given Viterbi path.

    Parameters:
    --------------
    perm (int): 
        The permutation number.
    viterbi_path (numpy.ndarray): 
        1D array or 2D matrix containing the Viterbi path.
    n_states (int): 
        The number of states

    Returns:
    ----------  
    viterbi_path_surrogate (numpy.ndarray): 
        A 1D array representing the surrogate Viterbi path
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


def surrogate_state_time_matrix(Nperm, vpath_data, n_states):
    vpath_array=generate_vpath_1D(vpath_data)
    vpath_surrogates = np.zeros((len(vpath_array),Nperm))
    for perm in tqdm(range(Nperm)):
        while True:
            vpath_surrogates[:,perm] = surrogate_state_time(perm, vpath_array, n_states)
            if len(np.unique(vpath_surrogates[:,perm])) == n_states:
                break  # Exit the loop if the condition is satisfied
    return vpath_surrogates


def viterbi_path_to_stc(viterbi_path, n_states):
    """
    Convert Viterbi path to state-time matrix.

    Parameters:
    --------------
    viterbi_path (numpy.ndarray): 
        1D array or 2D matrix containing the Viterbi path.
    n_states (int): 
        Number of states in the hidden Markov model.

    Returns:
    ----------  
    stc (numpy.ndarray): 
        State-time matrix where each row represents a time point and each column represents a state.
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
    viterbi_path (numpy.ndarray):   
        1D array or 2D matrix containing the Viterbi path.
    n_states (int):                
        Number of states in the hidden Markov model.

    Returns:
    ----------  
    viterbi_path_surrogate (numpy.ndarray): 
        Surrogate Viterbi path as a 1D array representing the state indices.
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

def calculate_baseline_difference(vpath_array, R_data, state, pairwise_statistic, category_columns):
    """
    Calculate the difference between the specified statistics of a state and all other states combined.

    Parameters:
    --------------
    vpath_data (numpy.ndarray): 
        The Viterbi path as of integer values that range from 1 to n_states.
    R_data (numpy.ndarray):     
        The dependent-variable associated with each state.
    state(numpy.ndarray):       
        The state for which the difference is calculated from.
    pairwise_statistic (str)             
        The chosen statistic to be calculated. Valid options are "mean" or "median".

    Returns:
    ----------  
    difference (float)            
        The calculated difference between the specified state and all other states combined.
    """
    if pairwise_statistic == 'median':
        state_R_data = np.nanmedian(R_data[vpath_array == state])
        other_R_data = np.nanmedian(R_data[vpath_array != state])
    elif pairwise_statistic == 'mean':
        state_R_data = np.nanmean(R_data[vpath_array == state])
        other_R_data = np.nanmean(R_data[vpath_array != state])
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
    vpath_data (numpy.ndarray): 
        The Viterbi path as of integer values that range from 1 to n_states.
    R_data (numpy.ndarray):     
        The dependent-variable associated with each state.
    state_1 (int):              
        First state for comparison.
    state_2 (int):              
        Second state for comparison.
    statistic (str):            
        The chosen statistic to be calculated. Valid options are "mean" or "median".

    Returns:
    ----------  
    difference (float):           
        The calculated difference between the two states.
    """
    if stat == 'mean':
        state_1_R_data = np.nanmean(R_data[vpath_array == state_1])
        state_2_R_data = np.nanmean(R_data[vpath_array == state_2])
    elif stat == 'median':
        state_1_R_data = np.nanmedian(R_data[vpath_array == state_1])
        state_2_R_data = np.nanmedian(R_data[vpath_array == state_2])
    else:
        raise ValueError("Invalid stat value")
    # Detect any difference
    difference = state_1_R_data - state_2_R_data
    return difference

def test_statistics_calculations(Din, Rin, perm, test_statistics, reg_pinv, method, category_columns=[], test_combination=False, idx_data=None, permute_beta=False, beta = None, test_indices=None):
    """
    Calculates the test_statistics array and pval_perms array based on the given data and method.

    Parameters:
    --------------
    Din (numpy.ndarray): 
        The data array.
    Rin (numpy.ndarray): 
        The dependent variable.
    perm (int): 
        The permutation index.
    pval_perms (numpy.ndarray): 
        The p-value permutation array.
    test_statistics (numpy.ndarray): 
        The permutation array.
    reg_pinv (numpy.ndarray or None):  
        The regularized pseudo-inverse of D_data
    method (str): 
        The method used for permutation testing.
    category_columns (dict):
        A dictionary marking the columns where t-test ("t_test_cols") and F-test ("f_anova_cols") have been applied.
    test_combination (str), default=False:       
        Specifies the combination method.
        Valid options: "True", "across_columns", "across_rows".
    idx_data (numpy.ndarray), default=None: 
        An array containing the indices for each session. The array can be either 1D or 2D:
        For a 1D array, a sequence of integers where each integer labels the session number. For example, [1, 1, 1, 1, 2, 2, 2, ..., N, N, N, N, N, N, N, N].
        For a 2D array, each row represents the start and end indices for the trials in a given session, with the format [[start1, end1], [start2, end2], ..., [startN, endN]].   
    permute_beta (bool, optional), default=False: 
        A flag indicating whether to permute beta coefficients.
    beta (numpy.ndarray), default=None:
        beta coefficient for each session.
        It has a shape (num_session, p, q), where the first dimension 
        represents the session, the second dimension represents the featires, 
        and the third dimension represent dependent variables. 
    test_indices (numpy.ndarray), default=None:
        Indices for data points that belongs to the test-set for each session.

    Returns:
    ----------  
    test_statistics (numpy.ndarray): 
        Updated test_statistics array.
    base_statistics (numpy.ndarray): 
        Updated pval_perms array.
    pval_matrix (numpy.ndarray): 
        P-values derived from t and f statistics using General linear models.
    """
    pval_matrix= None
    if method == 'multivariate':

        if category_columns["t_test_cols"]==[] and category_columns["f_anova_cols"]==[] and category_columns["f_reg_cols"]==[]:
            # We wont have a p-value matrix since we are not doing GLM inference here

            nan_values = np.sum(np.isnan(Rin))>0
            if nan_values:
                # NaN values are detected
                if test_combination in [True, "across_columns"]:
                    # Calculate F-statitics with no NaN values.
                    F_statistic, p_value =calculate_nan_regression_f_test(Din, Rin, reg_pinv, idx_data, permute_beta, perm, nan_values)
                    # Get the base statistics and store p-values as z-scores to the test statistic
                    base_statistics = calculate_combined_z_scores(p_value, test_combination)
                    test_statistics[perm] =abs(base_statistics) 
                else:
                    # Calculate the explained variance if R got NaN values.
                    base_statistics =calculate_nan_regression(Din, Rin, reg_pinv, idx_data, permute_beta, perm, beta, test_indices)
                    test_statistics[perm,:] =base_statistics           
            else:
                # No- NaN values are detected
                if idx_data is not None and permute_beta==True:
                    # Calculate predicted values with no NaN values
                    if test_indices is not None:
                        Rin =Rin[test_indices[0],:]
                        Din =Din[test_indices[0],:]
                        # calculate beta coefficients
                        R_pred =calculate_ols_predictions(Rin, Din, idx_data, beta, perm, permute_beta)
                    else:    
                        # calculate beta coefficients
                        R_pred =calculate_ols_predictions(Rin, Din, idx_data, beta, perm, permute_beta)
                else:
                    # Fit the  model 
                    beta_0 = reg_pinv @ Rin  # Calculate regression_coefficients (beta)
                    # Calculate the predicted values
                    R_pred = Din @ beta_0   
                # Calculate the residual sum of squares (rss)
                rss = np.sum((Rin-R_pred)**2, axis=0)
                # Calculate the total sum of squares (tss)
                tss = np.sum((Rin - np.nanmean(Rin, axis=0))**2, axis=0)
            
                if test_combination in [True, "across_columns"]:
                    # Calculate the parametric p-values using F-statistics
                    # Calculate the explained sum of squares (ESS)
                    ess = tss - rss
                    # Calculate the degrees of freedom for the model and residuals
                    df_model = Din.shape[1]  # Number of predictors including intercept
                    df_resid = Din.shape[0] - df_model
                    # Calculate the mean squared error (MSE) for the model and residuals
                    MSE_model = ess / df_model
                    MSE_resid = rss / df_resid
                    # Calculate the F-statistic
                    F_statistic = (MSE_model / MSE_resid)
                    pval_matrix = 1 - f.cdf(F_statistic, df_model, df_resid)
                    # Get the base statistics and store p-values as z-scores to the test statistic
                    base_statistics = calculate_combined_z_scores(pval_matrix, test_combination)
                    test_statistics[perm] =abs(base_statistics) 
                else:
                    # Calculate R^2
                    base_statistics = 1 - (rss / tss) #r_squared
                    # Store the R^2 values in the test_statistics array
                    test_statistics[perm] = base_statistics
                    
        else:
            # Now we have to do t- or f-statistics
            # If we are doing test_combinations, we need to calculate f-statistics on every column
            if test_combination in [True, "across_columns"]:
                nan_values = np.sum(np.isnan(Rin))>0
                # Calculate the explained variance if R got NaN values.
                F_statistic, pval_matrix =calculate_nan_regression_f_test(Din, Rin, reg_pinv, idx_data, permute_beta, perm, nan_values, beta)
                # Get the base statistics and store p-values as z-scores to the test statistic
                base_statistics = calculate_combined_z_scores(pval_matrix, test_combination)
                test_statistics[perm] =abs(base_statistics) 
            
            else:
            # If we are not perfomring test_combination, we need to perform a columnwise operation.  
            # We perform f-test if category_columns has flagged categorical columns otherwise it will be R^2 
                # Initialize variables  which
                #base_statistics =np.zeros_like(test_statistics[0,:]) if perm ==0 else None
                base_statistics =np.zeros_like(test_statistics[0,:])
                pval_matrix =np.zeros_like(test_statistics[0,:])
                for col in range(Rin.shape[1]):
                    # Get the R_column
                    R_column = Rin[:, col]
                    # Calculate f-statistics of columns of interest 
                    nan_values = np.sum(np.isnan(Rin[:,col]))>0 
                    if category_columns["f_anova_cols"] and col in category_columns["f_anova_cols"]:
                        # Nan values
                        if permute_beta:
                            # Calculate base statistics per column
                            if test_indices is not None:
                                base_statistics[col], pval_matrix[col] =calculate_nan_anova_f_test(Din[test_indices[col],:], Rin[test_indices[col],col], reg_pinv[:,test_indices[col]], idx_data, permute_beta, perm, nan_values, beta=np.expand_dims(beta[:,:,col],axis=2))
                            else:    
                                base_statistics[col], pval_matrix[col] =calculate_nan_anova_f_test(Din, R_column, reg_pinv, idx_data, permute_beta, perm, nan_values, beta=np.expand_dims(beta[:,:,col],axis=2))
                        else:
                            # Then we need to calculate beta
                            base_statistics[col], pval_matrix[col]=calculate_nan_anova_f_test(Din, R_column, reg_pinv, idx_data, permute_beta, perm, nan_values)
                        test_statistics[perm,col] = base_statistics[col]   
                          
                    elif category_columns["f_reg_cols"] and col in category_columns["f_reg_cols"]:
                        # Nan values
                        if permute_beta:
                            # Calculate base statistics per column
                            if test_indices is not None:
                                base_statistics[col], pval_matrix[col] =calculate_nan_regression_f_test(Din[test_indices[col],:], Rin[test_indices[col],col], reg_pinv[:,test_indices[col]], idx_data, permute_beta, perm, nan_values, beta=np.expand_dims(beta[:,:,col],axis=2))
                            else:    
                                base_statistics[col], pval_matrix[col] =calculate_nan_regression_f_test(Din, R_column, reg_pinv, idx_data, permute_beta, perm, nan_values, beta=np.expand_dims(beta[:,:,col],axis=2))
                        else:
                            # Then we need to calculate beta
                            base_statistics[col], pval_matrix[col]=calculate_nan_regression_f_test(Din, R_column, reg_pinv, idx_data, permute_beta, perm, nan_values)
                        test_statistics[perm,col] = base_statistics[col]  
                    else:
                        # Check for NaN values
                        if nan_values:
                            if permute_beta:
                                # This is done for across session testing
                                # Calculate the explained variance if R got NaN values.
                                base_statistics[col] =calculate_nan_regression(Din, Rin[:, col], reg_pinv, idx_data, permute_beta, perm, np.expand_dims(beta[:,:,col],axis=2), test_indices[col])  
                            else:
                                base_statistics[col] =calculate_nan_regression(Din, Rin[:, col], reg_pinv, idx_data, permute_beta, perm)        
                        else:
                            if beta is None:
                                # Fit the original model 
                                beta_0 = reg_pinv @ Rin[:, col]  # Calculate regression_coefficients (beta)
                                # # # # Include intercept
                                # # # Din = np.hstack((np.ones((Din.shape[0], 1)), Din))
                                # Calculate the predicted values
                                R_pred = Din @ beta_0
                            else:
                                # Update R_column
                                R_column = Rin[test_indices[col], col]
                                # Calculate predicted values using the test_set
                                R_pred = calculate_ols_predictions(R_column, Din[test_indices[col],:], idx_data, beta[:,:,col], perm, permute_beta)
                            # Calculate the residual sum of squares (rss)
                            rss = np.sum((R_column-R_pred)**2, axis=0)
                            # Calculate the total sum of squares (tss)
                            tss = np.sum((R_column - np.mean(R_column, axis=0))**2, axis=0)
                            # Calculate R^2
                            base_statistics[col] = 1 - (rss / tss) #r_squared
                        # Store the R^2 values in the test_statistics array
                        test_statistics[perm,col] = base_statistics[col]        
    # Calculate for univariate tests              
    elif method == "univariate":
        if category_columns["t_test_cols"]==[] and category_columns["f_anova_cols"]==[]and category_columns["f_reg_cols"]==[]:
            # We wont have a p-value matrix since we are not doing GLM inference here
            pval_matrix = None
            # Only calcuating the correlation matrix, since there is no need for t- or f-test
            if np.sum(np.isnan(Din))>0 or np.sum(np.isnan(Rin))>0:
                # Calculate the correlation matrix while handling NaN values 
                # column by column without removing entire rows.
                if test_combination in [True, "across_columns", "across_rows"] and permute_beta==False: 
                    if permute_beta:
                        _ , p_value =calculate_r2_univariate(Din, Rin, idx_data, beta, perm, reg_pinv, permute_beta, test_combination=test_combination, test_indices=test_indices)
                        base_statistics = calculate_combined_z_scores(p_value, test_combination)
                        test_statistics[perm] =abs(base_statistics) 
                    else:
                        # Return parametric p-values
                        _ ,base_statistics =calculate_nan_correlation_matrix(Din, Rin, test_combination)
                        test_statistics[perm, :] = base_statistics # Notice that shape of test_statistics are different
                elif permute_beta:
                    base_statistics, _ =calculate_r2_univariate(Din, Rin, idx_data, beta, perm, reg_pinv, permute_beta, test_indices=test_indices)
                    test_statistics[perm, :, :] = base_statistics
                else:
                    # Return correlation coefficients instead of p-values
                    base_statistics,_ =calculate_nan_correlation_matrix(Din, Rin)
                    test_statistics[perm, :, :] = np.abs(base_statistics)
            else:
                if test_combination in [True, "across_columns", "across_rows"]: 
                    if permute_beta:
                        # Calculate geometric mean of p-values
                        _, p_value =calculate_r2_univariate(Din, Rin, idx_data, beta, perm, reg_pinv, permute_beta, test_combination=test_combination, test_indices=test_indices)
                        base_statistics = calculate_combined_z_scores(p_value, test_combination)
                        test_statistics[perm] =abs(base_statistics) 
                    else:    
                        # Return parametric p-values
                        pval_matrix = np.zeros((Din.shape[1],Rin.shape[1]))
                        for i in range(Din.shape[1]):
                            for j in range(Rin.shape[1]):
                                _, pval_matrix[i, j] = pearsonr(Din[:, i], Rin[:, j])
                        base_statistics =calculate_combined_z_scores(pval_matrix, test_combination) 
                        test_statistics[perm,:]=abs(base_statistics)      
                elif permute_beta:
                    base_statistics, _ =calculate_r2_univariate(Din, Rin, idx_data, beta, perm, reg_pinv, permute_beta, test_indices=test_indices)
                    test_statistics[perm, :, :] = np.abs(base_statistics)
                else:
                    # Calculate correlation coeffcients without NaN values
                    corr_coef = np.corrcoef(Din, Rin, rowvar=False)
                    corr_matrix = corr_coef[:Din.shape[1], Din.shape[1]:]
                    base_statistics = corr_matrix
                    test_statistics[perm, :, :] = np.abs(base_statistics)
        else: 
            # Calculate t-, f- and correlation statistics per column
            base_statistics =np.zeros_like(test_statistics[0,:]) if perm ==0 else None
            pval_matrix = np.zeros((Din.shape[-1],Rin.shape[-1]))
            base_statistics_com = np.zeros((Din.shape[-1],Rin.shape[-1]))
            for col in range(Rin.shape[1]):
                if category_columns["t_test_cols"] and col in category_columns["t_test_cols"]:
                    nan_values = True if np.sum(np.isnan(Din))>0 or np.sum(np.isnan(Rin))>0 else False
                    # Perform  t-statistics per column if nan_values=True 
                    t_test, pval = calculate_nan_t_test(Din, Rin[:, col], nan_values=nan_values)    
                    # Store values
                    pval_matrix[:, col] = pval
                    test_statistics[perm, :, col] = np.abs(t_test)
                    # save t-test to base_statistics
                    if perm==0:
                        if test_combination in [True, "across_columns", "across_rows"]:
                            # Convert pval to z-score
                            base_statistics_com[:,col] = norm.ppf(1 - np.array(np.squeeze(pval)))
                            # base_statistics_com[:,col] = np.squeeze(pval)
                        else:
                            base_statistics[:,col]= t_test 
                elif category_columns["f_anova_cols"] and col in category_columns["f_anova_cols"]:
                    nan_values = True if np.sum(np.isnan(Din))>0 or np.sum(np.isnan(Rin))>0 else False
                    # Perform f-statistics
                    f_statistic, pval =calculate_anova_f_test(Din, Rin[:, col], nan_values=nan_values)
                    # Store values
                    pval_matrix[:, col] = pval
                    test_statistics[perm, :, col] = np.abs(f_statistic)
                    # Insert base statistics
                    if perm==0:
                        if test_combination in [True, "across_columns", "across_rows"]:
                            # Convert pval to z-score
                            base_statistics_com[:,col] = norm.ppf(1 - np.array(np.squeeze(pval)))
                            # base_statistics_com[:,col] = np.squeeze(pval)
                        else:
                            base_statistics[:,col]= f_statistic       

                elif category_columns["f_reg_cols"] and col in category_columns["f_reg_cols"]:
                    nan_values = True if np.sum(np.isnan(Din))>0 or np.sum(np.isnan(Rin))>0 else False
                    # Perform f-statistics
                    f_statistic, pval =calculate_reg_f_test(Din, Rin[:, col], idx_data, beta, perm, reg_pinv, permute_beta, test_indices=test_indices)
                    # Store values 
                    pval_matrix[:, col] = pval
                    test_statistics[perm, :, col] = np.abs(f_statistic)
                    # Insert base statistics
                    if perm==0:
                        if test_combination in [True, "across_columns", "across_rows"]:
                            # Convert pval to z-score
                            base_statistics_com[:,col] = norm.ppf(1 - np.array(np.squeeze(pval)))
                            #base_statistics_com[:,col] = np.squeeze(pval)
                        else:
                            base_statistics[:,col]= f_statistic    
                                                          
                else:
                    # Perform corrrelation analysis
                    if np.sum(np.isnan(Din))>0 or np.sum(np.isnan(Rin))>0:
                        # Calculate the correlation matrix while handling NaN values 
                        # column by column without removing entire rows.
                        corr_array, pval =calculate_nan_correlation_matrix(Din, Rin[:, col], test_combination)
                    else:
                        # calculate correlation and pval between Din and Rin if there are no NaN values
                        #corr_array, pval= pearsonr(Din, np.expand_dims(Rin[:, col],axis=1))
                        if test_combination in [True, "across_columns", "across_rows"]:
                            
                            pval = np.zeros((Din.shape[-1],1)) 
                            # Have to make a loop between each columns to get the p-values
                            for i in range(Din.shape[1]):
                                _, pval[i] = pearsonr(Din[:, i], Rin[:, col])
                        else:
                            #Calculate correlation coefficient matrix - Faster calculation
                            corr_coef = np.corrcoef(Din, Rin[:, col], rowvar=False)
                            # get the correlation matrix
                            corr_array = np.squeeze(corr_coef[:Din.shape[1], Din.shape[1]:])
                            # Update test_statistics
                    if test_combination in [True, "across_columns", "across_rows"]:
                        pval_matrix[:, col] = np.squeeze(pval)
                    else:
                        test_statistics[perm, :, col] = np.abs(np.squeeze(corr_array))       
                    # Insert base statistics
                    if perm==0:
                        if test_combination in [True, "across_columns", "across_rows"]:
                            # Convert pval to z-score
                            base_statistics_com[:,col] = norm.ppf(1 - np.array(np.squeeze(pval)))
                            # base_statistics_com[:,col] = np.squeeze(pval)
                        else:
                            base_statistics[:,col]= np.squeeze(corr_array)
                            
                          
            if test_combination!=False:
                # remember that the base_statistics_com only contains p-values
                #pval_matrix = base_statistics_com.copy()
                base_statistics = calculate_combined_z_scores(base_statistics_com, test_combination) if perm==0 else base_statistics_com
                test_statistics[perm,:] =abs(calculate_combined_z_scores(pval_matrix, test_combination)) 


    elif method =="cca":
        # Create CCA object with 1 component
        cca = CCA(n_components=1)
        # Fit the CCA model to your data
        cca.fit(Din, Rin)
        # Transform the input data using the learned CCA model
        X_c, Y_c = cca.transform(Din, Rin)
        # Calcualte the correlation coefficients between X_c and Y_c
        base_statistics = np.corrcoef(X_c, Y_c, rowvar=False)[0, 1]
        # Update test_statistics
        test_statistics[perm] = np.abs(base_statistics)
        pval_matrix = None
    # Check if perm is 0 before returning the result
    return test_statistics, base_statistics, pval_matrix


def calculate_r2_univariate(Din, Rin, idx_data, beta, perm, reg_pinv, permute_beta, test_combination=False, test_indices=None):
    
    """
    Calculate the R-squared values for the regression of between Din and Rin column-wise in a univarete procedure.

    Parameters:
    --------------
    Din (numpy.ndarray): 
        Input data matrix for the independent variables.
    Rin (numpy.ndarray): 
        Input data matrix for the dependent variables.
    idx_data (numpy.ndarray): 
        An array containing the indices for each session. The array can be either 1D or 2D:
        For a 1D array, a sequence of integers where each integer labels the session number. For example, [1, 1, 1, 1, 2, 2, 2, ..., N, N, N, N, N, N, N, N].
        For a 2D array, each row represents the start and end indices for the trials in a given session, with the format [[start1, end1], [start2, end2], ..., [startN, endN]].  
    beta (numpy.ndarray):
        beta coefficient for each session.
        It has a shape (num_session, p, q), where the first dimension 
        represents the session, the second dimension represents the featires, 
        and the third dimension represent dependent variables. 
    perm (int): 
        The permutation index.    
    reg_pinv (numpy.ndarray): 
        The regularized pseudo-inverse of D_data
    permute_beta (bool, optional): 
        A flag indicating whether to permute beta coefficients.
    test_combination (str), default=False:       
        Specifies the combination method.
        Valid options: "True", "across_columns", "across_rows".
    test_indices (numpy.ndarray), default=None:
        Indices for data points that belongs to the test-set for each session.
    
    Returns:
    --------
    base_statistics (numpy.ndarray): 
        The expanded base statistics array.
    pval_matrix (numpy.ndarray): 
        calculated p-values estimated from the F-statistic
    """
    base_statistics = np.zeros((Din.shape[-1],Rin.shape[-1]))
    pval_matrix = np.zeros((Din.shape[1], Rin.shape[1]))
    for q_i in range(Rin.shape[-1]):
        # Identify columns with NaN values
        R_column = np.expand_dims(Rin[test_indices[0] if len(test_indices) == 1 else test_indices[q_i], q_i], axis=1)
        valid_indices = np.all(~np.isnan(R_column), axis=1)

        if test_combination !=False:
            # Need to get the p-values
            for p_j in range(Din.shape[-1]):
                D_column = np.expand_dims(Din[test_indices[0 if len(test_indices) == 1 else q_i], p_j], axis=1)
                reg_pinv_column = np.expand_dims(reg_pinv[p_j, test_indices[0 if len(test_indices) == 1 else q_i]], axis=0)
                beta_column, _ = np.squeeze(calculate_ols_beta(reg_pinv_column, R_column, idx_data)[0]) if beta is None else beta[:,p_j,q_i]
                nan_values = np.any(np.isnan(valid_indices)) or np.any(np.isnan(D_column))
                R_pred =calculate_ols_predictions(R_column, D_column, idx_data, beta_column, perm, permute_beta, nan_values,  valid_indices)
                # Calculate the residual sum of squares (rss)
                rss = np.sum((np.expand_dims(Rin[valid_indices,q_i],axis=1)-R_pred)**2, axis=0)
                # Calculate the total sum of squares (tss)
                tss = np.sum((Rin[valid_indices,q_i] - np.nanmean(Rin[valid_indices,q_i], axis=0))**2, axis=0)
                ess = tss - rss
                # Calculate the degrees of freedom for the model and residuals
                df_model = D_column.shape[1]  # Number of predictors including intercept
                df_resid = D_column.shape[0] - df_model
                # Calculate the mean squared error (MSE) for the model and residuals
                MSE_model = ess / df_model
                MSE_resid = rss / df_resid
                # Calculate the F-statistic
                f_statistics = MSE_model / MSE_resid
                # Calculate the p-value for the F-statistic
                pval = 1 - f.cdf(f_statistics, df_model, df_resid)
                # Store the p-value
                pval_matrix[p_j, q_i] = pval 
        else:
            # Explained variance
            for p_j in range(Din.shape[-1]):
                D_column = np.expand_dims(Din[test_indices[0 if len(test_indices) == 1 else q_i], p_j], axis=1)
                reg_pinv_column = np.expand_dims(reg_pinv[p_j, test_indices[0 if len(test_indices) == 1 else q_i]], axis=0)
                beta_column = np.squeeze(calculate_ols_beta(reg_pinv_column, R_column, idx_data)[0]) if beta is None else beta[:,p_j,q_i]
                nan_values = np.any(np.isnan(valid_indices)) or np.any(np.isnan(D_column))
                R_pred =calculate_ols_predictions(R_column, D_column, idx_data, beta_column, perm, permute_beta, nan_values,  valid_indices)
                # Calculate the residual sum of squares (rss)
                rss = np.sum((R_column - R_pred)**2, axis=0)
                # Calculate the total sum of squares (tss)
                tss = np.sum((R_column - np.mean(R_column, axis=0))**2, axis=0)
                base_statistics[p_j, q_i] = 1 - (rss / tss) #r_squared 

    return base_statistics, pval_matrix
        
def calculate_combined_z_scores(p_values, test_combination):
    """
    Calculate test statistics of z-scores converted from p-values based on the specified combination.

    Parameters:
    --------------
    p_values (numpy.ndarray):  
        Matrix of p-values.
    test_combination (str):       
        Specifies the combination method.
        Valid options: "True", "across_columns", "across_rows".
        Default is "True".

    Returns:
    ----------  
    result (numpy.ndarray):       
        Test statistics of z-scores converted from p-values.
    """
    if test_combination == True:
        pval = np.squeeze(np.exp(np.mean(np.log(p_values))))
        z_scores = norm.ppf(1 - np.array(pval))
        test_statistics = z_scores
    elif test_combination == "across_columns" or test_combination == "across_rows":
        axis = 0 if test_combination == "across_columns" else 1
        pval = np.squeeze(np.exp(np.mean(np.log(p_values), axis=axis)))
        z_scores = norm.ppf(1 - np.array(pval))
        test_statistics = z_scores
    else:
        raise ValueError("Invalid value for test_combination parameter")

    return test_statistics

def pval_correction(pval, method='fdr_bh', alpha=0.05, include_nan=True, nan_diagonal=False):
    """
    Adjusts p-values for multiple testing.

    Parameters:
    --------------
    pval (numpy.ndarray): 
        numpy array of p-values.
    method (str, optional): method used for FDR correction, default='fdr_bh.
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
    alpha (float, optional): 
        Significance level (default: 0.05).
    include_nan, default=True: 
        Include NaN values during the correction of p-values if True. Exclude NaN values if False.
    nan_diagonal, default=False: 
        Add NaN values to the diagonal if True.

    Returns:
    ---------- 
    pval_corrected (numpy.ndarray): 
        numpy array of corrected p-values.
    significant (numpy.ndarray): 
        numpy array of boolean values indicating significant p-values.
    """
    # Input validation
    if nan_diagonal and pval.ndim != 2:
        raise ValueError("If nan_diagonal is True, input pval must be a 2D array.")
    
    if include_nan:
        # Flatten the matrix and keep track of NaN positions
        flat_pval = pval.flatten()
        nan_positions = np.isnan(flat_pval)

        # Replace NaN values with 1 (or any value representing non-significance) for correction
        flat_pval[nan_positions] = 1

        # Perform multiple testing correction
        significant, pval_corrected, _, _ = smt.multipletests(flat_pval, alpha=alpha, method=method, returnsorted=False)

        # Replace the NaN values back in the corrected p-values
        pval_corrected[nan_positions] = np.nan
        significant[nan_positions] = np.nan

        # Reshape the corrected p-value and significant arrays back to the original shape
        pval_corrected = pval_corrected.reshape(pval.shape)
        significant = significant.reshape(pval.shape)

    else:
        # Flatten the matrix and remove NaN values for correction
        flat_pval = pval.flatten()
        non_nan_positions = ~np.isnan(flat_pval)
        flat_pval_no_nan = flat_pval[non_nan_positions]

        # Perform multiple testing correction on non-NaN values
        significant_no_nan, pval_corrected_no_nan, _, _ = smt.multipletests(flat_pval_no_nan, alpha=alpha, method=method, returnsorted=False)

        # Create an array filled with NaN values
        pval_corrected = np.full_like(flat_pval, np.nan)
        significant = np.full_like(flat_pval, np.nan)

        # Assign the corrected values to their respective positions in the original shape
        pval_corrected[non_nan_positions] = pval_corrected_no_nan
        significant[non_nan_positions] = significant_no_nan

        # Reshape the corrected p-value and significant arrays back to the original shape
        pval_corrected = pval_corrected.reshape(pval.shape)
        significant = significant.reshape(pval.shape)

    if nan_diagonal:
        pval_corrected =np.fill_diagonal(pval_corrected, np.nan)
        significant =np.fill_diagonal(significant, np.nan)

    # Return the corrected p-values and boolean values indicating significant p-values
    return pval_corrected, significant

def pval_cluster_based_correction(test_statistics, pval, alpha=0.05, individual_feature=False):
    """
    Perform cluster-based correction on test statistics using the output from permutation testing.
    The function corrects p-values by using the test statistics and p-values obtained from permutation testing.
    It converts the test statistics into z-scores, thresholds them to identify clusters, and uses the cluster sizes to adjust the p-values.
        
    Parameters:
    ------------
    test_statistics (numpy.ndarray): 
        2D or 3D array of test statistics.
        For a 2D array, it should have a shape of (timepoints, permutations).
        For a 3D array, it should have a shape of (timepoints, permutations, p), 
        where p represents the number of predictors/features. The first dimension corresponds to timepoints, 
        the second dimension to different permutations, and the third (if present) to multiple features.
    pval (numpy.ndarray): 
        1D or 2D array of p-values obtained from permutation testing.
        For a 1D array, it should have a shape of (timepoints), containing a single p-value per timepoint.
        For a 2D array, it should have a shape of (timepoints, p), where p represents the number of predictors/features.
    alpha (float, optional), default=0.05: 
        Significance level for cluster-based correction.
    individual_feature (bool, optional), default=False: 
        If True, the cluster-based correction is performed separately for each feature in the test_statistics.
        If False, the correction is applied to the entire p-value matrix.
        
    Returns:
    ----------
    p_values (numpy.ndarray): 
        Corrected p-values after cluster-based correction.
    """
    if test_statistics is []:
        raise ValueError("The variable 'test_statistics' is an empty list. To run the cluster-based permutation correction, you need to set 'test_statistics_option=True' when performing your test, as the distribution of test statistics is required for this function.")

    if individual_feature:
        if test_statistics.ndim==3:
            q_size = test_statistics.shape[-1]
            p_values = np.zeros_like(pval)
        else:
            test_statistics = np.expand_dims(test_statistics,2)
            q_size = test_statistics.shape[-1]
            pval = np.expand_dims(pval,axis=1)
            p_values = np.zeros_like(pval)
        for q_i in range(q_size):
            # Compute mean and standard deviation under the null hypothesis
            mean_h0 = np.squeeze(np.mean(test_statistics[:,:,q_i], axis=1))
            std_h0 = np.std(test_statistics[:,:,q_i], axis=1)

            # Initialize array to store maximum cluster sums for each permutation
            Nperm = test_statistics[:,:,q_i].shape[1]
            # Not including the first permuation
            max_cluster_sums = np.zeros(Nperm-1)

            # Define zval_thresh threshold based on alpha
            zval_thresh = norm.ppf(1 - alpha)
            
            # Iterate over permutations to find maximum cluster sums
            for perm in range(Nperm-1):
                # Take each permutation map and transform to Z
                thresh_nperm = (test_statistics[:,perm+1,q_i])
                if np.sum(thresh_nperm)!=0:
                    #thresh_nperm = permmaps[perm, :]
                    thresh_nperm = (thresh_nperm - np.mean(thresh_nperm)) / np.std(thresh_nperm)
                    # Threshold line at p-value
                    thresh_nperm[np.abs(thresh_nperm) < zval_thresh] = 0

                    # Find clusters
                    cluster_label = label(thresh_nperm > 0)

                    if len(np.unique(cluster_label)>0) or np.sum(cluster_label)==0:
                        # Sum values inside each cluster
                        temp_cluster_sums = [np.sum(thresh_nperm[cluster_label == label]) for label in range(1, len(np.unique(cluster_label)))]

                        # Store the sum of values for the biggest cluster
                        max_cluster_sums[perm] = max(temp_cluster_sums)
            # Calculate cluster threshold
            cluster_thresh = np.percentile(max_cluster_sums, 100 - (100 * alpha))

            # Convert p-value map calculated using permutation testing into z-scores
            pval_zmap = norm.ppf(1 - pval[:,q_i])
            # Threshold the p-value map based on alpha
            pval_zmap[(pval_zmap)<zval_thresh] = 0

            # Find clusters in the real thresholded pval_zmap
            # If they are too small, set them to zero
            cluster_labels = label(pval_zmap>0)
        
            for cluster in range(1,len(np.unique(cluster_labels))):
                if np.sum(pval_zmap[cluster_labels == cluster]) < cluster_thresh:
                    #print(np.sum(region.intensity_image))
                    pval_zmap[cluster_labels == cluster] = 0
                    
            # Convert z-map to p-values
            p_values[:,q_i] = 1 - norm.cdf(pval_zmap)
            p_values[p_values[:,q_i] == 0.5, q_i] = 1
    
    else:    
        # Compute mean and standard deviation under the null hypothesis
        mean_h0 = np.squeeze(np.mean(test_statistics, axis=1))
        std_h0 = np.std(test_statistics, axis=1)

        # Initialize array to store maximum cluster sums for each permutation
        Nperm = test_statistics.shape[1]
        # Not including the first permuation
        max_cluster_sums = np.zeros(Nperm-1)

        # Define zval_thresh threshold based on alpha
        zval_thresh = norm.ppf(1 - alpha)
        
        # Iterate over permutations to find maximum cluster sums
        for perm in range(Nperm-1):
            # 
            if test_statistics.ndim==3:
                thresh_nperm = np.squeeze(test_statistics[:, perm+1, :])
                thresh_nperm = (thresh_nperm - mean_h0) / std_h0

                # Threshold image at p-value
                thresh_nperm[np.abs(thresh_nperm) < zval_thresh] = 0

                # Find clusters using connected components labeling
                cluster_label = label(thresh_nperm > 0)
                regions = regionprops(cluster_label, intensity_image=thresh_nperm)

                if regions:
                    # Sum values inside each cluster
                    temp_cluster_sums = [np.sum(region.intensity_image) for region in regions]
                    max_cluster_sums[perm] = max(temp_cluster_sums)
            # Otherwise it is a 2D matrix
            else: 
                # Take each permutation map and transform to Z
                thresh_nperm = (test_statistics[:,perm+1])
                if np.sum(thresh_nperm)!=0:
                    #thresh_nperm = permmaps[perm, :]
                    thresh_nperm = (thresh_nperm - np.mean(thresh_nperm)) / np.std(thresh_nperm)
                    # Threshold line at p-value
                    thresh_nperm[np.abs(thresh_nperm) < zval_thresh] = 0

                    # Find clusters
                    cluster_label = label(thresh_nperm > 0)

                    if len(np.unique(cluster_label)>0) or np.sum(cluster_label)==0:
                        # Sum values inside each cluster
                        temp_cluster_sums = [np.sum(thresh_nperm[cluster_label == label]) for label in range(1, len(np.unique(cluster_label)))]

                        # Store the sum of values for the biggest cluster
                        max_cluster_sums[perm] = max(temp_cluster_sums)
        # Calculate cluster threshold
        cluster_thresh = np.percentile(max_cluster_sums, 100 - (100 * alpha))

        # Convert p-value map calculated using permutation testing into z-scores
        pval_zmap = norm.ppf(1 - pval)
        # Threshold the p-value map based on alpha
        pval_zmap[(pval_zmap)<zval_thresh] = 0

        # Find clusters in the real thresholded pval_zmap
        # If they are too small, set them to zero
        cluster_labels = label(pval_zmap>0)
        
        if test_statistics.ndim==3:
            regions = regionprops(cluster_labels, intensity_image=pval_zmap)

            for region in regions:
                # If real clusters are too small, remove them by setting to zero
                if np.sum(region.intensity_image) < cluster_thresh:
                    pval_zmap[cluster_labels == region.label] = 0
        else: 
            for cluster in range(1,len(np.unique(cluster_labels))):
                if np.sum(pval_zmap[cluster_labels == cluster]) < cluster_thresh:
                    #print(np.sum(region.intensity_image))
                    pval_zmap[cluster_labels == cluster] = 0
                
        # Convert z-map to p-values
        p_values = 1 - norm.cdf(pval_zmap)
        p_values[p_values == 0.5] = 1
    return p_values

def get_indices_array(idx_data):
    """
    Generates an indices array based on given data indices.

    Parameters:
    --------------
    idx_data (numpy.ndarray): 
        The data indices array.

    Returns:
    ----------  
    idx_array (numpy.ndarray): 
        The generated indices array.
        
    Example:
    ----------      
    >>> idx_data = np.array([[0, 3], [3, 6], [6, 9]])
    >>> get_indices_array(idx_data)
    array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    """
    # Create a copy of idx_data to avoid modifying the original outside the function
    idx_data_copy = np.copy(idx_data)
    
    # # # Check if any values in column 1 are equal to any values in column 2
    # # # If equal remove one value from the second column
    # # # if np.any(np.isin(idx_data_copy[:, 0], idx_data_copy[:, 1])):
    # # #     idx_data_copy[:, 1] -= 1

    # Get an array of indices based on the given idx_data ranges
    max_value = np.max(idx_data_copy[:, 1])
    idx_array = np.zeros(max_value, dtype=int)
    for count, (start, end) in enumerate(idx_data_copy):
        idx_array[start:end] = count
    return idx_array

def get_indices_timestamp(n_timestamps, n_subjects):
    """
    Generate indices of the timestamps for each subject in the data.

    Parameters:
    --------------
    n_timestamps (int): 
        Number of timestamps.
    n_subjects (int): 
        Number of subjects.

    Returns:
    ----------  
    indices (ndarray): 
        Array representing the indices of the timestamps for each subject.

    Example:
    ----------
    >>> n_timestamps = 3
    >>> n_subjects = 4
    >>> get_indices_timestamp(n_timestamps, n_subjects)
    array([[ 0,  3],
           [ 3,  6],
           [ 6,  9],
           [ 9, 12]])
    """
    indices = np.column_stack([np.arange(0, n_timestamps * n_subjects, n_timestamps),
                               np.arange(0 + n_timestamps, n_timestamps * n_subjects + n_timestamps, n_timestamps)])

    return indices

def get_indices_session(data_label):
    """
    Generate session indices in the data based on provided labels.
    This is done by using 'data_label' to define sessions and generates corresponding indices. 
    The resulting 'idx_data_sessions' array represents the intervals for each session in the data.
    
    Parameters:
    --------------
    data_label (ndarray): 
        Array representing the labels for data to be indexed into sessions.

    Returns:
    ----------  
    idx_data_sessions (ndarray): 
        The indices of datapoints within each session. It should be a 2D array 
        where each row represents the start and end index for a trial. 

    Example:
    ----------
    >>> data_label = np.array([0, 0, 0, 1, 1, 2])
    >>> get_indices_session(data_label)
    array([[0, 3],
           [3, 5],
           [5, 6]])

    """
    # Get unique labels from the data_label array
    unique_labels = np.unique(data_label)

    # Initialize an array to store session indices
    idx_data_sessions  = np.zeros((len(unique_labels), 2)).astype(int)

    # Iterate over unique labels
    for session in range(len(unique_labels)):
        # Count occurrences of the current label in the data_label array
        occurrences = len(data_label[data_label == unique_labels[session]])

        # Update the session indices array
        if session == 0:
            idx_data_sessions [session, 1] = occurrences
        else:
            idx_data_sessions [session, 0] = idx_data_sessions [session - 1, 1]
            idx_data_sessions [session, 1] = idx_data_sessions [session - 1, 1] + occurrences

    # Return the generated session indices array
    return idx_data_sessions 

def get_indices_from_list(data_list, count_timestamps = True):
    """
    Generate indices representing the start and end timestamps for each subject or session from a given data list.

    Parameters:
    --------------
    data_list (list): 
        List containing data for each subject or session.
    count_timestamps (bool), default=True: 
        If True, counts timestamps for each element in data_list, otherwise assumes each element in data_list is already a count of timestamps.

    Returns:
    ----------  
    indices (ndarray): 
        Array with start and end indices for each subject's timestamps.
    
    Example:
    ----------
    >>> data_list = [[1, 2, 3], [4, 5], [6]]
    >>> get_indices_from_list(data_list, count_timestamps=True)
    array([[0, 3],
           [3, 5],
           [5, 6]])

    >>> data_list = [3, 2, 1]
    >>> get_indices_from_list(data_list, count_timestamps=False)
    array([[0, 3],
           [3, 5],
           [5, 6]])
    """
    # Initialize an empty NumPy array to store start and end indices for each subject
    indices = np.zeros((len(data_list), 2), dtype=int)
    
    # Iterate through each element in the data list along with its index
    for i, data in enumerate(data_list):
        if count_timestamps:
            # Get the number of timestamps for the current subject or session
            n_timestamps = len(data)
        else: 
            n_timestamps = data 
        
        # Update indices based on whether it's the first subject or subsequent ones
        if i == 0:
            indices[i, 1] = n_timestamps  # For the first subject, set the end index
        else:
            indices[i, 0] = indices[i - 1, 1]  # Set the start index based on the previous subject's end index
            indices[i, 1] = indices[i - 1, 1] + n_timestamps  # Set the end index
        
    # Return the generated indices array
    return indices

def get_indices_update_nan(idx_data, nan_mask):
    """
    Update interval indices based on missing values in the data.

    Parameters:
    -----------
    idx_data (numpy.ndarray):
        Array of shape (n_intervals, 2) representing the start and end indices of each interval.
    nan_mask (bool):
        Boolean mask indicating the presence of missing values in the data.

    Returns:
    --------
    idx_data_update (numpy.ndarray):
        Updated interval indices after accounting for missing values.
    """
    # Find the indices of missing values
    nan_vals = np.where(nan_mask)
    #nan_flat = nan_vals.flatten()
    # Digitize the indices of missing values to determine which interval they belong to
    count_vals_digitize = np.digitize(nan_vals, idx_data[:, 0]) - 1
    
    if len(nan_vals[0]) > 1:
        # Sort the digitized values and count the occurrences
        count_vals_digitize_flat = count_vals_digitize.flatten()  # Convert to tuple
        #count_vals_digitize_tuple= tuple(count_vals_digitize_flat.sort())
        counts = Counter(count_vals_digitize_flat)
        
        # Update the interval indices
        idx_data_update = idx_data.copy()
        for i in range(len(idx_data)):
            if i == 0:
                idx_data_update[0, 1] -= counts[i]
                idx_data_update[1:] -= counts[i]
            else:
                idx_data_update[i:] -= counts[i]
    else:
        # If only one missing value, update the interval indices accordingly
        idx_data_update = idx_data.copy()
        count_vals_digitize = count_vals_digitize[0]
        if count_vals_digitize == 0:
            idx_data_update[0, 1] -= 1
            idx_data_update[1:] -= 1
        else:
            idx_data_update[count_vals_digitize-1, 1] -= 1
            idx_data_update[count_vals_digitize:] -= 1
    
    return idx_data_update


def get_concatenate_subjects(D_sessions):
    """
    Converts a 3D matrix into a 2D matrix by concatenating timepoints of every subject into a new D-matrix.

    Parameters:
    --------------
    D_sessions (numpy.ndarray): 
        D-matrix for each subject.

    Returns:
    ----------  
    D_con (numpy.ndarray): 
        Concatenated D-matrix.
    """
    D_con = []

    for i in range(D_sessions.shape[1]):
        # Extend D-matrix with selected trials
        D_con.extend(D_sessions[:, i, :])

    return np.array(D_con)

def get_concatenate_sessions(D_sessions, R_sessions=None, idx_sessions=None):
    """
    Converts a  3D matrix into a 2D matrix by concatenating timepoints of every trial session into a new D-matrix.

    Parameters:
    --------------
    D_sessions (numpy.ndarray): 
        D-matrix for each session.
    R_sessions (numpy.ndarray): 
        R-matrix time for each trial.
    idx_sessions (numpy.ndarray): 
        Indices representing the start and end of trials for each session.

    Returns:
    ----------  
    D_con (numpy.ndarray): 
        Concatenated D-matrix.
    R_con (numpy.ndarray): 
        Concatenated R-matrix.
    idx_sessions_con (numpy.ndarray): 
        Updated indices after concatenation.
    """
    if idx_sessions is None:
        raise ValueError("idx_sessions cannot be None")
    D_con, R_con, idx_sessions_con = [], [], np.zeros_like(idx_sessions)

    for i, (start_idx, end_idx) in enumerate(idx_sessions):
        # Iterate over trials in each session
        for j in range(start_idx, end_idx):
            # Extend D-matrix with selected trials
            D_con.extend(D_sessions[:, j, :])
            if R_sessions is not None:
                # Extend time list for each trial
                R_con.extend([R_sessions[j]] * D_sessions.shape[0])


        # Update end index for the concatenated D-matrix
        idx_sessions_con[i, 1] = len(D_con)

        if i < len(idx_sessions) - 1:
            # Update start index for the next session if not the last iteration
            idx_sessions_con[i + 1, 0] = idx_sessions_con[i, 1]

    # Convert lists to numpy arrays
    return np.array(D_con), np.array(R_con), idx_sessions_con


def reconstruct_concatenated_design(D_con, D_sessions=None, n_timepoints=None, n_trials=None, n_channels = None):
    """
    Reconstructs the concatenated D-matrix to the original session variables.

    Parameters:
    --------------
    D_con (numpy.ndarray): 
        Concatenated D-matrix.
    D_sessions (numpy.ndarray, optional): 
        Original D-matrix for each session.
    n_timepoints (int, optional): 
        Number of timepoints per trial.
    n_trials (int, optional): 
        Number of trials per session.
    n_channels (int, optional): 
        Number of channels.

    Returns:
    ----------  
    D_reconstruct (numpy.ndarray): 
        Reconstructed D-matrix for each session.
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


def identify_coloumns_for_t_and_f_tests(R_data, method, Nperm, identify_categories=False, category_lim=None,permute_beta=False):
    """
    Detect columns in R_data that are categorical. Used to detect which columns to perm t-statistics and F-statistics for later analysis.

    Parameters:
    -----------
    R_data (numpy.ndarray): 
        The 3D array containing categorical values.
    identify_categories : bool or list or numpy.ndarray, optional, default=False
        If True, automatically identify categorical columns. If list or ndarray, use the provided list of column indices.
    method (str, optional, default="univariate"): 
        The method to perform the tests. Only "univariate" is currently supported.
    category_lim (int or None, optional, default=None): 
        Maximum allowed number of categories for F-test. Acts as a safety measure for columns 
        with integer values, like age, which may be mistakenly identified as multiple categories.
    
    Returns:
    -----------
    category_columns (dict):
        A dictionary marking the columns where t-test ("t_test_cols") and F-test ("f_anova_cols") have been applied.
    """
    # Initialize variable
    category_columns = {'t_test_cols': [], 'f_anova_cols': [], 'f_reg_cols': []} 
    idx_cols =np.arange(R_data.shape[-1])
    # Perform t-statistics for binary columns
    if identify_categories == True or isinstance(identify_categories, (list,np.ndarray)):
        if identify_categories==True:
            # Identify binary columns automatically in R_data
            # Do not calculate t-statistics when using regression for two categories
            if method!="multivariate" or permute_beta:
                category_columns["t_test_cols"] = [col for col in range(R_data.shape[-1]) if np.unique(R_data[0,:, col]).size == 2]
            if category_lim != None:
                if method == "multivariate":
                    category_columns["f_anova_cols"] = [col for col in range(R_data.shape[-1]) 
                                    if np.unique(R_data[0,:, col]).size >= 2  # Check if more than 2 unique values
                                    and np.unique(R_data[0,:, col]).size < category_lim] # Check if the data type is above category_lim
                else:
                    category_columns["f_anova_cols"] = [col for col in range(R_data.shape[-1]) 
                                                    if np.unique(R_data[0,:, col]).size > 2  # Check if more than 2 unique values
                                                    and np.unique(R_data[0,:, col]).size < category_lim] # Check if the data type is above category_lim
            # if Nperm==1:
            #     category_columns["t_test_cols"] = [col for col in range(R_data.shape[-1]) if np.unique(R_data[0,:, col]).size == 2]
                idx_test = category_columns["t_test_cols"]+category_columns["f_anova_cols"]
                # columns not in idx_cols would be f_reg_cols
                category_columns["f_reg_cols"] = list(idx_cols[~np.isin(idx_cols,idx_test)])
            else:

                unique_counts = [np.unique(R_data[0, :, col]).size for col in range(R_data.shape[-1])]

                if max(unique_counts) > category_lim:
                    warnings.warn(
                        f"Detected more than {category_lim} unique numbers in column {idx_cols[np.array(unique_counts)>category_lim]} dataset. "
                        f"If this is not intended as categorical data, you can ignore this warning. "
                        f"Otherwise, consider defining 'category_lim' to set the maximum allowed categories or specifying the indices of categorical columns."
                    )
                # category_columns["f_anova_cols"] = [col for col, unique_count in enumerate(unique_counts)
                #                         if unique_count > 2]

                 
        else:
            # Do not calculate t-statistics when using regression for two categories
            if method!="multivariate" or permute_beta:
                # Customize columns defined by the user
                category_columns["t_test_cols"] = [col for col in identify_categories if np.unique(R_data[0,:, col]).size == 2]
            #  Filter out the values from list identify_categories that are present in list category_columns["t_test_cols"]
            identify_categories_filtered = [value for value in identify_categories if value not in category_columns["t_test_cols"]]
            if category_lim != None:
                # We do not check if categories are integar values
                category_columns["f_anova_cols"] = [col for col in identify_categories_filtered 
                                                   if np.unique(R_data[0,:, col]).size > 2 
                                                   and np.unique(R_data[0,:, col]).size < category_lim]
            else:
                # We do not check if categories are integar values
                category_columns["f_anova_cols"] = [col for col in identify_categories_filtered 
                                                   if np.unique(R_data[0,:, col]).size > 2]
           
    return category_columns

def calculate_nan_regression(Din, Rin, reg_pinv, idx_data, permute_beta, perm, beta=None, test_indices=None):
    """
    Calculate the R-squared values for the regression of each dependent variable 
    in Rin on the independent variables in Din, while handling NaN values column-wise.

    Parameters:
    --------------
    Din (numpy.ndarray): 
        Input data matrix for the independent variables.
    Rin (numpy.ndarray): 
        Input data matrix for the dependent variables.
    reg_pinv (numpy.ndarray): 
        The regularized pseudo-inverse of D_data
    idx_data (numpy.ndarray): 
        Marks the indices for each trial within the session.
        It is a 2D array where each row represents the start and end index for a session.
    permute_beta (bool, optional): 
        A flag indicating whether to permute beta coefficients.
    beta (numpy.ndarray):
        beta coefficient for each session.
        It has a shape (num_session, p, q), where the first dimension 
        represents the session, the second dimension represents the featires, 
        and the third dimension represent dependent variables. 
    test_indices (numpy.ndarray):
        Indices for data points that belongs to the test-set for each session.

    Returns:
    ----------  
        R2_test (numpy.ndarray): Array of R-squared values for each regression.
    """
    
    Rin = np.expand_dims(Rin, axis=1) if Rin.ndim==1 else Rin
    q = Rin.shape[-1]
    R2_test = np.zeros(q)                
    # Calculate t-statistic for each pair of columns (D_column, R_data)
    for q_i in range(q):
        if permute_beta and idx_data is not None:
            R_column = np.expand_dims(Rin[test_indices[q_i], q_i],axis=1)
            valid_indices = np.all(~np.isnan(R_column), axis=1)
            nan_values = np.any(np.isnan(valid_indices))
            beta_column = beta[:,:,q_i]
            # Calculate the predicted values using permuted beta
            R_pred =calculate_ols_predictions(R_column, Din[test_indices[q_i],:], idx_data, beta_column, perm, permute_beta, nan_values,  valid_indices)
            R_pred = np.expand_dims(R_pred,axis=1) if R_pred.ndim==1 else R_pred

        elif idx_data is not None: 
            # Do not permute beta but calculate each session individually
            R_column = np.expand_dims(Rin[:, q_i],axis=1)
            valid_indices = np.all(~np.isnan(R_column), axis=1)
            # Detect if there are any NaN values
            nan_values= np.any(np.isnan(valid_indices))
            beta_column = beta[:,:,q_i]
            # Calculate the predicted values without permuting beta, since permute_beta =False
            R_pred =calculate_ols_predictions(R_column, Din, idx_data, beta_column, perm, permute_beta, nan_values,  valid_indices)
            R_pred = np.expand_dims(R_pred,axis=1) if R_pred.ndim==1 else R_pred
            
        else:
            R_column = np.expand_dims(Rin[:, q_i],axis=1)
            valid_indices = np.all(~np.isnan(R_column), axis=1)
            # Detect for any NaN values
            nan_values= np.any(np.isnan(valid_indices))
            # Calculate beta using the regularized pseudo-inverse of D_data
            beta = reg_pinv[:,valid_indices] @ R_column[valid_indices]  # Calculate regression_coefficients (beta)
            # Calculate the predicted values
            R_pred = Din[valid_indices] @ beta
            
        # Calculate the total sum of squares (tss)
        tss = np.sum((R_column[valid_indices] - np.mean(R_column[valid_indices], axis=0))**2, axis=0)
        # Calculate the residual sum of squares (rss)
        rss = np.sum((R_column[valid_indices]-R_pred)**2, axis=0)
        # Calculate R^2
        base_statistics = 1 - (rss / tss) #r_squared
        # Store the R2 in an array
        R2_test[q_i] = base_statistics
    return R2_test


def preprocess_response(Rin):
    # Ensure Rin is a 2D array
    Rin = np.atleast_2d(Rin)

    # Handle the case where Rin was originally a 1D array
    if Rin.shape[0] == 1:
        Rin = Rin.T  # Transpose to make it a column vector if necessary

    # Convert to dummy variables and center
    Rin = pd.get_dummies(Rin.flatten(), drop_first=False).values
    Rin_centered = Rin - np.mean(Rin, axis=0)  # Center the response data around 0

    return Rin_centered

def calculate_nan_anova_f_test(Din, Rin, reg_pinv, idx_data, permute_beta, perm=0, nan_values=False, beta= None):
    """
    Calculate the f-test values for the regression of each dependent variable 
    in Rin on the independent variables in Din, while handling NaN values column-wise.

    Parameters:
    --------------
    Din (numpy.ndarray): 
        Input data matrix for the independent variables.
    Rin (numpy.ndarray): 
        Input data matrix for the dependent variables.
    reg_pinv (numpy.ndarray): 
        The regularized pseudo-inverse of D_data.
    idx_data (numpy.ndarray): 
        Marks the indices for each trial within the session.
        It is a 2D array where each row represents the start and end index for a session.       
    permute_beta (bool, optional): 
        A flag indicating whether to permute beta coefficients.
    perm (int): 
        The permutation index.
    nan_values (bool, optional), default=False:: 
        A flag indicating whether there are NaN values.
    beta (numpy.ndarray):
    
    Returns:
    ----------  
        R2_test (numpy.ndarray): Array of f-test values for each regression.
    """
    if nan_values:
        # Calculate F-statistics if there are Nan_values
        # Expand the dimension of Rin, if it is just a single array and center Rin
        Rin = preprocess_response(Rin)
        q = Rin.shape[-1]
        p_value = np.zeros(q)
        f_statistic = np.zeros(q)
        # Calculate t-statistic for each pair of columns (D_column, R_data)
        for i in range(q):
            # Indentify columns with NaN values
            R_column = np.expand_dims(Rin[:, i],axis=1)
            valid_indices = np.all(~np.isnan(R_column), axis=1)
                  
            if permute_beta and idx_data is not None:
                # Calculate the predicted values using permuted beta
                R_pred =calculate_ols_predictions(R_column, Din, idx_data, beta, perm, permute_beta, nan_values,  valid_indices)
 
            elif idx_data is not None: # Do not permute beta but calculate each session individually
                # Calculate the predicted values without permuting beta, since permute_beta =False
                R_pred =calculate_ols_predictions(R_column, Din, idx_data, beta, perm, permute_beta, nan_values,  valid_indices)
                
            else:
                # Calculate beta coefficients using regularized pseudo-inverse of D_data
                beta = reg_pinv[:,valid_indices] @ R_column[valid_indices]  # Calculate regression_coefficients (beta)
                # Calculate the predicted values
                R_pred = Din[valid_indices] @ beta
            # Calculate the total sum of squares (tss)
            tss = np.sum((R_column[valid_indices] - np.mean(R_column[valid_indices], axis=0))**2, axis=0)
            # Calculate the residual sum of squares (rss)
            rss = np.sum((R_column[valid_indices]-R_pred)**2, axis=0)
            # Calculate the parametric p-values using F-statistics
            # Calculate the explained sum of squares (ESS)
            ess = tss - rss
            # Calculate the degrees of freedom for the model and residuals
            df_model = Din.shape[1]  # Number of predictors including intercept
            df_resid = Din.shape[0] - df_model
            # Calculate the mean squared error (MSE) for the model and residuals
            MSE_model = ess / df_model
            MSE_resid = rss / df_resid
            # Calculate the F-statistic
            base_statistics = (MSE_model / MSE_resid)# Calculate R^2
            # Store the R2 in an array
            f_statistic[i] = base_statistics
            p_value[i] = 1 - f.cdf(f_statistic, df_model, df_residual)
    else:
        # Expand the dimension of Rin, if it is just a single array and center Rin
        Rin = preprocess_response(Rin)
        if permute_beta:    
            if beta is None:
                # permute beta and calulate predicted values
                beta, _ = calculate_ols_beta(reg_pinv, Rin, idx_data)
            R_pred =calculate_ols_predictions(Rin, Din, idx_data, beta, perm, permute_beta)
        else:
            # Calculate f-statistics

            # Fit the original model 
            beta = reg_pinv @ Rin  # Calculate regression_coefficients (beta)
            # Calculate the predicted values
            R_pred = Din @ beta
        # Calculate the residuals
        residuals = Rin - R_pred
        
        # Calculate sum of squares for the model (SS Model) and residuals (SS Residual)
        ss_total = np.sum((Rin - np.mean(Rin, axis=0))**2)
        ss_residual = np.sum(residuals**2)
        ss_model = ss_total - ss_residual
        
        # Degrees of freedom
        n = Rin.shape[0]  # Number of observations
        p = Rin.shape[1]  # Number of predictors
        df_model = p - 1  # Degrees of freedom for the model (excluding intercept)
        df_residual = n - p  # Degrees of freedom for the residuals
        
        # Mean squares
        ms_model = ss_model / df_model
        ms_residual = ss_residual / df_residual
        
        # F-statistic
        f_statistic = ms_model / ms_residual
        p_value = 1 - f.cdf(f_statistic, df_model, df_residual)
    return f_statistic, p_value

def calculate_nan_regression_f_test(Din, Rin, reg_pinv, idx_data, permute_beta, perm=0, nan_values=False, beta= None):
    """
    Calculate the f-test values for the regression of each dependent variable 
    in Rin on the independent variables in Din, while handling NaN values column-wise.

    Parameters:
    --------------
    Din (numpy.ndarray): 
        Input data matrix for the independent variables.
    Rin (numpy.ndarray): 
        Input data matrix for the dependent variables.
    reg_pinv (numpy.ndarray): 
        The regularized pseudo-inverse of D_data.
    idx_data (numpy.ndarray): 
        Marks the indices for each trial within the session.
        It is a 2D array where each row represents the start and end index for a session.       
    permute_beta (bool, optional): 
        A flag indicating whether to permute beta coefficients.
    perm (int): 
        The permutation index.
    nan_values (bool, optional), default=False:: 
        A flag indicating whether there are NaN values.
    beta (numpy.ndarray):
    
    Returns:
    ----------  
        R2_test (numpy.ndarray): Array of f-test values for each regression.
    """
    if nan_values:
        # Calculate F-statistics if there are Nan_values
        Rin = np.expand_dims(Rin, axis=1) if Rin.ndim==1 else Rin
        q = Rin.shape[-1]
        f_statistic = np.zeros(q)
        p_value = np.zeros(q)
        # Calculate t-statistic for each pair of columns (D_column, R_data)
        for i in range(q):
            # Indentify columns with NaN values
            R_column = np.expand_dims(Rin[:, i],axis=1)
            valid_indices = np.all(~np.isnan(R_column), axis=1)
                  
            if permute_beta and idx_data is not None:
                # Calculate the predicted values using permuted beta
                R_pred =calculate_ols_predictions(R_column, Din, idx_data, beta, perm, permute_beta, nan_values,  valid_indices)
 
            elif idx_data is not None: # Do not permute beta but calculate each session individually
                # Calculate the predicted values without permuting beta, since permute_beta =False
                R_pred =calculate_ols_predictions(R_column, Din, idx_data, beta, perm, permute_beta, nan_values,  valid_indices)
                
            else:
                # Calculate beta coefficients using regularized pseudo-inverse of D_data
                beta = reg_pinv[:,valid_indices] @ R_column[valid_indices]  # Calculate regression_coefficients (beta)
                # Calculate the predicted values
                R_pred = Din[valid_indices] @ beta
            # Calculate the total sum of squares (tss)
            tss = np.sum((R_column[valid_indices] - np.mean(R_column[valid_indices], axis=0))**2, axis=0)
            # Calculate the residual sum of squares (rss)
            rss = np.sum((R_column[valid_indices]-R_pred)**2, axis=0)
            # Calculate the parametric p-values using F-statistics
            # Calculate the explained sum of squares (ESS)
            ess = tss - rss
            # Calculate the degrees of freedom for the model and residuals
            df_model = Din.shape[1]  # Number of predictors including intercept
            df_resid = Din.shape[0] - df_model
            # Calculate the mean squared error (MSE) for the model and residuals
            MSE_model = ess / df_model
            MSE_resid = rss / df_resid
            # Calculate the F-statistic
            base_statistics = (MSE_model / MSE_resid)# Calculate R^2
            # Store the R2 in an array
            f_statistic[i] = base_statistics
            p_value[i] = 1 - f.cdf(f_statistic, df_model, df_resid)
    else:
        # Expand the dimension of Rin, if it is just a single array
        Rin = np.expand_dims(Rin, axis=1) if Rin.ndim==1 else Rin
        if permute_beta:    
            if beta is None:
                # permute beta and calulate predicted values
                beta, _ = calculate_ols_beta(reg_pinv, Rin, idx_data)
            R_pred =calculate_ols_predictions(Rin, Din, idx_data, beta, perm, permute_beta)
        else:
            # Calculate f-statistics
            # Fit the original model 
            beta = reg_pinv @ Rin  # Calculate regression_coefficients (beta)
            # Calculate the predicted values
            R_pred = Din @ beta
        # Calculate the residual sum of squares (rss)
        rss = np.sum((Rin-R_pred)**2, axis=0)
        # Calculate the total sum of squares (tss)
        tss = np.sum((Rin - np.nanmean(Rin, axis=0))**2, axis=0)
        # Calculate the parametric p-values using F-statistics
        # Calculate the explained sum of squares (ESS)
        ess = tss - rss
        # Calculate the degrees of freedom for the model and residuals
        df_model = Din.shape[1]  # Number of predictors including intercept
        df_resid = Din.shape[0] - df_model # Degrees of freedom for the residuals
        # Calculate the mean squared error (MSE) for the model and residuals
        MSE_model = ess / df_model
        MSE_resid = rss / df_resid
        # Calculate the F-statistic
        f_statistic = (MSE_model / MSE_resid)
    
        p_value = 1 - f.cdf(f_statistic, df_model, df_resid)

    return f_statistic, p_value

def calculate_r2_univariate(Din, Rin, idx_data, beta, perm, reg_pinv, permute_beta, test_combination=False, test_indices=None):
    
    """
    Calculate the R-squared values for the regression of between Din and Rin column-wise in a univarete procedure.

    Parameters:
    --------------
    Din (numpy.ndarray): 
        Input data matrix for the independent variables.
    Rin (numpy.ndarray): 
        Input data matrix for the dependent variables.
    idx_data (numpy.ndarray): 
        An array containing the indices for each session. The array can be either 1D or 2D:
        For a 1D array, a sequence of integers where each integer labels the session number. For example, [1, 1, 1, 1, 2, 2, 2, ..., N, N, N, N, N, N, N, N].
        For a 2D array, each row represents the start and end indices for the trials in a given session, with the format [[start1, end1], [start2, end2], ..., [startN, endN]].  
    beta (numpy.ndarray):
        beta coefficient for each session.
        It has a shape (num_session, p, q), where the first dimension 
        represents the session, the second dimension represents the featires, 
        and the third dimension represent dependent variables. 
    perm (int): 
        The permutation index.    
    reg_pinv (numpy.ndarray): 
        The regularized pseudo-inverse of D_data
    permute_beta (bool, optional): 
        A flag indicating whether to permute beta coefficients.
    test_combination (str), default=False:       
        Specifies the combination method.
        Valid options: "True", "across_columns", "across_rows".
    test_indices (numpy.ndarray), default=None:
        Indices for data points that belongs to the test-set for each session.
    
    Returns:
    --------
    base_statistics (numpy.ndarray): 
        The expanded base statistics array.
    pval_matrix (numpy.ndarray): 
        calculated p-values estimated from the F-statistic
    """
    base_statistics = np.zeros((Din.shape[-1],Rin.shape[-1]))
    pval_matrix = np.zeros((Din.shape[1], Rin.shape[1]))
    for q_i in range(Rin.shape[-1]):
        # Identify columns with NaN values
        R_column = np.expand_dims(Rin[test_indices[0] if len(test_indices) == 1 else test_indices[q_i], q_i], axis=1)
        valid_indices = np.all(~np.isnan(R_column), axis=1)

        if test_combination !=False:
            for p_j in range(Din.shape[-1]):
                D_column = np.expand_dims(Din[test_indices[0 if len(test_indices) == 1 else q_i], p_j], axis=1)
                reg_pinv_column = np.expand_dims(reg_pinv[p_j, test_indices[0 if len(test_indices) == 1 else q_i]], axis=0)
                beta_column, _ = np.squeeze(calculate_ols_beta(reg_pinv_column, R_column, idx_data)[0]) if beta is None else beta[:,p_j,q_i]
                nan_values = np.any(np.isnan(valid_indices)) or np.any(np.isnan(D_column))
                R_pred =calculate_ols_predictions(R_column, D_column, idx_data, beta_column, perm, permute_beta, nan_values,  valid_indices)
                # Calculate the residual sum of squares (rss)
                rss = np.sum((np.expand_dims(Rin[valid_indices,q_i],axis=1)-R_pred)**2, axis=0)
                # Calculate the total sum of squares (tss)
                tss = np.sum((Rin[valid_indices,q_i] - np.nanmean(Rin[valid_indices,q_i], axis=0))**2, axis=0)
                ess = tss - rss
                # Calculate the degrees of freedom for the model and residuals
                df_model = D_column.shape[1]  # Number of predictors including intercept
                df_resid = D_column.shape[0] - df_model
                # Calculate the mean squared error (MSE) for the model and residuals
                MSE_model = ess / df_model
                MSE_resid = rss / df_resid
                # Calculate the F-statistic
                f_statistics = MSE_model / MSE_resid
                # Calculate the p-value for the F-statistic
                pval = 1 - f.cdf(f_statistics, df_model, df_resid)
                # Store the p-value
                pval_matrix[p_j, q_i] = pval 
        else:
            # Explained variance
            for p_j in range(Din.shape[-1]):
                D_column = np.expand_dims(Din[test_indices[0 if len(test_indices) == 1 else q_i], p_j], axis=1)
                reg_pinv_column = np.expand_dims(reg_pinv[p_j, test_indices[0 if len(test_indices) == 1 else q_i]], axis=0)
                beta_column = np.squeeze(calculate_ols_beta(reg_pinv_column, R_column, idx_data)[0]) if beta is None else beta[:,p_j,q_i]
                nan_values = np.any(np.isnan(valid_indices)) or np.any(np.isnan(D_column))
                R_pred =calculate_ols_predictions(R_column, D_column, idx_data, beta_column, perm, permute_beta, nan_values,  valid_indices)
                # Calculate the residual sum of squares (rss)
                rss = np.sum((R_column - R_pred)**2, axis=0)
                # Calculate the total sum of squares (tss)
                tss = np.sum((R_column - np.mean(R_column, axis=0))**2, axis=0)
                base_statistics[p_j, q_i] = 1 - (rss / tss) #r_squared 

    return base_statistics, pval_matrix


def calculate_beta_session(reg_pinv, Rin, idx_data, permute_beta, category_lim):
    """
    Calculate beta coefficients for each session. 
    If there are NaN values the procedure will be done per column.

    Parameters:
    --------------
    reg_pinv (numpy.ndarray): 
        The regularized pseudo-inverse of D_data.
    Rin (numpy.ndarray): 
        Input data matrix for the dependent variables.
    idx_data (numpy.ndarray): 
        Marks the indices for each trial within the session.
        It is a 2D array where each row represents the start and end index for a session.    
    permute_beta (bool, optional), default=False: 
        A flag indicating whether to permute beta coefficients.
    category_lim : int or None, optional, default=10
        Maximum allowed number of categories for F-test. Acts as a safety measure for columns 
        with integer values, like age, which may be mistakenly identified as multiple categories.  
    """

    # detect nan values
    nan_values = np.sum(np.isnan(Rin))>0
    test_indices_list =[]
    # Do it columnwise if NaN values are detected
    if nan_values:
        Rin = np.expand_dims(Rin, axis=1) if Rin.ndim==1 else Rin
        q = Rin.shape[-1]
        beta = np.zeros((len(idx_data),len(reg_pinv),Rin.shape[1] ))
        # Calculate beta coefficients for each session
        for col in range(q):
            # Indentify columns with NaN values
            R_column = np.expand_dims(Rin[:, col],axis=1)
            valid_indices = np.all(~np.isnan(R_column), axis=1)
                    
            if permute_beta and idx_data is not None:
                # Calculate the predicted values using permuted beta
                beta_col, test_indices = calculate_ols_beta(reg_pinv, R_column, idx_data,  category_lim)
                beta[:,:,col] = np.squeeze(beta_col)

            else:
                # Calculate beta coefficients using regularized pseudo-inverse of D_data
                beta[:,:,col], test_indices = reg_pinv[:,valid_indices] @ R_column[valid_indices]  # Calculate regression_coefficients (beta)
            
            test_indices_list.append(test_indices)
    else:
        # permute beta and calulate predicted values
        beta, test_indices = calculate_ols_beta(reg_pinv, Rin, idx_data, category_lim)
        test_indices_list.append(test_indices)
    return beta, test_indices_list
        
def calculate_ols_beta(reg_pinv, Rin, idx_data,  category_lim= 10):
    """
    Calculate beta for ordinary least squares regression.

    Parameters:
    -----------
    reg_pinv (numpy.ndarray): 
        The regularized pseudo-inverse of D_data
    Rin (numpy.ndarray):
        Response matrix.
    idx_data (numpy.ndarray):
        Indices representing the start and end of trials.
    nan_values (numpy.ndarray):
        Whether to handle NaN values. Default is False.

    Returns:
    --------
    beta (numpy.ndarray):
        Beta coefficients
    test_indices_list (list)
    """
    # detect nan values
    nan_values = np.sum(np.isnan(Rin))>0
    test_indices_list = []
    # test_indices.sort()
    # values, counts = np.unique(Rin[train_indices], return_counts=True)
    # values, counts = np.unique(Rin[test_indices], return_counts=True)
        
    if nan_values:
        # Need to do a columnwise calculcation
        
        # Calculate F-statistics if there are Nan_values
        Rin = np.expand_dims(Rin, axis=1) if Rin.ndim==1 else Rin
        # q = Rin.shape[-1]
        
        beta = []
        # Indentify columns with NaN values
        Rin = np.expand_dims(Rin, axis=1) if Rin.ndim==1 else Rin
        indices = np.all(~np.isnan(Rin), axis=1)
        indices_range = np.arange(len(indices))
        nan_indices = indices_range[~indices]
        
        bool_data = ~np.isin(indices_range, nan_indices)
        # make a train test split, which we are going to estimate the beta's from for each session.
    
        for start, end in idx_data:
            idx_range = np.arange(start, end)
    
            # Find matches
            matches = np.isin(nan_indices, idx_range)
            matched_values = nan_indices[matches]
            # Calculate the range length considering matched_values
            range_length = end - start - len(matched_values) if len(matched_values) >0 else end - start
            
            unique_values = np.unique(Rin[idx_range[bool_data[idx_range]],:])
            if len(unique_values)<10:
                # Create the train test split
                train_indices, test_indices = train_test_split(np.arange(range_length), test_size=0.5, stratify=Rin[idx_range[bool_data[idx_range]],:])
                # train_indices, test_indices = train_test_split(np.arange(range_length), test_size=0.5, stratify=Rin[idx_range[bool_data[idx_range]],:], random_state=42)
            else:
                # Perform random split for continuous values
                train_indices, test_indices = train_test_split(np.arange(end-start), test_size=0.5)
                # train_indices, test_indices = train_test_split(np.arange(end-start), test_size=0.5, random_state=42)
                
            train_indices.sort()
            test_indices.sort()
            # adjust the indices so they account for the NaN values
            for value in matched_values:
                index_increase_train =train_indices>=value
                train_indices =train_indices[index_increase_train]+1
                index_increase_test =test_indices>=value
                test_indices =test_indices[index_increase_test]+1
                
            train_indices+=start
            test_indices+=start
            session_beta = reg_pinv[:,train_indices] @ Rin[train_indices, :]
            beta.append(session_beta) 
            test_indices_list.append(test_indices)
    elif Rin.ndim==2: 
        beta = []
        for start, end in idx_data:
            unique_values = np.unique(Rin[start:end,:])
            if len(unique_values)<category_lim:
                train_indices, test_indices = train_test_split(np.arange(end-start), test_size=0.5, stratify=Rin[start:end,:])
                # train_indices, test_indices = train_test_split(np.arange(end-start), test_size=0.5, stratify=Rin[start:end,:],random_state=42)
            else:
                # Perform random split for continuous values
                train_indices, test_indices = train_test_split(np.arange(end-start), test_size=0.5)
                # train_indices, test_indices = train_test_split(np.arange(end-start), test_size=0.5, random_state=42)
            
            train_indices.sort()
            test_indices.sort()
            train_indices+=start
            test_indices+=start
            session_beta = reg_pinv[:,train_indices] @ Rin[train_indices, :]
            beta.append(session_beta)
            test_indices_list.append(test_indices)

    else:
        # Column wise calculation
        beta = []
        for start, end in idx_data:
            if len(unique_values)<10:
                train_indices, test_indices = train_test_split(np.arange(end-start), test_size=0.5, stratify=Rin[start:end])
            else:
                # Perform random split for continuous values
                train_indices, test_indices = train_test_split(np.arange(end-start), test_size=0.5)
            train_indices.sort()
            test_indices.sort()
            train_indices+=start
            test_indices+=start
            session_beta = reg_pinv[:,train_indices] @ Rin[train_indices]
            beta.append(session_beta)
            test_indices_list.append(test_indices)
  
    return np.array(beta), test_indices_list



def calculate_ols_predictions(Rin, Din, idx_data, beta, perm, permute_beta=False, nan_values=False,  valid_indices=None):
    """
    Calculate predictions for ordinary least squares regression.

    Parameters:
    -----------
    reg_pinv (numpy.ndarray): 
        The regularized pseudo-inverse of D_data.
    D_data (numpy.ndarray): 
        The input data array.
    R_data (numpy.ndarray): 
        The dependent variable.
        Design matrix.
    idx_data (numpy.ndarray): 
        Marks the indices for each trial within the session.
        It is a 2D array where each row represents the start and end index for a session.
    perm (int): 
        Value marking the number of permutations that have been performed. 
    permute_beta (bool, optional): 
        Whether to permute beta. Default is False.
    nan_values (bool, optional): 
        Whether to handle NaN values. Default is False.
    valid_indices (numpy.ndarray, optional): 
        Valid indices. Default is None.


    idx_data (numpy.ndarray): 
        Marks the indices for each trial within the session.
        It is a 2D array where each row represents the start and end index for a session.
    Returns:
    --------
    numpy.ndarray
        Predicted values.
    """
    if nan_values:
        R_pred = []
        valid_indices_range = np.arange(len(valid_indices))
        invalid_indices = valid_indices_range[~valid_indices]

        # permute beta
        beta_perm = np.random.permutation(np.array(beta)) if permute_beta and perm != 0 else beta
        
        for idx, (start, end) in enumerate(idx_data):
            idx_range = np.arange(start, end)
            bool_data = ~np.isin(idx_range, invalid_indices)
            R_pred.append(np.dot(Din[idx_range[bool_data], :], beta_perm[idx]))
        
        return np.concatenate(R_pred, axis=0)
    
    elif Rin.ndim==2: 
        R_pred = np.zeros((len(Din), Rin.shape[-1]))
        # Permute beta if perm is not equal to 0
        beta_perm = np.random.permutation(beta) if permute_beta and perm != 0 else beta
        beta_perm = np.expand_dims(beta_perm, axis=1) if beta_perm.ndim ==1 else beta_perm
        for idx, (start, end) in enumerate(idx_data):
            # Make prediction with beta
            if beta_perm[idx, :].ndim ==1:
                # Reshape beta_perm[idx, :] to a 2D array with a single row or column based on Din[start:end, :].shape
                beta_reshaped = beta_perm[idx, :].reshape(-1, 1) if Din[start:end, :].shape[-1] != 1 else beta_perm[idx, :].reshape(1, -1)
                # Perform the matrix multiplication
                R_pred[start:end, :] = Din[start:end, :] @ beta_reshaped
            else:
                R_pred[start:end,:] = Din[start:end, :] @ beta_perm[idx, :]
                
            
        return R_pred
    else:
        # Column wise calculation
        R_pred = []
        # Permute beta
        beta_perm = np.random.permutation(np.array(beta)) if permute_beta and perm != 0 else np.array(beta)
        for idx, (start, end) in enumerate(idx_data):
            idx_range = np.arange(start, end)
            R_pred.append(np.dot(Din[idx_range, :], beta_perm[idx]))
        
        return np.concatenate(R_pred, axis=0)


def calculate_nan_correlation_matrix(D_data, R_data, test_combination=False):
    """
    Calculate the correlation matrix between independent variables (D_data) and dependent variables (R_data),
    while handling NaN values column by column of dimension p without  without removing entire rows.
    
    Parameters:
    --------------
    D_data (numpy.ndarray): 
        Input D-matrix for the independent variables.
    R_data (numpy.ndarray): 
        Input R-matrix for the dependent variables.

    Returns:
    ----------  
    correlation_matrix (numpy.ndarray): 
        correlation matrix between columns in D_data and R_data.
    """
    # Initialize a matrix to store correlation coefficients
    p = D_data.shape[1]
    q = R_data.shape[1] if R_data.ndim>1 else 1
    correlation_matrix = np.zeros((p, q))
    pval_matrix = np.zeros((p, q))
    pval_combination = None
    # Calculate correlation coefficient for each pair of columns (D_column, R_column)
    for i in range(p):
        D_column = D_data[:, i]
        for j in range(q):
            # Do it column by column if R_data got more than 1 column
            R_column = R_data[:, j] if R_data.ndim>1 else R_data
            # If there are no variability between variables then set the value to NaN
            if np.all(D_column == D_column[0]) or np.all(R_column == R_column[0]):
                if test_combination in [True, "across_columns", "across_rows"]:
                   pval_matrix[i, j]  = np.nan 
                else:
                    correlation_matrix[i, j] = np.nan  
            else:
                # Find non-NaN indices for both D_column and R_column
                valid_indices = ~np.isnan(D_column) & ~np.isnan(R_column)           
                if test_combination in [True, "across_columns", "across_rows"]:
                    #pval_matrix = np.zeros(corr_matrix.shape)
                    _, pval= pearsonr(D_column[valid_indices], R_column[valid_indices])
                    pval_matrix[i, j]  = pval
                else:
                    # Calculate correlation coefficient matrix
                    corr_coef = np.corrcoef(D_column[valid_indices], R_column[valid_indices], rowvar=False)
                    # get the correlation matrix
                    correlation_matrix[i, j] = corr_coef[0, 1] 
    if test_combination!=False:
        # Calculate the geometric mean
        if test_combination== True:
            pval_combination=np.squeeze(np.exp(np.nanmean(np.log(pval_matrix))))
        elif test_combination== "across_columns":
            pval_combination=np.squeeze(np.exp(np.nanmean(np.log(pval_matrix), axis=0)))
        elif test_combination== "across_rows":
            pval_combination = np.squeeze(np.exp(np.nanmean(np.log(pval_matrix), axis=1)))
    else:
        # No pval combination has been performed, this is done if we only calculate the p-values columnwise
        pval_combination = pval_matrix.copy()
        
    return correlation_matrix, pval_combination


def calculate_anova_f_test(D_data, R_column, nan_values=False):
    """
    Calculate F-statistics for each feature of D_data against categories in R_data, while handling NaN values column by column without removing entire rows.
        - The function handles NaN values for each feature in D_data without removing entire rows.
        - NaN values are omitted on a feature-wise basis, and the F-statistic is calculated for each feature.
        - The resulting array contains F-statistics corresponding to each feature in D_data.

    Parameters:
    --------------
    D_data (numpy.ndarray): 
        The input matrix of shape (n_samples, n_features).
    R_column (numpy.ndarray): 
        The categorical labels corresponding to each sample in D_data.

    Returns:
    ----------  
    f_test (numpy.ndarray): 
        F-statistics for each feature in D_data against the categories in R_data.
 
    """
    if nan_values:
        p = D_data.shape[1]
        f_test = np.zeros(p)
        pval_array = np.zeros(p)
        
        for i in range(p):
            D_column = np.expand_dims(D_data[:, i],axis=1)
            # Find rows where both D_column and R_data are non-NaN
            valid_indices = np.all(~np.isnan(D_column) & ~np.isnan(R_column), axis=1)
            categories =np.unique(R_column)
            # Omit NaN rows in single columns - nan_policy='omit'    
            f_stats, pval = f_oneway(*[D_column[R_column*valid_indices == category] for category in categories])
            # Store the t-statistic in the matrix
            f_test[i] = f_stats
            pval_array[i] = pval
    else:
        # Calculate f-statistics if there are no NaN values
        f_test, pval_array = f_oneway(*[D_data[R_column == category] for category in np.unique(R_column)])   
        
    return f_test, pval_array

def calculate_nan_t_test(Din, R_column, nan_values=False):
    """
    Calculate the t-statistics between paired independent (D_data) and dependent (R_data) variables, while handling NaN values column by column without removing entire rows.
        - The function handles NaN values for each feature in D_data without removing entire rows.
        - NaN values are omitted on a feature-wise basis, and the t-statistic is calculated for each feature.
        - The resulting array contains t-statistics corresponding to each feature in D_data.

    Parameters:
    --------------
    Din (numpy.ndarray): 
        The input matrix of shape (n_samples, n_features).
    R_column (numpy.ndarray): 
        The binary labels corresponding to each sample in D_data.

    Returns:
    ----------  
    t_test (numpy.ndarray):
        t-statistics for each feature in D_data against the binary categories in R_data.
 
    """
    if nan_values:
        # Initialize a matrix to store t-statistics
        p = Din.shape[1]
        t_test = np.zeros(p)
        pval_array = np.zeros(p)
        # Extract non-NaN values for each group
        groups = np.unique(R_column)
        # Calculate t-statistic for each pair of columns (D_column, R_data)
        for i in range(p):
            D_column = np.expand_dims(Din[:, i],axis=1)
                
            # Find rows where both D_column and R_data are non-NaN
            # valid_indices = np.all(~np.isnan(D_column) & ~np.isnan(R_data), axis=1)
            # Omit NaN rows in single columns - nan_policy='omit'    
            t_stat, pval = ttest_ind(D_column[R_column == groups[0]], D_column[R_column == groups[1]], nan_policy='omit')

            # Store the t-statistic in the matrix
            t_test[i] = t_stat
            pval_array[i] = pval  
    else:
        # Get the t-statistic if there are no NaN values
        t_test_group = np.unique(R_column)
        # Get the t-statistic
        t_test, pval_array = ttest_ind(Din[R_column == t_test_group[0]], Din[R_column == t_test_group[1]]) 
    return t_test, pval_array


def calculate_reg_f_test(Din, Rin, idx_data, beta, perm, reg_pinv, permute_beta, test_indices=None):
    """
    Calculate F-statistics for each feature of Din against categories in R_data, while handling NaN values column by column without removing entire rows.
        - The function handles NaN values for each feature in Din without removing entire rows.
        - NaN values are omitted on a feature-wise basis, and the F-statistic is calculated for each feature.
        - The resulting array contains F-statistics corresponding to each feature in Din.

    Parameters:
    --------------
    Din (numpy.ndarray): 
        The input matrix of shape (n_samples, n_features).
    R_column (numpy.ndarray): 
        The categorical labels corresponding to each sample in Din.

    Returns:
    ----------  
    f_statistic (numpy.ndarray): 
        F-statistics for each feature in Din against the categories in R_data.
 
    """
    p_dim = Din.shape[1]
    q_dim =Rin.ndim if Rin.ndim==1 else Rin.shape[-1]
    f_statistics = np.zeros(p_dim)
    pval_array = np.zeros(p_dim)
    # base_statistics = np.zeros((Din.shape[-1],Rin.shape[-1]))
    # pval_matrix = np.zeros((p_dim, 1))
    if test_indices is not None:
        for q_i in range(q_dim):
            # Identify columns with NaN values
            R_column = np.expand_dims(Rin[test_indices[0] if len(test_indices) == 1 else test_indices[q_i], q_i], axis=1)
            valid_indices = np.all(~np.isnan(R_column), axis=1)

            for p_j in range(p_dim):
                D_column = np.expand_dims(Din[test_indices[0 if len(test_indices) == 1 else q_i], p_j], axis=1)
                reg_pinv_column = np.expand_dims(reg_pinv[p_j, test_indices[0 if len(test_indices) == 1 else q_i]], axis=0)
                beta_column, _ = np.squeeze(calculate_ols_beta(reg_pinv_column, R_column, idx_data)[0]) if beta is None else beta[:,p_j,q_i]
                nan_values = np.any(np.isnan(valid_indices)) or np.any(np.isnan(D_column))
                R_pred =calculate_ols_predictions(R_column, D_column, idx_data, beta_column, perm, permute_beta, nan_values,  valid_indices)
                # Calculate the residual sum of squares (rss)
                rss = np.sum((np.expand_dims(Rin[valid_indices,q_i],axis=1)-R_pred)**2, axis=0)
                # Calculate the total sum of squares (tss)
                tss = np.sum((Rin[valid_indices,q_i] - np.nanmean(Rin[valid_indices,q_i], axis=0))**2, axis=0)
                ess = tss - rss
                # Calculate the degrees of freedom for the model and residuals
                df_model = D_column.shape[1]  # Number of predictors including intercept
                df_resid = D_column.shape[0] - df_model
                # Calculate the mean squared error (MSE) for the model and residuals
                MSE_model = ess / df_model
                MSE_resid = rss / df_resid
                # Calculate the F-statistic
                f_statistic = MSE_model / MSE_resid
                # Calculate the p-value for the F-statistic
                pval = 1 - f.cdf(f_statistic, df_model, df_resid)
                # Store the p-value
                f_statistics[p_j] =f_statistic
                pval_array[p_j] = pval 
    else:
        for q_i in range(q_dim):
            # Identify columns with NaN values
            R_column = np.expand_dims(Rin, axis=1) if Rin.ndim==1 else Rin
            valid_indices = np.all(~np.isnan(R_column), axis=1)

            for p_j in range(p_dim):
                D_column =  np.expand_dims(Din[:, p_j], axis=1)
                reg_pinv_column = np.expand_dims(reg_pinv[p_j,:], axis=1) #np.expand_dims(reg_pinv, axis=1) if reg_pinv.ndim==1 else reg_pinv
                
                # reg_pinv[:,train_indices] @ Rin[train_indices, :]
                # np.dot(D_column, beta_perm))
                # Calculate beta coefficients using regularized pseudo-inverse of D_data
                beta = reg_pinv_column[valid_indices].T @ R_column[valid_indices] if reg_pinv_column.shape[-1]==1 else  reg_pinv_column[valid_indices] @ R_column[valid_indices]  # Calculate regression_coefficients (beta)
                # Predicted R
                R_pred = D_column @ beta
                # Calculate the residual sum of squares (rss)
                rss = np.sum((np.expand_dims(R_column[valid_indices,q_i],axis=1)-R_pred)**2, axis=0)
                # Calculate the total sum of squares (tss)
                tss = np.sum((R_column[valid_indices,q_i] - np.nanmean(R_column[valid_indices,q_i], axis=0))**2, axis=0)
                ess = tss - rss
                # Calculate the degrees of freedom for the model and residuals
                df_model = D_column.shape[1]  # Number of predictors including intercept
                df_resid = D_column.shape[0] - df_model
                # Calculate the mean squared error (MSE) for the model and residuals
                MSE_model = ess / df_model
                MSE_resid = rss / df_resid
                # Calculate the F-statistic
                f_statistic = MSE_model / MSE_resid
                # Calculate the p-value for the F-statistic
                pval = 1 - f.cdf(f_statistic, df_model, df_resid)
                # Store the p-value
                pval_array[p_j,] = pval       
                f_statistics[p_j] =f_statistic  
        
    return f_statistics, pval_array

def detect_significant_intervals(pval, alpha):
    """
    Detect intervals of consecutive True values in a boolean array.

    Parameters:
    ------------
    p_values (numpy.ndarray): 
        An array of p-values. 
    alpha (float, optional): 
        Threshold for significance.

    Returns:
    ----------  
    intervals (list of tuple): 
        A list of tuples representing the start and end indices
        (inclusive) of each interval of consecutive True values.

    Example:
    ----------  
        array = [False, False, False, True, True, True, False, False, True, True, False]
        detect_intervals(array)
        output: [(3, 5), (8, 9)]
    """
    # Boolean array of p-values
    array = pval<alpha
    intervals = []  # List to store intervals
    start_index = None  # Variable to track the start index of each interval

    # Iterate through the array
    for i, value in enumerate(array):
        if value:
            # If True, check if it's the start of a new interval
            if start_index is None:
                start_index = i
        else:
            # If False, check if the end of an interval is reached
            if start_index is not None:
                intervals.append((start_index, i - 1))  # Store the interval
                start_index = None  # Reset start index for the next interval

    # Handle the case where the last interval extends to the end of the array
    if start_index is not None:
        intervals.append((start_index, len(array) - 1))

    return intervals

def vpath_check_2D(vpath):
    """
    Validate whether a 2D matrix of the Viterbi path is one-hot encoded or 
    if a 1D array contains integers.

    Parameters:
    ------------
    vpath (numpy.ndarray): 
        A numpy array that can be either 2D or 1D.

    Returns:
    ----------
    bool: 
        Returns True if the following conditions are met:
        - For a 2D array: The array is one-hot encoded, meaning each row contains exactly one '1' 
          (or 1.0) and all other elements are '0' (or 0.0).
        - For a 1D array: The array contains only integer values.
        Returns False if any of these conditions are not satisfied.

    Example:
    ----------
        # Example 1: 2D one-hot encoded array
        matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        vpath_check_2D(matrix)
        output: True

        # Example 2: 1D integer array
        array = np.array([1, 2, 1, 0])
        vpath_check_2D(array)
        output: True
    """
    
    if vpath.ndim==2:
        # Check that each element is either 0 or 1 (or 0.0 or 1.0)
        if not np.all(np.isin(vpath, [0, 1])):
            return False
        
        # Check that each row has exactly one 1
        row_sums = np.sum(vpath, axis=1)
        if not np.all(row_sums == 1):
            return False
    else:
        # now vpath is a 1D array
        if np.issubdtype(vpath.dtype, np.integer) == False:
            return False
    
    return True
def __palm_quickperms(EB, M=None, nP=1000, CMC=False, EE=True):
    # Call palm_quickperms from palm_functions
    return palm_quickperms(EB, M, nP, CMC, EE)
