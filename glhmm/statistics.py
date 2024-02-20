"""
Permutation testing from Gaussian Linear Hidden Markov Model
@author: Nick Y. Larsen 2023
"""

import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from glhmm.palm_functions import *
from statsmodels.stats import multitest as smt
from sklearn.cross_decomposition import CCA
from skimage.measure import label, regionprops
from scipy.stats import ttest_ind, f_oneway, pearsonr, f, norm


def test_across_subjects(D_data, R_data, method="regression", Nperm=0, confounds = None, dict_family = None, test_statistics_option=False, FWER_correction=False, identify_categories=False, category_lim=10, test_combination=False):
    """
    This function performs statistical tests between a independent variable (`D_data`) and the dependent-variable (`R_data`) using permutation testing.
    The permutation testing is performed across across different subjects and it is possible to take family structure into account.
    This procedure is particularly valuable for investigating the differences between subjects in one's study. 
    
    Three options are available to customize the statistical analysis to a particular research questions:
        - "regression": Perform permutation testing using regression analysis.
        - "univariate": Conduct permutation testing with correlation analysis.
        - "cca": Apply permutation testing using canonical correlation analysis.
        
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
                            "regression", "univariate", or "cca". (default: "regression").      
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
    test_statistics_option (bool, optional): 
                            If True, the function will return the test statistics for each permutation.
                            (default: False) 
    FWER_correction (bool, optional): 
                            Specify whether to perform family-wise error rate (FWER) correction using the MaxT method (default: False)                   
    identify_categories : bool or list or numpy.ndarray, optional, default=True
                            If True, automatically identify categorical columns. If list or ndarray, use the provided list of column indices.    
    category_lim : int or None, optional, default=10
                            Maximum allowed number of categories for F-test. Acts as a safety measure for columns 
                            with integer values, like age, which may be mistakenly identified as multiple categories.        
    test_combination:       Calculates geometric means of p-values using permutation testing (default: False). Valid options are:
                                - True (bool): Return a single geometric mean per time point.
                                - "rows" (str): Calculate geometric means for each row.
                                - "columns" (str): Calculate geometric means for each column.
                                
                            
                         
    Returns:
    ----------  
    result (dict): A dictionary containing the following keys. Depending on the `test_statistics_option` and `method`, it can return the p-values, 
        correlation coefficients, test statisticss.
        'pval': P-values for the test with shapes based on the method:
            - method=="Regression": (T, p)
            - method=="univariate": (T, p, q)
            - method=="cca": (T, 1)
        'test_statistics': test statistics is the permutation distribution if `test_statistics_option` is True, else None.
            - method=="Regression": (T, Nperm, p)
            - method=="univariate": (T, Nperm, p, q)
            - method=="cca": (T, Nperm, 1)
        'base_statistics': Correlation coefficients for the test with shape (T, p, q) if method=="univariate", else None.
        'test_type': the type of test, which is the name of the function
        'method': the method used for analysis Valid options are
                "regression", "univariate", or "cca", "one_vs_rest" and "state_pairs" (default: "regression").
        'max_correction': Specifies if FWER has been applied using MaxT, can either output True or False.  
        'performed_tests': A dictionary that marks the columns in the test_statistics or p-value matrix corresponding to the (q dimension) where t-tests or F-tests have been performed.
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
    valid_methods = ["regression", "univariate", "cca"]
    validate_condition(method in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    # Check validity of method
    valid_test_combination = [False, True, "columns", "rows"]
    validate_condition(test_combination in valid_test_combination, "Invalid option specified for 'test_combination'. Must be one of: " + ', '.join(map(str, valid_test_combination)))
    
    if method=="regression" and test_combination in valid_test_combination[-1:]:
            raise ValueError("method is set to 'regression' and 'test_combination' is set to 'rows' "
                         "If you want to perform 'test_combination' while doing 'regression' then please set 'test_combination' to 'True' or 'columns'.")

    if FWER_correction and test_combination in [True, "columns", "rows"]:
       # Raise an exception and stop function execution
        raise ValueError("'FWER_correction' is set to True and 'test_combination' is either True, 'columns', or 'rows'. "
                         "Please set 'FWER_correction' to False if you want to apply 'test_combination' or set 'test_combination' to False if you want to run 'FWER_correction'.")
    # Get the shapes of the data
    n_T, _, n_p, n_q, D_data, R_data = get_input_shape(D_data, R_data)
    # Note for convension we wrote (T, p, q) => (n_T, n_p, n_q)
    
    # Identify categorical columns in R_data
    category_columns = identify_coloumns_for_t_and_f_tests(R_data, method, identify_categories, category_lim) if method=="univariate" or method =="regression" else {'t_test_cols': [], 'f_test_cols': []}

    if category_columns["t_test_cols"]!=[] or category_columns["f_test_cols"]!=[]:
        if FWER_correction and (len(category_columns.get('t_test_cols')) != R_data.shape[-1] or len(category_columns.get('f_test_cols')) != R_data.shape[-1]):
            print("Warning: Cannot perform FWER_correction with different test statisticss.\nConsider to set identify_categories=False")
            raise ValueError("Cannot perform FWER_correction")
    
    # Crate the family structure by looking at the dictionary 
    if dict_family is not None:
        # process dictionary of family structure
        dict_mfam=process_family_structure(dict_family, Nperm) 
        
    
    # Initialize arrays based on shape of data shape and defined options
    pval, base_statistics, test_statistics_list = initialize_arrays(R_data, n_p, n_q, n_T, method, Nperm, test_statistics_option, test_combination)
    

    for t in tqdm(range(n_T)) if n_T > 1 else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        D_t, R_t = deconfound_values(D_data[t, :],R_data[t, :], confounds)
        
        # Removing rows that contain nan-values
        if method == "regression" or method == "cca":
            # Removing rows that contain nan-values
            D_t, R_t = remove_nan_values(D_t, R_t, t, n_T, method)
        
        # Create test_statistics based on method
        test_statistics, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, D_t, test_combination)

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
            test_statistics, bstat = test_statistics_calculations(D_t, Rin, perm, test_statistics, proj, method, category_columns,test_combination)
            base_statistics[t, :] = bstat if perm == 0 and bstat is not None else base_statistics[t, :]

        # Calculate p-values
        pval = get_pval(test_statistics, Nperm, method, t, pval, FWER_correction, test_combination) if Nperm>1 else 0
        
        # Output test statistics if it is set to True can be hard for memory otherwise
        if test_statistics_option==True:
            test_statistics_list[t,:] = test_statistics
            
    pval =np.squeeze(pval) if np.abs(np.nansum(pval))>0 else np.nan 
    base_statistics =np.squeeze(base_statistics) if base_statistics is not None else [] 
    test_statistics_list =np.squeeze(test_statistics_list) if test_statistics_list is not None else []
    Nperm = 0 if Nperm==1 else Nperm
    
    if np.sum(np.isnan(pval))>0:
        print("Warning: Permutation testing resulted in p-values equal to NaN.")
        print("This may indicate an issue with the input data. Please review your data.")
    
    # Return results
    result = {
        'pval': pval,
        'base_statistics': base_statistics,
        'test_statistics': test_statistics_list,
        'test_type': 'test_across_subjects',
        'method': method,
        'test_combination': test_combination,
        'max_correction':FWER_correction,
        'performed_tests': category_columns,
        'Nperm': Nperm}
    return result



def test_across_trials_within_session(D_data, R_data, idx_data, method="regression", Nperm=0, confounds=None, trial_timepoints=None,test_statistics_option=False, FWER_correction=False, identify_categories=False, category_lim=10, test_combination=False):
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
                                "regression", "univariate", or "cca". (default: "regression").
                                Note: "cca" stands for Canonical Correlation Analysis    
        Nperm (int): Number of permutations to perform (default: 1000). 
        confounds (numpy.ndarray or None, optional): 
                                The confounding variables to be regressed out from the input data (D_data).
                                If provided, the regression analysis is performed to remove the confounding effects. 
                                (default: None):    
        trial_timepoints (int): Number of timepoints for each trial (default: None)                                                          
        test_statistics_option (bool, optional): 
                                If True, the function will return the test statistics for each permutation.
                                (default: False) 
        FWER_correction (bool, optional): 
                                Specify whether to perform family-wise error rate (FWER) correction for multiple comparisons using the MaxT method(default: False).                        
        identify_categories : bool or list or numpy.ndarray, optional, default=True
                                If True, automatically identify categorical columns. If list or ndarray, use the provided list of column indices.    
        category_lim : int or None, optional, default=None
                                Maximum allowed number of categories for F-test. Acts as a safety measure for columns 
                                with integer values, like age, which may be mistakenly identified as multiple categories.     
        test_combination:       Calculates geometric means of p-values using permutation testing (default: False). Valid options are:
                                - True (bool): Return a single geometric mean per time point.
                                - "rows" (str): Calculate geometric means for each row.
                                - "columns" (str): Calculate geometric means for each column.                                                
    Returns:
    ----------  
    result (dict): A dictionary containing the following keys. Depending on the `test_statistics_option` and `method`, it can return the p-values, 
        correlation coefficients, test statisticss.
        'pval': P-values for the test with shapes based on the method:
            - method=="Regression": (T, p)
            - method=="univariate": (T, p, q)
            - method=="cca": (T, 1)
        'test_statistics': test statistics is the permutation distribution if `test_statistics_option` is True, else None.
            - method=="Regression": (T, Nperm, p)
            - method=="univariate": (T, Nperm, p, q)
            - method=="cca": (T, Nperm, 1)
        'base_statistics': Correlation coefficients for the test with shape (T, p, q) if method=="univariate", else None.
        'test_type': the type of test, which is the name of the function
        'method': the method used for analysis Valid options are
                "regression", "univariate", or "cca", "one_vs_rest" and "state_pairs" (default: "regression").
        'max_correction': Specifies if FWER has been applied using MaxT, can either output True or False.
        'Nperm' :The number of permutations that has been performed.   

    Note:
        The function automatically determines whether permutation testing is performed per timepoint for each subject or
        for the whole data based on the dimensionality of `D_data`.
        The function assumes that the number of rows in `D_data` and `R_data` are equal
    """
    # Initialize variable
    category_columns = []    
    # Have to run the permutation test function 1 time at least once
    if Nperm==0:
        Nperm+=1 
        
          
    # Check validity of method and data_type
    valid_methods = ["regression", "univariate", "cca"]
    validate_condition(method in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    # Check validity of method
    valid_test_combination = [False, True, "columns", "rows"]
    validate_condition(test_combination in valid_test_combination, "Invalid option specified for 'test_combination'. Must be one of: " + ', '.join(map(str, valid_test_combination)))
    
    if method=="regression" and test_combination in valid_test_combination[-1:]:
            raise ValueError("method is set to 'regression' and 'test_combination' is set to 'rows' "
                         "If you want to perform 'test_combination' while doing 'regression' then please set 'test_combination' to 'True' or 'columns'.")

    if FWER_correction and test_combination in [True, "columns", "rows"]:
       # Raise an exception and stop function execution
        raise ValueError("'FWER_correction' is set to True and 'test_combination' is either True, 'columns', or 'rows'. "
                         "Please set 'FWER_correction' to False if you want to apply 'test_combination' or set 'test_combination' to False if you want to run 'FWER_correction'.")

    # Get input shape information
    n_T, _, n_p, n_q, D_data, R_data = get_input_shape(D_data, R_data)

    # Identify categorical columns in R_data
    category_columns = identify_coloumns_for_t_and_f_tests(R_data, method, identify_categories, category_lim) if method=="univariate" or method =="regression" else {'t_test_cols': [], 'f_test_cols': []}
    
    if category_columns["t_test_cols"]!=[] or category_columns["f_test_cols"]!=[]:
        if FWER_correction and (len(category_columns.get('t_test_cols')) != R_data.shape[-1] or len(category_columns.get('f_test_cols')) != R_data.shape[-1]):
            print("Warning: Cannot perform FWER_correction with different test statisticss.\nConsider to set identify_categories=False")
            raise ValueError("Cannot perform FWER_correction")
    
    # Get indices for permutation
    if len(idx_data.shape)==2:
        idx_array = get_indices_array(idx_data)
    else:
        idx_array =idx_data.copy()        

    # Initialize arrays based on shape of data shape and defined options
    pval, base_statistics, test_statistics_list = initialize_arrays(R_data, n_p, n_q, n_T, method, Nperm, test_statistics_option, test_combination)

    for t in tqdm(range(n_T)) if n_T > 1 else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        D_t, R_t = deconfound_values(D_data[t, :],R_data[t, :], confounds)
        
        # Removing rows that contain nan-values
        if method == "regression" or method == "cca":
            D_t, R_t = remove_nan_values(D_t, R_t, t, n_T, method)
        
        # Create test_statistics and pval_perms based on method
        test_statistics, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, D_t, test_combination)
    

        # Calculate permutation matrix of D_t 
        permutation_matrix = permutation_matrix_across_trials_within_session(Nperm,R_t, idx_array,trial_timepoints)
                
        for perm in range(Nperm):
        #for perm in tqdm(range(Nperm)) if n_T == 1 else range(n_T):
            # Perform permutation on R_t
            Rin = R_t[permutation_matrix[:, perm]]
            # Calculate the permutation distribution
            test_statistics, bstat = test_statistics_calculations(D_t, Rin, perm, test_statistics, proj, method, category_columns,test_combination)
            base_statistics[t, :] = bstat if perm == 0 and bstat is not None else base_statistics[t, :]

        # Calculate p-values
        pval = get_pval(test_statistics, Nperm, method, t, pval, FWER_correction, test_combination) if Nperm>1 else 0
        if test_statistics_option==True:
            test_statistics_list[t,:] = test_statistics
    pval =np.squeeze(pval) if np.abs(np.nansum(pval))>0 else np.nan
    base_statistics =np.squeeze(base_statistics) if base_statistics is not None  else []
    test_statistics_list =np.squeeze(test_statistics_list) if test_statistics_list is not None else []
    Nperm = 0 if Nperm==1 else Nperm
    
    if np.sum(np.isnan(pval))>0:
        print("Warning: Permutation testing resulted in p-values equal to NaN.")
        print("This may indicate an issue with the input data. Please review your data.")
        
    # Return results
    result = {
        'pval': pval,
        'base_statistics': base_statistics,
        'test_statistics': test_statistics_list,
        'test_type': 'test_across_subjects',
        'method': method,
        'test_combination': test_combination,
        'max_correction':FWER_correction,
        'performed_tests': category_columns,
        'Nperm': Nperm}
    
    return result

def test_across_sessions_within_subject(D_data, R_data, idx_data, method="regression", Nperm=0, confounds=None,test_statistics_option=False,FWER_correction=False, identify_categories=False, category_lim=10, test_combination=False):
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
                                "regression", "univariate", or "cca". (default: "regression").
                                Note: "cca" stands for Canonical Correlation Analysis    
        Nperm (int): Number of permutations to perform (default: 1000).                
        confounds (numpy.ndarray or None, optional): 
                                The confounding variables to be regressed out from the input data (D_data).
                                If provided, the regression analysis is performed to remove the confounding effects. 
                                (default: None):                                                              
        test_statistics_option (bool, optional): 
                                If True, the function will return the test statistics for each permutation.
                                (default: False) 
        FWER_correction (bool, optional): 
                                Specify whether to perform family-wise error rate (FWER) correction for multiple comparisons using the MaxT method(default: False).
        identify_categories : bool or list or numpy.ndarray, optional, default=True
                                If True, automatically identify categorical columns. If list or ndarray, use the provided list of column indices.    
        category_lim : int or None, optional, default=None
                                Maximum allowed number of categories for F-test. Acts as a safety measure for columns 
                                with integer values, like age, which may be mistakenly identified as multiple categories.
        test_combination:       Calculates geometric means of p-values using permutation testing (default: False). Valid options are:
                                - True (bool): Return a single geometric mean per time point.
                                - "rows" (str): Calculate geometric means for each row.
                                - "columns" (str): Calculate geometric means for each column.                         
    Returns:
    ----------  
        result (dict): A dictionary containing the following keys. Depending on the `test_statistics_option` and `method`, it can return the p-values, 
            correlation coefficients, test statisticss.
            'pval': P-values for the test with shapes based on the method:
                - method=="Regression": (T, p)
                - method=="univariate": (T, p, q)
                - method=="cca": (T, 1)
            'test_statistics': test statistics is the permutation distribution if `test_statistics_option` is True, else None.
                - method=="Regression": (T, Nperm, p)
                - method=="univariate": (T, Nperm, p, q)
                - method=="cca": (T, Nperm, 1)
            'base_statistics': Correlation coefficients for the test with shape (T, p, q) if method=="univariate", else None.
            'test_type': the type of test, which is the name of the function
            'method': the method used for analysis Valid options are
                    "regression", "univariate", or "cca", "one_vs_rest" and "state_pairs" (default: "regression").
            'max_correction': Specifies if FWER has been applied using MaxT, can either output True or False.
            'Nperm' :The number of permutations that has been performed.
                  
    Note:
        The function automatically determines whether permutation testing is performed per timepoint for each subject or
        for the whole data based on the dimensionality of `D_data`.
        The function assumes that the number of rows in `D_data` and `R_data` are equal

    """ 
    # Initialize variable
    category_columns = []    
    # Have to run the permutation test function 1 time at least once
    if Nperm==0:
        Nperm+=1
     
    # Check validity of method and data_type
    valid_methods = ["regression", "univariate", "cca"]
    validate_condition(method in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    # Check validity of method
    valid_test_combination = [False, True, "columns", "rows"]
    validate_condition(test_combination in valid_test_combination, "Invalid option specified for 'test_combination'. Must be one of: " + ', '.join(map(str, valid_test_combination)))
    
    if method=="regression" and test_combination in valid_test_combination[-1:]:
            raise ValueError("method is set to 'regression' and 'test_combination' is set to 'rows' "
                         "If you want to perform 'test_combination' while doing 'regression' then please set 'test_combination' to 'True' or 'columns'.")

    if FWER_correction and test_combination in [True, "columns", "rows"]:
       # Raise an exception and stop function execution
        raise ValueError("'FWER_correction' is set to True and 'test_combination' is either True, 'columns', or 'rows'. "
                         "Please set 'FWER_correction' to False if you want to apply 'test_combination' or set 'test_combination' to False if you want to run 'FWER_correction'.")
    
    # Get indices for permutation
    if len(idx_data.shape)==2:
        idx_array = get_indices_array(idx_data)
    else:
        idx_array =idx_data.copy()

    # Get input shape information
    n_T, _, n_p, n_q, D_data, R_data = get_input_shape(D_data, R_data)
    
    # Identify categorical columns in R_data
    category_columns = identify_coloumns_for_t_and_f_tests(R_data, method, identify_categories, category_lim) if method=="univariate" or method =="regression" else {'t_test_cols': [], 'f_test_cols': []}
    
    if category_columns["t_test_cols"]!=[] or category_columns["f_test_cols"]!=[]:
        if FWER_correction and (len(category_columns.get('t_test_cols')) != R_data.shape[-1] or len(category_columns.get('f_test_cols')) != R_data.shape[-1]):
            print("Warning: Cannot perform FWER_correction with different test statisticss.\nConsider to set identify_categories=False")
            raise ValueError("Cannot perform FWER_correction")

# Initialize arrays based on shape of data shape and defined options
    pval, base_statistics, test_statistics_list = initialize_arrays(R_data, n_p, n_q, n_T, method, Nperm, test_statistics_option, test_combination)
    for t in tqdm(range(n_T)) if n_T > 1 else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        D_t, R_t = deconfound_values(D_data[t, :],R_data[t, :], confounds)
        
        # Removing rows that contain nan-values
        if method == "regression" or method == "cca":
            D_t, R_t = remove_nan_values(D_t, R_t, t, n_T, method)

        # Create test_statistics and pval_perms based on method
        test_statistics, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, D_t, test_combination)

        # Calculate permutation matrix of D_t 
        permutation_matrix = permutation_matrix_within_subject_across_sessions(Nperm, D_t, idx_array)
        
        for perm in range(Nperm):
        #for perm in tqdm(range(Nperm)) if n_T == 1 else range(n_T):
            # Perform permutation on R_t
            Rin = R_t[permutation_matrix[:, perm]]
            # Calculate the permutation distribution
            test_statistics, bstat = test_statistics_calculations(D_t, Rin, perm, test_statistics, proj, method, category_columns,test_combination)
            base_statistics[t, :] = bstat if perm == 0 and bstat is not None else base_statistics[t, :]

        # Caluclate p-values
        pval = get_pval(test_statistics, Nperm, method, t, pval, FWER_correction, test_combination) if Nperm>1 else 00
        if test_statistics_option==True:
            test_statistics_list[t,:] = test_statistics
            
    pval =np.squeeze(pval) if np.abs(np.nansum(pval))>0 else np.nan 
    base_statistics =np.squeeze(base_statistics) if base_statistics is not None  else []
    test_statistics_list =np.squeeze(test_statistics_list) if test_statistics_list is not None  else []
    Nperm = 0 if Nperm==1 else Nperm    
    if np.sum(np.isnan(pval))>0:
        print("Warning: Permutation testing resulted in p-values equal to NaN.")
        print("This may indicate an issue with the input data. Please review your data.")
              
    # Return results
    result = {
        'pval': pval,
        'base_statistics': base_statistics,
        'test_statistics': test_statistics_list,
        'test_type': 'test_across_subjects',
        'method': method,
        'test_combination': test_combination,
        'max_correction':FWER_correction,
        'performed_tests': category_columns,
        'Nperm': Nperm}
    return result

def test_across_visits(input_data, vpath_data, n_states, method="regression", Nperm=0, confounds=None, test_statistics_option=False, pairwise_statistic ="mean",FWER_correction=False, category_lim=None, identify_categories = False):
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
                                    "regression", "univariate", "cca", "one_vs_rest" or "state_pairs". 
                                    Note: "cca" stands for Canonical Correlation Analysis.   
        Nperm (int):                Number of permutations to perform (default: 0). 
        test_statistics_option (bool, optional): 
                                    If True, the function will return the test statistics for each permutation.
                                    (default: False) 
        pairwise_statistic (str, optional)  
                                    The chosen statistic when applying methods "one_vs_rest" or "state_pairs". 
                                    Valid options are "mean" or "median" (default: "mean").
        FWER_correction (bool, optional): 
                                    Specify whether to perform family-wise error rate (FWER) correction for multiple comparisons using the MaxT method(default: False).
                     
    Returns:
    ----------  
        result (dict): A dictionary containing the following keys. Depending on the `test_statistics_option` and `method`, it can return the p-values, 
            correlation coefficients, test statisticss.
            'pval': P-values for the test with shapes based on the method:
                - method=="Regression": (T, p)
                - method=="univariate": (T, p, q)
                - method=="cca": (T, 1)
            'test_statistics': test statistics is the permutation distribution if `test_statistics_option` is True, else None.
                - method=="Regression": (T, Nperm, p)
                - method=="univariate": (T, Nperm, p, q)
                - method=="cca": (T, Nperm, 1)
            'base_statistics': Correlation coefficients for the test with shape (T, p, q) if method=="univariate", else None.
            'test_type': the type of test, which is the name of the function
            'method': the method used for analysis Valid options are
                    "regression", "univariate", or "cca", "one_vs_rest" and "state_pairs" (default: "regression").
            'max_correction': Specifies if FWER has been applied using MaxT, can either output True or False.
            'Nperm' :The number of permutations that has been performed.
                
    Note:
        The function assumes that the number of rows in `vpath_data` and `Y_data` are equal
    """
    # Have to run the permutation test function 1 time at least once
    if Nperm==0:
        Nperm+=1
            
    # Check validity of method
    valid_methods = ["regression", "univariate", "cca", "one_vs_rest", "state_pairs"]
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

    # Identify categorical columns in R_data
    category_columns = identify_coloumns_for_t_and_f_tests(vpath_data, method, identify_categories, category_lim) if method=="univariate" or method =="regression" else {'t_test_cols': [], 'f_test_cols': []}
    
    # Identify categorical columns
    if category_columns["t_test_cols"]!=[] or category_columns["f_test_cols"]!=[]:
        if FWER_correction and (len(category_columns.get('t_test_cols')) != vpath_data.shape[-1] or len(category_columns.get('f_test_cols')) != R_data.shape[-1]):
            print("Warning: Cannot perform FWER_correction with different test statisticss.\nConsider to set identify_categories=False")
            raise ValueError("Cannot perform FWER_correction")    
   
    # Initialize arrays based on shape of data shape and defined options
    pval, base_statistics, test_statistics_list = initialize_arrays(vpath_data, n_p, n_q,
                                                                            n_T, method, Nperm,
                                                                            test_statistics_option)


    # Print tqdm over n_T if there are more than one timepoint
    for t in tqdm(range(n_T)) if n_T > 1 else range(n_T):
        # Correct for confounds and center data_t
        data_t, _ = deconfound_values(input_data[t, :],None, confounds)
        
        # Removing rows that contain nan-values
        if method == "regression" or method == "cca":
            data_t, vpath_array = remove_nan_values(data_t, vpath_array, t, n_T, method)
        
        if method != "state_pairs":
            ###################### Permutation testing for other tests beside state pairs #################################
            # Create test_statistics and pval_perms based on method
            test_statistics, proj = initialize_permutation_matrices(method, Nperm, n_p, n_q, 
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
                        test_statistics[perm,state] =calculate_baseline_difference(vpath_surrogate, data_t, state+1, pairwise_statistic.lower())
                elif method =="regression":
                    test_statistics, bstat = test_statistics_calculations(data_t,vpath_surrogate , perm,
                                                                            test_statistics, proj, method, category_columns)
                    base_statistics[t, :] = bstat if perm == 0 and bstat is not None else base_statistics[t, :]
                else:
                    # Apply 1 hot encoding
                    vpath_surrogate_onehot = viterbi_path_to_stc(vpath_surrogate,n_states)
                    # Apply t-statistic on the vpath_surrogate
                    test_statistics, bstat = test_statistics_calculations(data_t, vpath_surrogate_onehot, perm, 
                                                                          test_statistics, proj, method, category_columns)
                    base_statistics[t, :] = bstat if perm == 0 and bstat is not None else base_statistics[t, :]

            pval = get_pval(test_statistics, Nperm, method, t, pval, FWER_correction) if Nperm>1 else 0
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
            for idx, (state_1, state_2) in tqdm(enumerate(pairwise_comparisons), total=len(pairwise_comparisons), desc="Pairwise comparisons"):    
                # Generate surrogate state-time data and calculate differences for each permutation
                for perm in range(Nperm):
                    # Redo vpath_surrogate calculation if the number of states are not the same (out of 1000 permutations it happens maybe 1-2 times with this demo dataset)
                    while True:
                        # Create vpath_surrogate
                        vpath_surrogate = surrogate_state_time(perm, vpath_array, n_states)
                        if len(np.unique(vpath_surrogate)) == n_states:
                            break  # Exit the loop if the condition is satisfied
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

    pval =np.squeeze(pval) if np.abs(np.nansum(pval))>0 else np.nan 
    base_statistics =np.squeeze(base_statistics) if base_statistics is not None else []
    test_statistics_list =np.squeeze(test_statistics_list) if test_statistics_list is not None else []
    Nperm = 0 if Nperm==1 else Nperm
    
    if np.sum(np.isnan(pval))>0:
        print("Warning: Permutation testing resulted in p-values equal to NaN.")
        print("This may indicate an issue with the input data. Please review your data.")
        
    # Return results
    result = {
        'pval': pval,
        'base_statistics': base_statistics,
        'test_statistics': test_statistics_list,
        'test_type': 'test_across_subjects',
        'method': method,
        'max_correction':FWER_correction,
        'performed_tests': category_columns,
        'Nperm': Nperm}
    return result

def remove_nan_values(D_data, R_data, t, n_T, method):
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
    removed_indices = None
    
    if R_data.ndim == 1:
        FLAG = 1
        R_data = R_data.reshape(-1,1) 
    if method == "regression":
        # When applying "regression" we need to remove rows for our D_data, as we cannot use it as a predictor for
        # Check for NaN values and remove corresponding rows
        nan_mask = np.isnan(D_data).any(axis=1)
        # nan_mask = np.isnan(D_data).any(axis=1)
        # Get indices or rows that have been removed
        removed_indices = np.where(nan_mask)[0]

        D_data = D_data[~nan_mask]
        R_data = R_data[~nan_mask]
    elif method== "cca":
        # When applying cca we need to remove rows at both D_data and R_data
        # Check for NaN values and remove corresponding rows
        nan_mask = np.isnan(D_data).any(axis=1) | np.isnan(R_data).any(axis=1)
        # nan_mask = np.isnan(D_data).any(axis=1)
        # Get indices or rows that have been removed
        removed_indices = np.where(nan_mask)[0]

        D_data = D_data[~nan_mask]
        R_data = R_data[~nan_mask]

    # Only print this 1 time
    # Check if the array is empty
    # if n_T==1 and np.any(removed_indices): 
    #     print("Rows with NaN values have been removed:")
    #     print("Removed Indices:", removed_indices)
    # # Check if the array is empty
    # elif np.any(removed_indices):
    #     print(f"Rows with NaN values have been removed at timepoint {t}:")
    #     print("Removed Indices:", removed_indices)
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

def initialize_arrays(R_data, n_p, n_q, n_T, method, Nperm, test_statistics_option, test_combination=False):
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
        test_statistics_option (bool): If True, return the test statistics values.

    Returns:
    ----------  
        pval (numpy array): p-values for the test (n_T, n_p) if test_statistics_option is False, else None.
        base_statistics (numpy array): Correlation coefficient for the test (n_T, n_p, n_q) if method="univariate", else None.
        test_statistics_list (numpy array): test statistics values (n_T, Nperm, n_p) or (n_T, Nperm, n_p, n_q) if method="univariate" , else None.
    """
    
    # Initialize the arrays based on the selected method and data dimensions
    if  method == "regression":
        if test_combination in [True, "columns", "rows"]: 
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
        if test_combination in [True, "columns", "rows"]: 
            pval_shape = (n_T, 1) if test_combination == True else (n_T, n_q) if test_combination == "columns" else (n_T, n_p)
            pval = np.zeros(pval_shape)
            base_statistics = pval.copy()
            if test_statistics_option:
                test_statistics_list_shape = (n_T, Nperm, 1) if test_combination == True else (n_T, Nperm, n_q) if test_combination == "columns" else (n_T, Nperm, n_p)
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
        pval = np.zeros((n_T, n_p, n_q))
        if test_statistics_option==True:
            test_statistics_list = np.zeros((n_T, Nperm, n_q))
        else:
            test_statistics_list= None
        base_statistics= np.zeros((n_T, 1, n_q))

    return pval, base_statistics, test_statistics_list


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
        # Centering D_data
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

def initialize_permutation_matrices(method, Nperm, n_p, n_q, D_data, test_combination=False):
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
        test_statistics (numpy.ndarray): The permutation array.
        pval_perms (numpy.ndarray): The p-value permutation array.
        proj (numpy.ndarray or None): The projection matrix (None for correlation methods).
    """
    # Define projection matrix
    proj = None
    # Initialize the permutation matrices based on the selected method
    if method in {"univariate"}:
        if test_combination in [True, "columns", "rows"]: 
            test_statistics_shape = (Nperm, 1) if test_combination == True else (Nperm, n_q) if test_combination == "columns" else (Nperm, n_p)
            test_statistics = np.zeros(test_statistics_shape)
        else:
            # Initialize test statistics output matrix based on the selected method
            test_statistics = np.zeros((Nperm, n_p, n_q))
        proj = None
    elif method =="cca":
        # Initialize test statistics output matrix based on the selected method
        test_statistics = np.zeros((Nperm, 1))
    else:
        if test_combination in [True, "columns", "rows"]:
            test_statistics = np.zeros((Nperm, 1))
        else:
            # Regression got a N by q matrix 
            test_statistics = np.zeros((Nperm, n_q))
        # Define regularization parameter
        regularization = 0.001
        # Regularized parameter estimation
        regularization_matrix = regularization * np.eye(D_data.shape[1])  # Regularization term for Ridge regression
        
        # Fit the Ridge regression model
        # The projection matrix is then used to project permuted data matrix (Din) to obtain the regression coefficients (beta)
        proj = np.linalg.inv(D_data.T @ D_data + regularization_matrix) @ D_data.T  # Projection matrix for Ridge regression
    return test_statistics, proj

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

def get_pval(test_statistics, Nperm, method, t, pval, FWER_correction=False, test_combination=False):
    """
    Computes p-values and correlation matrix for permutation testing.

    Parameters:
    --------------
        test_statistics (numpy.ndarray): The permutation array.
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
            nCols = test_statistics[0,:].shape[-1]
            nRows = len(test_statistics)
            # Get the maximum explained variance for each column
            max_test_statistics =np.tile(np.max(test_statistics, axis=1), (1, nCols)).reshape(nCols, nRows).T
            # Count every time there is a higher estimated R2 (better fit)
            pval[t, :] = np.nansum(max_test_statistics>= test_statistics[0,:], axis=0) / (Nperm + 1)
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
            # Get the maximum explained variance for each column
            max_test_statistics =np.tile(np.nanmax(test_statistics, axis=1), (1, nRows)).reshape(nCols,nRows, nPerm).T
            # Count every time there is a higher correlation coefficient
            pval[t, :] = np.nansum(max_test_statistics>= test_statistics[0,:], axis=0) / (Nperm + 1)
        else:    
            # Count every time there is a higher correlation coefficient
            pval[t, :] = np.nansum(test_statistics >= test_statistics[0,:], axis=0) / (Nperm + 1)
    
    # Convert 0 values to NaN
    pval[pval == 0] = np.nan
    
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

def test_statistics_calculations(Din, Rin, perm, test_statistics, proj, method, category_columns=[], test_combination=False):
    """
    Calculates the test_statistics array and pval_perms array based on the given data and method.

    Parameters:
    --------------
        Din (numpy.ndarray): The data array.
        Rin (numpy.ndarray): The dependent variable.
        perm (int): The permutation index.
        pval_perms (numpy.ndarray): The p-value permutation array.
        test_statistics (numpy.ndarray): The permutation array.
        proj (numpy.ndarray or None): The projection matrix (None for correlation methods).
        method (str): The method used for permutation testing.

    Returns:
    ----------  
        test_statistics (numpy.ndarray): Updated test_statistics array.
        pval_perms (numpy.ndarray): Updated pval_perms array.
    """
    if method == 'regression':
        if category_columns["t_test_cols"]==[] and category_columns["f_test_cols"]==[]:
            if np.sum(np.isnan(Rin))>0:
                if test_combination in [True, "columns"]:
                    # Calculate F-statitics with no NaN values.
                    F_statistic =calculate_nan_regression_f_test(Din, Rin, proj, nan_values=True)
                     # Calculate the degrees of freedom for the model and residuals
                    df_model = Din.shape[1]  # Number of predictors including intercept
                    df_resid = Din.shape[0] - df_model
                    p_value = 1 - f.cdf(F_statistic, df_model, df_resid)
                    # Get the base statistics and store p-values as z-scores to the test statistic
                    base_statistics = calculate_geometric_pval(p_value, test_combination)
                    test_statistics[perm] =abs(base_statistics) 
                else:
                    # Calculate the explained variance if R got NaN values.
                    base_statistics =calculate_nan_regression(Din, Rin, proj)
                    test_statistics[perm,:] =base_statistics           
            else:
                # Fit the original model 
                beta = proj @ Rin  # Calculate regression_coefficients (beta)
                # Calculate the predicted values
                predicted_values = Din @ beta
                # Calculate the residual sum of squares (rss)
                rss = np.sum((predicted_values - Rin)**2, axis=0)
                # Calculate the total sum of squares (tss)
                tss = np.sum((Rin - np.nanmean(Rin, axis=0))**2, axis=0)
                
                if test_combination in [True, "columns"]:
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
                    p_value = 1 - f.cdf(F_statistic, df_model, df_resid)
                    # Get the base statistics and store p-values as z-scores to the test statistic
                    base_statistics = calculate_geometric_pval(p_value, test_combination)
                    test_statistics[perm] =abs(base_statistics) 
                else:
                    # Calculate R^2
                    base_statistics = 1 - (rss / tss) #r_squared
                    # Store the R^2 values in the test_statistics array
                    test_statistics[perm] = base_statistics
        else:
            # If we are doing test_combinations, we need to calculate f-statistics on every column
            if test_combination in [True, "columns"]:
                if np.sum(np.isnan(Rin))>0:
                    # Calculate the explained variance if R got NaN values.
                    F_statistic =calculate_nan_regression_f_test(Din, Rin, proj, nan_values=True)
                else:
                    # Calculate F-statitics with no NaN values.
                    F_statistic =calculate_nan_regression_f_test(Din, Rin, proj, nan_values=False)
                     # Calculate the degrees of freedom for the model and residuals
                df_model = Din.shape[1]  # Number of predictors including intercept
                df_resid = Din.shape[0] - df_model
                p_value = 1 - f.cdf(F_statistic, df_model, df_resid)
                # Get the base statistics and store p-values as z-scores to the test statistic
                base_statistics = calculate_geometric_pval(p_value, test_combination)
                test_statistics[perm] =abs(base_statistics) 
            else:
            # If we are not perfomring test_combination, we need to perform a columnwise operation.  
            # We perform f-test if category_columns has flagged categorical columns otherwise it will be R^2 
                # Initialize variables  which
                #base_statistics =np.zeros_like(test_statistics[0,:]) if perm ==0 else None
                base_statistics =np.zeros_like(test_statistics[0,:])
                for col in range(Rin.shape[1]):
                    if category_columns["f_test_cols"] and col in category_columns["f_test_cols"]:
                        # Calculate f-statistics of columns of interest  
                        if np.sum(np.isnan(Rin))>0:
                            # Calculate the explained variance if R got NaN values.
                            base_statistics[col] =calculate_nan_regression_f_test(Din, Rin[:, col], proj, nan_values=True)
                        else:
                            # Calculate F-statitics with no NaN values.
                            base_statistics[col] =calculate_nan_regression_f_test(Din, Rin[:, col], proj, nan_values=False)
                        test_statistics[perm,col]  =np.abs(base_statistics[col])
                    else:
                        # Check for NaN values
                        if np.sum(np.isnan(Rin))>0:
                            # Calculate the explained variance if R got NaN values.
                            base_statistics[col] =calculate_nan_regression(Din, Rin[:, col], proj)         
                        else:
                            # Fit the original model 
                            beta = proj @ Rin[:, col]  # Calculate regression_coefficients (beta)
                            # Calculate the predicted values
                            predicted_values = Din @ beta
                            # Calculate the residual sum of squares (rss)
                            rss = np.sum((predicted_values - Rin[:, col])**2, axis=0)
                            # Calculate the total sum of squares (tss)
                            tss = np.sum((Rin[:, col] - np.nanmean(Rin[:, col], axis=0))**2, axis=0)
                            # Calculate R^2
                            base_statistics[col] = 1 - (rss / tss) #r_squared
                        # Store the R^2 values in the test_statistics array
                        test_statistics[perm,col] = base_statistics[col]        
    # Calculate for univariate tests              
    elif method == "univariate":
        if category_columns["t_test_cols"]==[] and category_columns["f_test_cols"]==[]:
            # Only calcuating the correlation matrix, since there is no need for t- or f-test
            if np.sum(np.isnan(Din))>0 or np.sum(np.isnan(Rin))>0:
                # Calculate the correlation matrix while handling NaN values 
                # column by column without removing entire rows.
                if test_combination in [True, "columns", "rows"]: 
                    # Return parametric p-values
                    _,base_statistics =calculate_nan_correlation_matrix(Din, Rin, test_combination, reduce_pval_dims=True)
                    test_statistics[perm, :] = base_statistics # Notice that shape of test_statistics are different
                else:
                    # Return correlation coefficients instead of p-values
                    base_statistics,_ =calculate_nan_correlation_matrix(Din, Rin)
                    test_statistics[perm, :, :] = np.abs(base_statistics)
            else:
                if test_combination in [True, "columns", "rows"]: 
                    # Return parametric p-values
                    pval_matrix = np.zeros((Din.shape[1],Rin.shape[1]))
                    for i in range(Din.shape[1]):
                        for j in range(Rin.shape[1]):
                            _, pval_matrix[i, j] = pearsonr(Din[:, i], Rin[:, j])
                    base_statistics =calculate_geometric_pval(pval_matrix, test_combination) 
                    test_statistics[perm,:]=abs(base_statistics)         
                else:
                    # Calculate correlation coeffcients without NaN values
                    corr_coef = np.corrcoef(Din, Rin, rowvar=False)
                    corr_matrix = corr_coef[:Din.shape[1], Din.shape[1]:]
                    base_statistics = corr_matrix
                    test_statistics[perm, :, :] = np.abs(base_statistics)
        else: 
            # Calculate t-, f- and correlation statistics per column
            base_statistics =np.zeros_like(test_statistics[0,:]) if perm ==0 else None
            pval_statistics_com = np.zeros((Din.shape[-1],Rin.shape[-1]))
            base_statistics_com = np.zeros((Din.shape[-1],Rin.shape[-1]))
            for col in range(Rin.shape[1]):
                if category_columns["t_test_cols"] and col in category_columns["t_test_cols"]:
                    # t-test for each column
                    if np.sum(np.isnan(Din))>0 or np.sum(np.isnan(Din))>0:
                        # Get the t-statistic if when NaN values are detected
                        t_test, pval =calculate_nan_t_test(Din, Rin[:, col], nan_values=True)
                    else:
                        # Get the t-statistic if there are no NaN values
                        t_test, pval = calculate_nan_t_test(Din, Rin[:, col], nan_values=False)
                        
                    # Put values inside test_statistics    
                    if test_combination in [True, "columns", "rows"]:
                        # Save to pval_statistics_com
                        pval_statistics_com[:, col] = pval
                    else:
                        # Save directly to test_statistics
                        test_statistics[perm, :, col] = np.abs(t_test)
                    # save t-test to base_statistics
                    if perm==0:
                        if test_combination in [True, "columns", "rows"]:
                            # Convert pval to z-score
                            #base_statistics_com[:,col] = norm.ppf(1 - np.array(np.squeeze(pval)))
                            base_statistics_com[:,col] = np.squeeze(pval)
                        else:
                            base_statistics[:,col]= t_test 
                elif category_columns["f_test_cols"] and col in category_columns["f_test_cols"]:
                    if np.sum(np.isnan(Din))>0 or np.sum(np.isnan(Rin))>0:
                        # Perform f-test while accounting for NaNs
                        f_test, pval =calculate_nan_f_test(Din, Rin[:, col], nan_values=True)
                    else:    
                        # Perform f-statistics
                        f_test, pval =calculate_nan_f_test(Din, Rin[:, col], nan_values=False)
                    # Put values inside test_statistics    
                    if test_combination in [True, "columns", "rows"]:
                        pval_statistics_com[:, col] = pval
                    else:
                        test_statistics[perm, :, col] = np.abs(f_test)
                    # Insert base statistics
                    if perm==0:
                        if test_combination in [True, "columns", "rows"]:
                            # Convert pval to z-score
                            #base_statistics_com[:,col] = norm.ppf(1 - np.array(np.squeeze(pval)))
                            base_statistics_com[:,col] = np.squeeze(pval)
                        else:
                            base_statistics[:,col]= f_test                  
                else:
                    # Perform corrrelation analysis
                    if np.sum(np.isnan(Din))>0 or np.sum(np.isnan(Rin))>0:
                        # Calculate the correlation matrix while handling NaN values 
                        # column by column without removing entire rows.
                        corr_array, pval =calculate_nan_correlation_matrix(Din, Rin[:, col], test_combination)
                    else:
                        # calculate correlation and pval between Din and Rin if there are no NaN values
                        #corr_array, pval= pearsonr(Din, np.expand_dims(Rin[:, col],axis=1))
                        if test_combination in [True, "columns", "rows"]:
                            
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
                    if test_combination in [True, "columns", "rows"]:
                        pval_statistics_com[:, col] = np.squeeze(pval)
                    else:
                        test_statistics[perm, :, col] = np.abs(np.squeeze(corr_array))       
                    # Insert base statistics
                    if perm==0:
                        if test_combination in [True, "columns", "rows"]:
                            # Convert pval to z-score
                            #base_statistics_com[:,col] = norm.ppf(1 - np.array(np.squeeze(pval)))
                            base_statistics_com[:,col] = np.squeeze(pval)
                        else:
                            base_statistics[:,col]= np.squeeze(corr_array)      
            if test_combination!=False:
                base_statistics = calculate_geometric_pval(base_statistics_com, test_combination) if perm==0 else base_statistics_com
                test_statistics[perm,:] =abs(calculate_geometric_pval(pval_statistics_com, test_combination)) 


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
        
    # Check if perm is 0 before returning the result
    return test_statistics, base_statistics

def calculate_geometric_pval(p_values, test_combination):
    """
    Calculate test statistics of z-scores converted from p-values based on the specified combination.

    Parameters:
    --------------
        p_values (numpy.ndarray):  Matrix of p-values.
        test_combination (str):       Specifies the combination method.
                                      Valid options: "True", "columns", "rows".
                                      Default is "True".

    Returns:
    ----------  
        result (numpy.ndarray):       Test statistics of z-scores converted from p-values.
    """
    if test_combination == True:
        pval = np.squeeze(np.exp(np.mean(np.log(p_values))))
        z_scores = norm.ppf(1 - np.array(pval))
        test_statistics = z_scores
    elif test_combination == "columns" or test_combination == "rows":
        axis = 0 if test_combination == "columns" else 1
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
        pval (numpy.ndarray): numpy array of p-values.
        method (str, optional): method used for FDR correction (default: 'fdr_bh).
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
        alpha (float, optional): Significance level (default: 0.05).
        include_nan: Include NaN values during the correction of p-values if True. Exclude NaN values if False (default: True).
        nan_diagonal: Add NaN values to the diagonal if True (default: False).

    Returns:
    ---------- 
    
        pval_corrected (numpy.ndarray): numpy array of corrected p-values.
        significant (numpy.ndarray): numpy array of boolean values indicating significant p-values.
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

def pval_cluster_based_correction(test_statistics, pval, alpha=0.05):
    """
    Perform cluster-based correction on test statistics using the output from permutation testing.
    The function corrects p-values by using the test statistics and p-values obtained from permutation testing.
    It converts the test statistics into z-based statistics, allowing to threshold and identify cluster sizes.
    The p-value map from permutation testing results is then thresholded using the cluster size derived from z-based statistics.
        
    Parameters
    ----------
    test_statistics : (numpy.ndarray)
        2D or 3D array of test statistics. 2D if you have applied permutation testing using "regression".
    pval : (numpy.ndarray)
        2D or 1D array of p-values obtained from permutation testing. 1D if you have applied permutation testing using "regression".
    alpha : (float, optional)
        Significance level for cluster-based correction (Defaults=0.05).

    Returns
    ----------
    p_values : (numpy.ndarray)
        Corrected p-values after cluster-based correction.
    """
    if test_statistics is []:
        raise ValueError("The variable 'test_statistics' is an empty list. To run the cluster-based permutation correction, you need to set 'test_statistics_option=True' when performing your test, as the distribution of test statistics is required for this function.")

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


def identify_coloumns_for_t_and_f_tests(R_data, method, identify_categories=True, category_lim=None):
    """
    Detect columns in R_data that are categorical. Used to detect which columns to perm t-statistics and F-statistics for later analysis.

    Parameters:
    -----------
    R_data : numpy.ndarray
        The 3D array containing categorical values.
    identify_categories : bool or list or numpy.ndarray, optional, default=True
        If True, automatically identify categorical columns. If list or ndarray, use the provided list of column indices.
    method : str, optional, default="univariate"
        The method to perform the tests. Only "univariate" is currently supported.
    category_lim : int or None, optional, default=None
        Maximum allowed number of categories for F-test. Acts as a safety measure for columns 
        with integer values, like age, which may be mistakenly identified as multiple categories.
    Returns:
    -----------
    dict
        A dictionary containing the columns for t-test ("t_test_cols") and F-test ("f_test_cols").

    Note: The function modifies the input dictionary `category_columns` in place.
    """
    # Initialize variable
    category_columns = {'t_test_cols': [], 'f_test_cols': []} 
    
    # Perform t-statistics for binary columns
    if identify_categories == True or isinstance(identify_categories, (list,np.ndarray)):
        if identify_categories==True:
            # Identify binary columns automatically in R_data
            # Do not calculate t-statistics when using regression for two categories
            if method!="regression":
                category_columns["t_test_cols"] = [col for col in range(R_data.shape[-1]) if np.unique(R_data[0,:, col]).size == 2]
            if category_lim != None:
                category_columns["f_test_cols"] = [col for col in range(R_data.shape[-1]) 
                                                   if np.unique(R_data[0,:, col]).size > 2  # Check if more than 2 unique values
                                                   # and np.issubdtype(R_data.dtype, np.integer) # Check if the data type is integer
                                                   and np.unique(R_data[0,:, col]).size < category_lim] # Check if the data type is above category_lim
            else:
                unique_counts = [np.unique(R_data[0, :, col]).size for col in range(R_data.shape[-1])]

                if max(unique_counts) > 20:
                    warnings.warn(
                        f"Detected more than 20 unique numbers in the dataset. "
                        f"If this is intended as categorical data, you can ignore this warning. "
                        f"Otherwise, consider defining 'category_lim' to set the maximum allowed categories or specifying the indices of categorical columns."
                    )
                category_columns["f_test_cols"] = [col for col, unique_count in enumerate(unique_counts)
                                        if unique_count > 2]
                # category_columns["f_test_cols"] = [col for col in range(R_data.shape[-1])
                #                                     if np.unique(R_data[0, :, col]).size > 2  # Check if more than 2 unique values
                #                                     #and np.issubdtype(R_data[0, :, col].dtype, np.integer)  # Check if the data type is integer
                #                                    ]     
                 
        else:
            # Do not calculate t-statistics when using regression for two categories
            if method!="regression":
                # Customize columns defined by the user
                category_columns["t_test_cols"] = [col for col in identify_categories if np.unique(R_data[0,:, col]).size == 2]
            #  Filter out the values from list identify_categories that are present in list category_columns["t_test_cols"]
            identify_categories_filtered = [value for value in identify_categories if value not in category_columns["t_test_cols"]]
            if category_lim != None:
                # We do not check if categories are integar values
                category_columns["f_test_cols"] = [col for col in identify_categories_filtered 
                                                   if np.unique(R_data[0,:, col]).size > 2 
                                                   and np.unique(R_data[0,:, col]).size < category_lim]
            else:
                # We do not check if categories are integar values
                category_columns["f_test_cols"] = [col for col in identify_categories_filtered 
                                                   if np.unique(R_data[0,:, col]).size > 2]
           
    return category_columns

def calculate_nan_regression(Din, Rin, proj):
    """
    Calculate the R-squared values for the regression of each dependent variable 
    in Rin on the independent variables in Din, while handling NaN values column-wise.

    Parameters:
    --------------
        Din (numpy.ndarray): Input data matrix for the independent variables.
        Rin (numpy.ndarray): Input data matrix for the dependent variables.
        proj (numpy.ndarray): Projection matrix.

    Returns:
    ----------  
        R2_test (numpy.ndarray): Array of R-squared values for each regression.
    """
    Rin = np.expand_dims(Rin, axis=1) if Rin.ndim==1 else Rin
    q = Rin.shape[-1]
    R2_test = np.zeros(q)
    # Calculate t-statistic for each pair of columns (D_column, R_data)
    for i in range(q):
        R_column = np.expand_dims(Rin[:, i],axis=1)
        valid_indices = np.all(~np.isnan(R_column), axis=1)
        
        beta = proj[:,valid_indices] @ R_column[valid_indices]  # Calculate regression_coefficients (beta)
        # Calculate the predicted values
        predicted_values = Din[valid_indices] @ beta
        # Calculate the total sum of squares (tss)
        tss = np.sum((R_column[valid_indices] - np.mean(R_column[valid_indices], axis=0))**2, axis=0)
        # Calculate the residual sum of squares (rss)
        rss = np.sum((predicted_values - R_column[valid_indices])**2, axis=0)
        # Calculate R^2
        base_statistics = 1 - (rss / tss) #r_squared
        # Store the R2 in an array
        R2_test[i] = base_statistics

    return R2_test
def calculate_nan_regression_f_test(Din, Rin, proj, nan_values=False):
    """
    Calculate the f-test values for the regression of each dependent variable 
    in Rin on the independent variables in Din, while handling NaN values column-wise.

    Parameters:
    --------------
        Din (numpy.ndarray): Input data matrix for the independent variables.
        Rin (numpy.ndarray): Input data matrix for the dependent variables.
        proj (numpy.ndarray): Projection matrix.

    Returns:
    ----------  
        R2_test (numpy.ndarray): Array of f-test values for each regression.
    """

    if nan_values:
        # Calculate F-statistics if there are Nan_values
        Rin = np.expand_dims(Rin, axis=1) if Rin.ndim==1 else Rin
        q = Rin.shape[-1]
        f_test = np.zeros(q)
        # Calculate t-statistic for each pair of columns (D_column, R_data)
        for i in range(q):
            # Indentify columns with NaN values
            R_column = np.expand_dims(Rin[:, i],axis=1)
            valid_indices = np.all(~np.isnan(R_column), axis=1)
            # Calculate beta
            beta = proj[:,valid_indices] @ R_column[valid_indices]  # Calculate regression_coefficients (beta)
            # Calculate the predicted values
            predicted_values = Din[valid_indices] @ beta
            # Calculate the total sum of squares (tss)
            tss = np.sum((R_column[valid_indices] - np.mean(R_column[valid_indices], axis=0))**2, axis=0)
            # Calculate the residual sum of squares (rss)
            rss = np.sum((predicted_values - R_column[valid_indices])**2, axis=0)
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
            f_test[i] = base_statistics
    else:
        # Calculate f-statistics
        # Fit the original model 
        beta = proj @ Rin  # Calculate regression_coefficients (beta)
        # Calculate the predicted values
        predicted_values = Din @ beta
        # Calculate the residual sum of squares (rss)
        rss = np.sum((predicted_values - Rin)**2, axis=0)
        # Calculate the total sum of squares (tss)
        tss = np.sum((Rin - np.nanmean(Rin, axis=0))**2, axis=0)
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
        f_test = (MSE_model / MSE_resid)

    return f_test
def calculate_nan_correlation_matrix(D_data, R_data, test_combination=False, reduce_pval_dims =False):
    """
    Calculate the correlation matrix between independent variables (D_data) and dependent variables (R_data),
    while handling NaN values column by column of dimension p without  without removing entire rows.
    
    Parameters:
    --------------
        D_data (numpy.ndarray): Input data matrix for the independent variables.
        R_data (numpy.ndarray): Input data matrix for the dependent variables.

    Returns:
    ----------  
        correlation_matrix (numpy.ndarray): Correlation matrix between columns in D_data and R_data.
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
                if test_combination in [True, "columns", "rows"]:
                   pval_matrix[i, j]  = np.nan 
                else:
                    correlation_matrix[i, j] = np.nan  
            else:
                # Find non-NaN indices for both D_column and R_column
                valid_indices = ~np.isnan(D_column) & ~np.isnan(R_column)           
                if test_combination in [True, "columns", "rows"]:
                    #pval_matrix = np.zeros(corr_matrix.shape)
                    _, pval= pearsonr(D_column[valid_indices], R_column[valid_indices])
                    pval_matrix[i, j]  = pval
                else:
                    # Calculate correlation coefficient matrix
                    corr_coef = np.corrcoef(D_column[valid_indices], R_column[valid_indices], rowvar=False)
                    # get the correlation matrix
                    correlation_matrix[i, j] = corr_coef[0, 1] 
    if reduce_pval_dims:
        if test_combination== True:
            pval_combination=np.squeeze(np.exp(np.nanmean(np.log(pval_matrix))))
        elif test_combination== "columns":
            pval_combination=np.squeeze(np.exp(np.nanmean(np.log(pval_matrix), axis=0)))
        elif test_combination== "rows":
            pval_combination = np.squeeze(np.exp(np.nanmean(np.log(pval_matrix), axis=1)))
    else:
        # No pval combination has been performed, this is done if we only calculate the p-values columnwise
        pval_combination = pval_matrix.copy()
        
    return correlation_matrix, pval_combination

def calculate_nan_t_test(D_data, R_column, nan_values=False):
    """
    Calculate the t-statistics between paired independent (D_data) and dependent (R_data) variables, while handling NaN values column by column without removing entire rows.
        - The function handles NaN values for each feature in D_data without removing entire rows.
        - NaN values are omitted on a feature-wise basis, and the t-statistic is calculated for each feature.
        - The resulting array contains t-statistics corresponding to each feature in D_data.

    Parameters:
    --------------
        D_data (numpy.ndarray): The input matrix of shape (n_samples, n_features).
        R_column (numpy.ndarray): The binary labels corresponding to each sample in D_data.

    Returns:
    ----------  
        t_test (numpy.ndarray): An array containing t-statistics for each feature in D_data against the binary categories in R_data.
 
    """
    if nan_values:
        # Initialize a matrix to store t-statistics
        p = D_data.shape[1]
        t_test = np.zeros(p)
        pval_array = np.zeros(p)
        # Extract non-NaN values for each group
        groups = np.unique(R_column)
        # Calculate t-statistic for each pair of columns (D_column, R_data)
        for i in range(p):
            D_column = np.expand_dims(D_data[:, i],axis=1)
                
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
        t_test, pval_array = ttest_ind(D_data[R_column == t_test_group[0]], D_data[R_column == t_test_group[1]]) 
    return t_test, pval_array


def calculate_nan_f_test(D_data, R_column, nan_values=False):
    """
    Calculate F-statistics for each feature of D_data against categories in R_data, while handling NaN values column by column without removing entire rows.
        - The function handles NaN values for each feature in D_data without removing entire rows.
        - NaN values are omitted on a feature-wise basis, and the F-statistic is calculated for each feature.
        - The resulting array contains F-statistics corresponding to each feature in D_data.

    Parameters:
    --------------
        D_data (numpy.ndarray): The input matrix of shape (n_samples, n_features).
        R_column (numpy.ndarray): The categorical labels corresponding to each sample in D_data.

    Returns:
    ----------  
        f_test (numpy.ndarray): An array containing F-statistics for each feature in D_data against the categories in R_data.
 
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

def detect_significant_intervals(pval, alpha):
    """
    Detect intervals of consecutive True values in a boolean array.

    Parameters
    ----------
    p_values : numpy.ndarray
        An array of p-values. 
    alpha : float, optional
        Threshold for significance (Default=0.05).

    Returns:
    ----------  
    list of tuple: A list of tuples representing the start and end indices
                   (inclusive) of each interval of consecutive True values.

    Example:
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

def __palm_quickperms(EB, M=None, nP=1000, CMC=False, EE=True):
    # Call palm_quickperms from palm_functions
    return palm_quickperms(EB, M, nP, CMC, EE)
