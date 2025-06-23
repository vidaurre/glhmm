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
from scipy.stats import ttest_ind, f_oneway, pearsonr, f, norm, t
from itertools import combinations
from sklearn.model_selection import train_test_split
import os
import re

def test_across_subjects(
    D_data, R_data, method="multivariate", Nnull_samples=0, confounds=None,  
    idx_data=None,  permute_within_blocks=False, permute_between_blocks=False, dict_family=None,   
    FWER_correction=False, combine_tests=False, detect_categorical=False, category_limit=10, n_cca_components=1,  
    predictor_names=[], outcome_names=[], return_base_statistics=True, verbose=True):
    """
    Perform permutation testing across subjects.

    This test assesses whether individual differences in predictor data (`D_data`) 
    correspond to variations in behavioural or demographic measures (`R_data`). 
    For example, it can test whether neural features (`D_data`) relate to cognitive scores, 
    clinical assessments, or demographic factors (`R_data`), such as examining 
    how brain activity patterns vary with age or cognitive performance across individuals.

    Parameters:
    --------------
    D_data (numpy.ndarray): 
        Input data as a 2D array of shape `(n, p)`, where:
        - `n` is the number of subjects.
        - `p` is the number of predictor variables.
    R_data (numpy.ndarray): 
        Outcome data as a 2D array.
        - 2D: `(n, q)`, where `n` is the number of subjects and `q` is the number of outcome variables.
    method (str, optional), default="multivariate": 
        The statistical method to use for the permutation test. Valid options are:
        - `"multivariate"`: Examines whether multiple predictors (e.g., brain states) varies across individuals based on outcome measures (e.g., memory scores, clinical assessments).  
        - `"univariate"`: Tests each predictor individually to assess whether it differs across subjects (e.g., whether time spent in a brain state is linked to cognitive performance).  
        - 'cca': Stands for Canonical Correlation Analysis; Identifies patterns of shared variation between predictors and outcomes across subjects, returning a single p-value indicating whether they are linked overall.  
    Nnull_samples (int), default=0:  
        Number of samples used to generate the null distribution, obtained via permutation. 
        Set `Nnull_samples=0` for parametric test output.   
    confounds (numpy.ndarray or None, optional), default=None:  
        Variables to control for (e.g., age, gender). These will be removed from the predictor data before analysis.  
        - Shape: `(n, c)`, where `c` is the number of confounding variables. 
    idx_data (numpy.ndarray), default=None:  
        Used when `permute_within_blocks=True` and/or `permute_between_blocks=True` 
        to define subject or session groupings for permutation.
        - 1D array: Specifies group labels for each subject or session (e.g., `[1,1,1,2,2,2,...,N]`).  
        - 2D array: Defines trial boundaries for each session as a list of `[start, end]` index pairs (e.g., `[[0, 50], [50, 100], [100, 150], ..., [N_start, N_end]]`).  
    permute_within_blocks (bool, optional), default=False:  
        If `True`, permutations happen **only within** predefined groups (e.g., subjects or sessions).  
    permute_between_blocks (bool, optional), default=False:  
        If `True`, permutations happen **between** different predefined groups.
    dict_family (dict, optional):  
        Only needed when subjects have family relationships that must be accounted for in the permutation scheme. 
        - Contains information about related individuals.  
        - See PALM documentation: [PALM Exchangeability Blocks](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/PALM(2f)ExchangeabilityBlocks.html).
        If provided, this must include:
        - file_location (str): Path to the Exchangeable Block (EB) CSV-file with family structure information.
    FWER_correction (bool, optional), default=False:  
        If `True`, applies family-wise error rate (FWER) correction using the MaxT method to reduce false positives.      
    combine_tests (bool or str, optional), default=False:  
        Summarises multiple p-values using the Non-Parametric Combination (NPC) method 
        for multivariate and univariate tests.
        - `True`: Returns a single p-value summarizing the entire test (`1 × 1`).
        - `"across_rows"`: Returns one p-value per outcome (`1 × q`).
        - `"across_columns"`: Returns one p-value per predictor (`1 × p`), only for univariate tests.
    detect_categorical (bool), default=False:  
        If `True`, automatically detects categorical columns in `R_data` and applies the appropriate statistical test:  
        - Binary categorical variables: Independent t-test.  
        - Multiclass categorical variables: MANOVA for multivariate tests, ANOVA (F-test) for univariate test.  
        - Continuous variables:  F-regression for multivariate tests, Pearson’s t-test for univariate tests.
    category_limit (int), default=10
        Maximum number of unique values allowed to decide whether to use F-ANOVA or F-regression in the F-test.
        Prevents integer-valued variables (e.g., age) from being misclassified as categorical.
    n_cca_components (int), default=1:
        Number of Canonical Correlation Analysis (CCA) components to compute. 
    predictor_names (list of str, optional):  
        Names of the predictor variables in `D_data` (should match column order).  
    outcome_names (list of str, optional):  
        Names of the outcome variables in `R_data`.
    return_base_statistics (bool, optional), default=True:  
        If `True`, stores test statistics from all permutations.  
    verbose (bool, optional):  
        If `True`, prints progress messages.      
        
    Returns:
    ----------  
    result (dict): 
        The returned dictionary contains:
        'pval': P-values for each test:
            - `"multivariate"`: `(q,)`
            - `"univariate"`: `(p, q)`
            - `"cca"`: `(1)`
        'base_statistics': Test statistics computed from the original/observed data (unpermuted).
        `null_stat_distribution`:  Distribution of test statistics under the null hypothesis, where the first row always corresponds to the observed test statistic.
            - `"multivariate"`: `(Nnull_samples, q)`
            - `"univariate"`: `(Nnull_samples, p, q)`
            - `"cca"`: `(Nnull_samples, 1)`
        'statistical_measures': Dictionary specifying the units used for each column in 'null_stat_distribution'.
        'test_type': `"test_across_subjects"`.
        'method': Analytical approach used (e.g., 'multivariate', 'univariate', 'CCA').
        'combine_tests': Marks whether the Non-Parametric Combination (NPC) method was applied to summarise p-values(`False`/`True`/`across_rows`/`across_columns`).
        'max_correction': Marks whether FWER correction was applied (`True`/`False`).
        'Nnull_samples': Number of samples used to generate the null distribution, obtained via permutation.
        'test_summary': Dictionary summarising test results based on the selected method.
        'pval_f_multivariate' *(only if `method="multivariate"` and `Nnull_samples > 1`)*: P-values from the F-test in the multivariate analysis.
        'pval_t_multivariate' *(only if `method="multivariate"` and `Nnull_samples > 1`)*: P-values from the t-tests in the multivariate analysis.

    """
    # Initialize variables
    test_type = 'test_across_subjects'
    method = method.lower()
    
    # Adding an extra permutation values, since the first one would be the base statistics
    FLAG_parametric = 0
    # Ensure Nnull_samples is at least 1
    if Nnull_samples ==0:
        # Set flag for identifying categories without permutation
        detect_categorical = True
        if method == 'cca' and Nnull_samples == 0:
            raise ValueError("CCA does not support parametric statistics. The number of permutations ('Nnull_samples') cannot be set to 1. "
                            "Please increase the number of permutations to perform a valid analysis.")
        Nnull_samples +=1
        FLAG_parametric =1
        print("Applying parametric test")
        
    # Check validity of method and data_type
    valid_methods = ["multivariate", "univariate", "cca"]
    validate_condition(method.lower() in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    if method == "cca":
        if FWER_correction and n_cca_components==1:
            raise ValueError(
                "FWER correction using MaxT is not applicable for CCA because only a single p-value is returned. "
                "Set FWER_correction=False to proceed."
            )
        if combine_tests:
            raise ValueError("CCA does not support multiple test combinations. Set 'combine_tests' to False.")

        max_components = min(D_data.shape[-1], R_data.shape[-1])
        if n_cca_components > max_components:
            warnings.warn(
                f"'n_cca_components' ({n_cca_components}) exceeds the maximum allowed ({max_components}). "
                f"Adjusting 'n_cca_components' to {max_components}."
            )
            n_cca_components = max_components
            
    # Check validity of method
    valid_combine_tests = [False, True, "across_columns", "across_rows"]
    validate_condition(combine_tests in valid_combine_tests, "Invalid option specified for 'combine_tests'. Must be one of: " + ', '.join(map(str, valid_combine_tests)))
    
    if method=="multivariate" and combine_tests is valid_combine_tests[-1]:
            raise ValueError("method is set to 'multivariate' and 'combine_tests' is set to 'across_rows.\n"
                            "The multivariate test will return (1-by-q) p-values, so the test combination can only be across_columns and return a single p-value.\n "
                         "If you want to perform 'combine_tests' while doing 'multivariate' then please set 'combine_tests' to 'True' or 'across_columns.")

    if FWER_correction and combine_tests in [True, "across_columns", "across_rows"]:
       # Raise an exception and stop function execution
        raise ValueError("'FWER_correction' is set to True and 'combine_tests' is either True, 'columns', or 'rows'. \n"
                         "Please set 'FWER_correction' to False if you want to apply 'combine_tests' or set 'combine_tests' to False if you want to run 'FWER_correction'.")

    if not isinstance(detect_categorical,bool):
        raise TypeError("detect_categorical must be a boolean value (True or False)") 
    if (permute_within_blocks or permute_between_blocks) and idx_data is None:
        # Raise an exception and stop function execution
        raise ValueError("Cannot perform within-block or between-block permutation without 'idx_data'.\n"
                        "Please provide 'idx_data' to define block boundaries for the permutation.")
  
    if not permute_within_blocks and not permute_between_blocks and idx_data is not None:
        # Raise an exception and stop function execution
        raise ValueError("When providing 'idx_data' to define block/group boundaries, either 'permute_within_blocks' or 'permute_between_blocks' must be set to True.\n"
                        "If neither is set to True, do not provide 'idx_data' and perform permutation testing across subjects instead.")
    
    # Get indices for permutation
    idx_array = get_indices_array(idx_data) if idx_data is not None and idx_data.ndim == 2 else idx_data.copy() if idx_data is not None else None

    # Calculate possible permutations
    if FLAG_parametric==0 and dict_family is None:
        compute_max_permutations(idx_array, permute_within_blocks, permute_between_blocks, Nnull_samples=Nnull_samples, verbose=verbose)

    # Get the shapes of the data
    n_T, n_N, n_p, n_q, D_data, R_data = get_input_shape(D_data, R_data, verbose)
    # Note for convension we wrote (T, p, q) => (n_T, n_p, n_q)
    
    # Identify categorical columns in R_data
    category_columns = categorize_columns_by_statistical_method(R_data, method, Nnull_samples, detect_categorical, category_limit) 

    if category_columns["t_stat_independent_cols"]!=[] or category_columns["f_anova_cols"]!=[] or category_columns["f_reg_cols"]!=[]:
        if FWER_correction and (len(category_columns.get('t_stat_independent_cols')) != R_data.shape[-1] or len(category_columns.get('f_anova_cols')) != R_data.shape[-1] or len(category_columns.get('f_reg_cols')) != R_data.shape[-1]):
            print("Warning: Cannot perform FWER_correction with different test statisticss.\n"
                  "Consider to set detect_categorical=False")
            raise ValueError("Cannot perform FWER_correction")  

    # Crate the family structure by looking at the dictionary 
    if dict_family is not None:
        # process dictionary of family structure
        dict_mfam=process_family_structure(dict_family, Nnull_samples) 
        
    
    # Initialize arrays based on shape of data shape and defined options
    pval, base_statistics, test_statistics_list, F_stats_list, t_stats_list, R2_stats_list = initialize_arrays(n_p, n_q, n_T, method, Nnull_samples, return_base_statistics, combine_tests, n_cca_components)
    
    # Custom variable names
    if combine_tests is not False:
        test_com_shape = (1, 1) if combine_tests == True else (n_p, 1) if combine_tests == "across_columns" else (1, n_q)
        n_p = test_com_shape[0]
        n_q = test_com_shape[-1]

    # Get the labels for the predictor and outcome variables
    predictor_names, outcome_names = define_predictor_outcome_names(method, combine_tests, predictor_names, outcome_names, n_p, n_q)

    # Permutation matrix
    if dict_family is not None and idx_array is None:
        permutation_matrix = __palm_quickperms(dict_mfam["EB"], M=dict_mfam["M"], nP=dict_mfam["nP"], 
                                            CMC=dict_mfam["CMC"], EE=dict_mfam["EE"])
        # Convert the index so it starts from 0
        permutation_matrix -= 1
    elif idx_array is None:
        # Get indices for permutation across subjects
        permutation_matrix = permutation_matrix_across_subjects(Nnull_samples, R_data)
    elif idx_array is not None:
        if permute_within_blocks:
            if permute_between_blocks:
                # Permutation within and between blocks/groups
                permutation_matrix = permutation_matrix_within_and_between_groups(Nnull_samples, R_data, idx_array)
            else:
                # Permutation within trials across sessions (within blocks/groups only)
                permutation_matrix = permutation_matrix_across_trials_within_session(Nnull_samples, R_data, idx_array)
        elif permute_between_blocks:
            # Permutation across sessions (between blocks/groups only)
            permutation_matrix = permutation_matrix_within_subject_across_sessions(Nnull_samples, R_data, idx_array)
    # handle NaN values in the dataset
    D_data, R_data, confounds, permutation_matrix_update =handle_nan_values(D_data, R_data, confounds, permutation_matrix, method)
            
    for t in tqdm(range(n_T)) if n_T > 1 & verbose ==True else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        D_t, R_t = deconfound_values(D_data[t, :],R_data, confounds)

        # Create test_statistics based on method
        test_statistics, reg_pinv = initialize_permutation_matrices(method, Nnull_samples, n_p, n_q, D_t, combine_tests, category_columns=category_columns, n_cca_components=n_cca_components)

        for perm in tqdm(range(Nnull_samples)) if n_T == 1 & verbose==True else range(Nnull_samples):
            # Perform permutation on R_t
            Rin = R_t[permutation_matrix_update[:, perm]]
            # Calculate the permutation distribution
            stats_results = test_statistics_calculations(D_t, Rin, perm, test_statistics, reg_pinv, method, category_columns,combine_tests, Nnull_samples=Nnull_samples, n_cca_components=n_cca_components)
            base_statistics[t, :] = stats_results["base_statistics"] if perm == 0 and stats_results["base_statistics"] is not None else base_statistics[t, :] 
            pval[t, :] = stats_results["pval_matrix"] if perm == 0 and stats_results["pval_matrix"] is not None else pval[t, :]
            F_stats_list[t, perm, :] = stats_results["F_stats"] if stats_results["F_stats"] is not None else F_stats_list[t, perm, :]
            t_stats_list[t, perm, :] = stats_results["t_stats"] if stats_results["t_stats"] is not None else t_stats_list[t, perm, :]
        if FLAG_parametric==0:
            # Calculate p-values
            pval = get_pval(test_statistics, Nnull_samples, method, t, pval, FWER_correction)

        # Output test statistics if it is set to True can be hard for memory otherwise
        if return_base_statistics==True:
            test_statistics_list[t,:] = stats_results["null_stat_distribution"]


    if combine_tests is not False:
        # Set all values to empty lists (except for cases like "all_columns")
        category_columns = {key: [] for key in category_columns}
        category_columns['z_score'] = combine_tests

    if test_statistics_list is not None or n_T==1:
        test_statistics_list =np.squeeze(test_statistics_list,axis=0) 


    # Get f and t states when doing a multivariate test
    f_t_stats = get_f_t_stats(D_data, R_data, F_stats_list, t_stats_list, Nnull_samples, n_T) if method =='multivariate' else None
    # Create report summary
    test_summary =create_test_summary(R_data, base_statistics, pval, predictor_names, outcome_names, method, f_t_stats,n_T, n_N, n_p,n_q, Nnull_samples, category_columns, combine_tests)
    # Change the output to say Nnull_samples=0
    Nnull_samples = 0 if Nnull_samples==1 else Nnull_samples    
    category_columns = {key: value for key, value in category_columns.items() if value}
    if len(category_columns)==1:
        category_columns[next(iter(category_columns))]='all_columns'
    elif combine_tests is not False:
        category_columns['z_score'] ='all_columns'
    elif category_columns=={} and method=='univariate':
        category_columns['t_stat_independent_cols']='all_columns'


    if n_T==1:
        pval =np.squeeze(pval,axis=0) 

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
        'null_stat_distribution': test_statistics_list,
        'statistical_measures': category_columns,
        'test_type': test_type,
        'method': method,
        'combine_tests': combine_tests,
        'max_correction':FWER_correction,
        'Nnull_samples': Nnull_samples,
        'test_summary':test_summary}
    if method =='multivariate' and Nnull_samples>1:
        result['pval_F_multivariate'] = f_t_stats['perm_p_values_F']
        result['pval_t_multivariate'] = f_t_stats['perm_p_values_t']
    return result



def test_across_trials(D_data, R_data, idx_data, method="multivariate", Nnull_samples=0, confounds=None, 
                       FWER_correction=False, combine_tests=False, detect_categorical=False, category_limit=10, n_cca_components=1,
                       predictor_names=[], outcome_names=[], return_base_statistics=True, verbose=True):
    """
    Perform permutation testing across different trials within a session.

    This test assesses whether trial-to-trial variations in predictor data (`D_data`) 
    correspond to differences in outcome data (`R_data`). 
    For example, it can evaluate whether neural responses (`D_data`) differ depending 
    on behavioural conditions (`R_data`), such as viewing an animate versus inanimate object.
             
    Parameters:
    --------------
    D_data (numpy.ndarray):  
        Predictor data representing brain activity or other features.  
        - 2D: `(n, p)`, where `n` is the number of trials and `p` is the number of predictors.  
        - 3D: `(T, n, p)`, where `T` is the number of timepoints, `n` is the number of trials, and `p` is the number of predictors.
    R_data (numpy.ndarray):  
        Outcome data representing behavioral responses or other dependent measures.  
        - 2D: `(n, q)`, where `n` is the number of trials and `q` is the number of outcome variables.  
    idx_data (numpy.ndarray):  
        Defines how trials are grouped into different sessions (e.g., trial type labels or session indices). 
        - 1D array: Specifies group labels for each subject or session (e.g., `[1,1,1,2,2,2,...,N]`).  
        - 2D array: Defines trial boundaries for each session as a list of `[start, end]` index pairs (e.g., `[[0, 50], [50, 100], [100, 150], ..., [N_start, N_end]]`).    
    method (str, optional), default="multivariate": 
        The statistical method to use for the permutation test. Valid options are:
        - `"multivariate"`: Examines whether multiple predictors (e.g., brain states) varies across trial types.  
        - `"univariate"`: Tests each predictor individually to assess whether it differs between trial types (e.g., whether specific brain states are different across conditions).  
        - 'cca': Stands for Canonical Correlation Analysis; Identifies patterns of shared variation between predictors and outcomes across trials, returning a single p-value indicating whether they are linked overall.
    Nnull_samples (int), default=0: 
        Number of samples used to generate the null distribution, obtained via permutation.
        Set `Nnull_samples=0` for parametric test output. 
    FWER_correction (bool, optional), default=False:  
        If `True`, applies family-wise error rate (FWER) correction using the MaxT method to reduce false positives.       
    confounds (numpy.ndarray or None, optional), default=None: 
        The confounding variables to be controlled for, i.e. regressed out from the input data.
        The array should have a shape (n, c), where n is the number of trials and c is the number of confounding variables. 
        Each column represents a different confound to be controlled for in the analysis.
    combine_tests (bool or str, optional), default=False:  
        Summarises multiple p-values using the Non-Parametric Combination (NPC) method 
        for multivariate and univariate tests.
        - `True`: Returns a single p-value summarizing the entire test (`1 × 1`).
        - `"across_rows"`: Returns one p-value per outcome (`1 × q`).
        - `"across_columns"`: Returns one p-value per predictor (`1 × p`), only for univariate tests.
    detect_categorical (bool), default=False:  
        If `True`, automatically detects categorical columns in `R_data` and applies the appropriate statistical test:  
        - Binary categorical variables: Independent t-test.  
        - Multiclass categorical variables: MANOVA for multivariate tests, ANOVA (F-test) for univariate test.  
        - Continuous variables:  F-regression for multivariate tests, Pearson’s t-test for univariate tests.
    category_limit (int), default=10
        Maximum number of unique values allowed to decide whether to use F-ANOVA or F-regression in the F-test.
        Prevents integer-valued variables (e.g., age) from being misclassified as categorical.
    n_cca_components (int), default=1:
        Number of Canonical Correlation Analysis (CCA) components to compute. 
    predictor_names (list of str, optional):  
        Names of the predictor variables in `D_data` (should match column order).  
    outcome_names (list of str, optional):  
        Names of the outcome variables in `R_data`.
    return_base_statistics (bool, optional), default=True:  
        If `True`, stores test statistics from all permutations.  
    verbose (bool, optional):  
        If `True`, prints progress messages.      

    Returns:
    ----------  
    result (dict): 
        The returned dictionary contains:
        'pval': P-values for each test:
            - `"multivariate"`: `(T, q)`
            - `"univariate"`: `(T, p, q)`
            - `"cca"`: `(T, 1)`
        'base_statistics': Test statistics computed from the original/observed data (unpermuted).
        `null_stat_distribution`: Distribution of test statistics under the null hypothesis, where the first row always corresponds to the observed test statistic.
            - `"multivariate"`: `(T, Nnull_samples, q)`
            - `"univariate"`: `(T, Nnull_samples, p, q)`
            - `"cca"`: `(T, Nnull_samples, 1)`
        'statistical_measures': Dictionary mapping each column in the 'q' dimension of 'null_stat_distribution' to its corresponding unit.
        'test_type': `"test_across_trials"`.
        'method': Analytical approach used (e.g., 'multivariate', 'univariate', 'CCA').
        'combine_tests': Marks whether the Non-Parametric Combination (NPC) method was applied to summarise p-values (`False`/`True`/`across_rows`/`across_columns`.
        'max_correction': Marks whether FWER correction was applied (`True`/`False`).
        'Nnull_samples': 'Nnull_samples': Number of samples used to generate the null distribution, obtained via permutation.
        'test_summary': Dictionary summarising test results based on the selected method.
        'pval_f_multivariate' *(only if `method="multivariate"` and `Nnull_samples > 1`)*: P-values from the F-test in the multivariate analysis.
        'pval_t_multivariate' *(only if `method="multivariate"` and `Nnull_samples > 1`)*: P-values from the t-tests in the multivariate analysis.

    """
    # Initialize variable
    category_columns = []   
    test_type =  'test_across_trials'
    method = method.lower()
    FLAG_parametric = 0
    # Ensure Nnull_samples is at least 1
    if Nnull_samples ==0:
        # Set flag for identifying categories without permutation
        detect_categorical = True
        if method == 'cca' and Nnull_samples == 0:
            raise ValueError("CCA does not support parametric statistics. The number of permutations ('Nnull_samples') cannot be set to 1. "
                            "Please increase the number of permutations to perform a valid analysis.")
        Nnull_samples +=1
        FLAG_parametric =1
        print("Applying parametric test")

    # Check validity of method and data_type
    valid_methods = ["multivariate", "univariate", "cca"]
    validate_condition(method.lower() in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    if method == "cca":
        if FWER_correction and n_cca_components==1:
            raise ValueError(
                "FWER correction using MaxT is not applicable for CCA because only a single p-value is returned. "
                "Set FWER_correction=False to proceed."
            )
        if combine_tests:
            raise ValueError("CCA does not support multiple test combinations. Set 'combine_tests' to False.")

        max_components = min(D_data.shape[-1], R_data.shape[-1])
        if n_cca_components > max_components:
            warnings.warn(
                f"'n_cca_components' ({n_cca_components}) exceeds the maximum allowed ({max_components}). "
                f"Adjusting 'n_cca_components' to {max_components}."
            )
            n_cca_components = max_components

    # Check validity of method
    valid_combine_tests = [False, True, "across_columns", "across_rows"]
    validate_condition(combine_tests in valid_combine_tests, "Invalid option specified for 'combine_tests'. Must be one of: " + ', '.join(map(str, valid_combine_tests)))
    if not isinstance(detect_categorical,bool):
        raise TypeError("detect_categorical must be a boolean value (True or False)")
    if method=="multivariate" and combine_tests is valid_combine_tests[-1]:
            raise ValueError("method is set to 'multivariate' and 'combine_tests' is set to 'across_rows.\n"
                            "The multivariate test will return (1-by-q) p-values, so the test combination can only be across_rows which is one in this case to single p-value.\n"
                         "If you want to perform 'combine_tests' while doing 'multivariate' then please set 'combine_tests' to 'True' or 'across_columns.")


    if FWER_correction and combine_tests in [True, "across_columns", "across_rows"]:
       # Raise an exception and stop function execution
        raise ValueError("'FWER_correction' is set to True and 'combine_tests' is either True, 'columns', or 'rows'.\n "
                         "Please set 'FWER_correction' to False if you want to apply 'combine_tests' or set 'combine_tests' to False if you want to run 'FWER_correction'.")

    # Get input shape information
    n_T, n_N, n_p, n_q, D_data, R_data = get_input_shape(D_data, R_data, verbose)
    # Identify categorical columns in R_data
    category_columns = categorize_columns_by_statistical_method(R_data, method, Nnull_samples, detect_categorical, category_limit)

    if category_columns["t_stat_independent_cols"]!=[] or category_columns["f_anova_cols"]!=[] or category_columns["f_reg_cols"]!=[]:
        if FWER_correction and (len(category_columns.get('t_stat_independent_cols')) != R_data.shape[-1] or len(category_columns.get('f_anova_cols')) != R_data.shape[-1] or len(category_columns.get('f_reg_cols')) != R_data.shape[-1]):
            print("Warning: Cannot perform FWER_correction with different test statisticss.\n"
                  "Consider to set detect_categorical=False")
            raise ValueError("Cannot perform FWER_correction")  
    
    # Get indices for permutation
    if len(idx_data.shape)==2:
        idx_array = get_indices_array(idx_data)
    else:
        idx_array =idx_data.copy()        

    # Calculate possible permutations
    if FLAG_parametric==0:
        compute_max_permutations(idx_array, permute_within_blocks=True, permute_between_blocks=False, Nnull_samples=Nnull_samples, verbose=verbose)

    # Initialize arrays based on shape of data shape and defined options
    pval, base_statistics, test_statistics_list, F_stats_list, t_stats_list, R2_stats_list = initialize_arrays(n_p, n_q, n_T, method, Nnull_samples, return_base_statistics, combine_tests, n_cca_components)
    
    # Custom variable names
    if combine_tests is not False:
        test_com_shape = (1, 1) if combine_tests == True else (n_p, 1) if combine_tests == "across_columns" else (1, n_q)
        n_p = test_com_shape[0]
        n_q = test_com_shape[-1]

    # Define names for the summary statistics
    predictor_names = [f"State {i+1}" for i in range(n_p)] if predictor_names==[] or len(predictor_names)!=n_p else predictor_names
    outcome_names = [f"Regressor {i+1}" for i in range(n_q)] if outcome_names==[] or len(outcome_names)!=n_q else outcome_names
    
    permutation_matrix = permutation_matrix_across_trials_within_session(Nnull_samples,R_data, idx_array)
    D_data, R_data, confounds, permutation_matrix_update =handle_nan_values(D_data, R_data, confounds, permutation_matrix, method)

    for t in tqdm(range(n_T)) if n_T > 1 & verbose ==True else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        D_t, R_t = deconfound_values(D_data[t, :],R_data, confounds)
        # Create test_statistics and the regularized pseudo-inverse of D_data
        test_statistics, reg_pinv = initialize_permutation_matrices(method, Nnull_samples, n_p, n_q, D_t, combine_tests, category_columns=category_columns, n_cca_components=n_cca_components)
       
        for perm in range(Nnull_samples):
            # Perform permutation on R_t
            Rin = R_t[permutation_matrix_update[:, perm]]
            # Calculate the permutation distribution
            stats_results = test_statistics_calculations(D_t, Rin, perm, test_statistics, reg_pinv, method, category_columns,combine_tests, n_cca_components=n_cca_components)
            base_statistics[t, :] = stats_results["base_statistics"] if perm == 0 and stats_results["base_statistics"] is not None else base_statistics[t, :] 
            pval[t, :] = stats_results["pval_matrix"] if perm == 0 and stats_results["pval_matrix"] is not None else pval[t, :]
            F_stats_list[t, perm, :] = stats_results["F_stats"] if stats_results["F_stats"] is not None else F_stats_list[t, perm, :]
            t_stats_list[t, perm, :] = stats_results["t_stats"] if stats_results["t_stats"] is not None else t_stats_list[t, perm, :]
            

        if FLAG_parametric==0:
            # Calculate p-values
            pval = get_pval(test_statistics, Nnull_samples, method, t, pval, FWER_correction)
            
            # Output test statistics if it is set to True can be hard for memory otherwise
            if return_base_statistics==True:
                test_statistics_list[t,:] = test_statistics

    if Nnull_samples >0 and combine_tests is not False:
        # Set all values to empty lists (except for cases like "all_columns")
        category_columns = {key: [] for key in category_columns}
        category_columns['z_score'] = combine_tests
    
    # Get f and t states when doing a multivariate test
    f_t_stats = get_f_t_stats(D_data, R_data, F_stats_list, t_stats_list, Nnull_samples, n_T) if method =='multivariate' else None
    # Create report summary
    test_summary =create_test_summary(R_data, base_statistics,pval, predictor_names, outcome_names, method, f_t_stats,n_T, n_N, n_p,n_q, Nnull_samples, category_columns, combine_tests)

    category_columns = {key: value for key, value in category_columns.items() if value}
    if len(category_columns)==1:
        category_columns[next(iter(category_columns))]='all_columns'
    elif combine_tests is not False:
        category_columns['z_score'] ='all_columns'
    elif category_columns=={} and method=='univariate':
        category_columns['t_stat_independent_cols']='all_columns'

    if np.sum(np.isnan(pval))>0 & verbose:
        print("Warning: Permutation testing resulted in p-values equal to NaN.\n")
        print("This may indicate an issue with the input data. Please review your data.")
        
    # Return results
    result = {
        'pval': pval,
        'base_statistics': base_statistics,
        'null_stat_distribution': test_statistics_list,
        'statistical_measures': category_columns,
        'test_type': test_type,
        'method': method,
        'combine_tests': combine_tests,
        'max_correction':FWER_correction,
        'Nnull_samples': Nnull_samples,
        'test_summary':test_summary}
    if method =='multivariate' and Nnull_samples>1:
        result['pval_F_multivariate'] = f_t_stats['perm_p_values_F']
        result['pval_t_multivariate'] = f_t_stats['perm_p_values_t']
    return result

def test_across_sessions_within_subject(D_data, R_data, idx_data, method="multivariate", Nnull_samples=0, confounds=None, 
                                        FWER_correction=False, combine_tests=False, detect_categorical=False, category_limit=10, n_cca_components=1,
                                        predictor_names=[], outcome_names=[], return_base_statistics=True, verbose = True):
    """
    Perform permutation testing across sessions while maintaining trial order.

    This test assesses whether session-to-session variations in predictor data (`D_data`) 
    correspond to differences in outcome data (`R_data`). For example, it can evaluate 
    whether neural responses (`D_data`) change across repeated sessions of the same task (`R_data`), 
    such as tracking performance improvements or shifts in brain activity over time.

    Parameters:
    --------------
    D_data (numpy.ndarray):  
        Predictor data representing brain activity or other features.  
        - 2D: `(n, p)`, where `n` is the number of trials and `p` is the number of predictors.  
        - 3D: `(T, n, p)`, where `T` is the number of timepoints, `n` is the number of trials, and `p` is the number of predictors.
    R_data (numpy.ndarray):  
        Outcome data representing behavioral responses or other dependent measures.  
        - 2D: `(n, q)`, where `n` is the number of trials and `q` is the number of outcome variables.  
    idx_data (numpy.ndarray):  
        Defines how trials are grouped into different sessions (e.g., trial type labels or session indices). 
        - 1D array: Specifies group labels for each subject or session (e.g., `[1,1,1,2,2,2,...,N]`).  
        - 2D array: Defines trial boundaries for each session as a list of `[start, end]` index pairs (e.g., `[[0, 50], [50, 100], [100, 150], ..., [N_start, N_end]]`).   
    method (str, optional), default="multivariate": 
        The statistical method used to compare differences across sessions. Valid options are:
        - `"multivariate"`: Examines whether multiple predictors (e.g., brain states) changes across sessions.   
        - `"univariate"`: Tests each predictor individually to assess whether it varies across sessions (e.g., whether specific brain states when doing the same task changes across session).  
        - 'cca': Stands for Canonical Correlation Analysis; Identifies patterns of shared variation between predictors and outcomes across sessions, returning a single p-value indicating whether they are linked overall.
    Nnull_samples (int), default=0: 
        Number of samples used to generate the null distribution, obtained via permutation.
        Set `Nnull_samples=0` for parametric test output. 
    FWER_correction (bool, optional), default=False:  
        If `True`, applies family-wise error rate (FWER) correction using the MaxT method to reduce false positives.       
    confounds (numpy.ndarray or None, optional), default=None: 
        The confounding variables to be controlled for, i.e. regressed out from the input data.
        The array should have a shape (n, c), where n is the number of trials and c is the number of confounding variables. 
        Each column represents a different confound to be controlled for in the analysis. 
    combine_tests (bool or str, optional), default=False:  
        Summarises multiple p-values using the Non-Parametric Combination (NPC) method 
        for multivariate and univariate tests.
        - `True`: Returns a single p-value summarizing the entire test (`1 × 1`).
        - `"across_rows"`: Returns one p-value per outcome (`1 × q`).
        - `"across_columns"`: Returns one p-value per predictor (`1 × p`), only for univariate tests.
    detect_categorical (bool), default=False:  
        If `True`, automatically detects categorical columns in `R_data` and applies the appropriate statistical test:  
        - Multiclass categorical variables: ANOVA (F-test).  
        - Continuous variables:  F-regression for multivariate tests.
    category_limit (int), default=10
        Maximum number of unique values allowed to decide whether to use F-ANOVA or F-regression in the F-test.
        Prevents integer-valued variables (e.g., age) from being misclassified as categorical.
    n_cca_components (int), default=1:
        Number of Canonical Correlation Analysis (CCA) components to compute. 
    predictor_names (list of str, optional):  
        Names of the predictor variables in `D_data` (should match column order).  
    outcome_names (list of str, optional):  
        Names of the outcome variables in `R_data`.
    return_base_statistics (bool, optional), default=True:  
        If `True`, stores test statistics from all permutations.  
    verbose (bool, optional):  
        If `True`, prints progress messages.      

    Returns:
    ----------  
    result (dict): 
        The returned dictionary contains:
        'pval': P-values for each test:
            - `"multivariate"`: `(T, q)`
            - `"univariate"`: `(T, p, q)`
            - `"cca"`: `(T, 1)`
        'base_statistics': Test statistics computed from the original/observed data (unpermuted).
        `null_stat_distribution`: Distribution of test statistics under the null hypothesis, where the first row always corresponds to the observed test statistic.
            - `"multivariate"`: `(T, Nnull_samples, q)`
            - `"univariate"`: `(T, Nnull_samples, p, q)`
            - `"cca"`: `(T, Nnull_samples, 1)`
        'statistical_measures': Dictionary mapping each column in the 'q' dimension of 'null_stat_distribution' to its corresponding unit.
        'test_type': `"test_across_sessions"`.
        'method': Analytical approach used (e.g., 'multivariate', 'univariate', 'CCA').
        'combine_tests': Marks whether the Non-Parametric Combination (NPC) method was applied to summarise p-values (`False`/`True`/`across_rows`/`across_columns`.
        'max_correction': Mraks whether FWER correction was applied (`True`/`False`).
        'Nnull_samples': Number of samples used to generate the null distribution, obtained via permutation for permutation-based tests.
        'test_summary': Dictionary summarising test results.
        'pval_f_multivariate' *(only if `method="multivariate"` and `Nnull_samples > 1`)*: P-values from the F-test in the multivariate analysis.
        'pval_t_multivariate' *(only if `method="multivariate"` and `Nnull_samples > 1`)*: P-values from the t-tests in the multivariate analysis. 
    """ 
    # Initialize variable
    category_columns = []  
    test_type = 'test_across_sessions'  
    method = method.lower()
    permute_beta = True # For across session test we are permuting the beta coefficients for each session
    FLAG_parametric = 0

    # Ensure Nnull_samples is at least 1
    if Nnull_samples ==0:
        # Set flag for identifying categories without permutation
        detect_categorical = True
        if method == 'cca' and Nnull_samples == 0:
            raise ValueError("CCA does not support parametric statistics. The number of permutations ('Nnull_samples') cannot be set to 1. "
                            "Please increase the number of permutations to perform a valid analysis.")
        Nnull_samples +=1
        FLAG_parametric =1
        print("Applying parametric test")


    if not isinstance(detect_categorical,bool):
        raise TypeError("detect_categorical must be a boolean value (True or False)")
    # Check validity of method and data_type
    valid_methods = ["multivariate", "univariate", "cca"]
    validate_condition(method.lower() in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    # Check validity of method
    valid_combine_tests = [False, True, "across_columns", "across_rows"]
    validate_condition(combine_tests in valid_combine_tests, "Invalid option specified for 'combine_tests'. Must be one of: " + ', '.join(map(str, valid_combine_tests)))
    
    if method == "cca":
        if FWER_correction and n_cca_components==1:
            raise ValueError(
                "FWER correction using MaxT is not applicable for CCA because only a single p-value is returned. "
                "Set FWER_correction=False to proceed."
            )
        if combine_tests:
            raise ValueError("CCA does not support multiple test combinations. Set 'combine_tests' to False.")

        max_components = min(D_data.shape[-1], R_data.shape[-1])
        if n_cca_components > max_components:
            warnings.warn(
                f"'n_cca_components' ({n_cca_components}) exceeds the maximum allowed ({max_components}). "
                f"Adjusting 'n_cca_components' to {max_components}."
            )
            n_cca_components = max_components
    
    if method=="multivariate" and combine_tests is valid_combine_tests[-1]:
            raise ValueError("method is set to 'multivariate' and 'combine_tests' is set to 'across_rows.\n"
                            "The multivariate test will return (1-by-q) p-values, so the test combination can only be across_columns and return a single p-value.\n "
                         "If you want to perform 'combine_tests' while doing 'multivariate' then please set 'combine_tests' to 'True' or 'across_columns.")

    if FWER_correction and combine_tests in [True, "across_columns", "across_rows"]:
       # Raise an exception and stop function execution
        raise ValueError("'FWER_correction' is set to True and 'combine_tests' is either True, 'columns', or 'rows'.\n "
                         "Please set 'FWER_correction' to False if you want to apply 'combine_tests' or set 'combine_tests' to False if you want to run 'FWER_correction'.")
        
    # Get indices of the sessions
    idx_array = get_indices_array(idx_data)

    # Calculate possible permutations
    if FLAG_parametric==0:
        compute_max_permutations(idx_array, permute_within_blocks=False, permute_between_blocks=True, Nnull_samples=Nnull_samples, verbose=verbose)

    # Get input shape information
    n_T, n_N, n_p, n_q, D_data, R_data = get_input_shape(D_data, R_data, verbose)

    # Identify categorical columns in R_data
    category_columns = categorize_columns_by_statistical_method(R_data, method, Nnull_samples, detect_categorical, category_limit, permute_beta)

    # Initialize arrays based on shape of data shape and defined options
    pval, base_statistics, test_statistics_list, F_stats_list, t_stats_list, R2_stats_list = initialize_arrays(n_p, n_q, n_T, method, Nnull_samples, return_base_statistics, combine_tests, n_cca_components)

    # Custom variable names
    if combine_tests is not False:
        test_com_shape = (1, 1) if combine_tests == True else (n_p, 1) if combine_tests == "across_columns" else (1, n_q)
        n_p = test_com_shape[0]
        n_q = test_com_shape[-1]

    predictor_names = [f"State {i+1}" for i in range(n_p)] if predictor_names==[] or len(predictor_names)!=n_p else predictor_names
    outcome_names = [f"Regressor {i+1}" for i in range(n_q)] if outcome_names==[] or len(outcome_names)!=n_q else outcome_names
    
    # Divide the sessions into two dataset to avoid overfit
    train_indices_list, test_indices_list, nan_R =train_test_indices(R_data, D_data, idx_data, method, category_limit) 
    train_indices_list_update = [[] for _ in range(len(train_indices_list))]
    test_indices_list_update = [[] for _ in range(len(train_indices_list))]

    # Removing rows that contain nan-values
    _, _, nan_mask = remove_nan_values(D_data[0,:], R_data, method)
    FLAG_NAN = np.any(nan_mask)  # Keep checking only if NaNs exist at t=0
    if FLAG_NAN:
        # Remove rows with NaN values
        D_data =D_data[:,~nan_mask,:]
        R_data =R_data[~nan_mask,:]
    # # Update indices
    # idx_data_update =update_indices(nan_mask, idx_data) if np.any(nan_mask) else idx_data.copy()

    # Update test and train indices in accordance with NaN values
    for col in range(len(train_indices_list)):
        if FLAG_NAN and np.array_equal(nan_R[col], nan_mask)==False:
            # Keep elements only that are True (i.e., remove False)
            nan_mask =(nan_R[col] | nan_mask)
            # Get the indices corresponding to NaN values
            indices = np.arange(len(nan_mask))
            nan_indices = indices[nan_mask]
            # Update indices due to NaN values in D_matrix
            train_indices_list_update[col], test_indices_list_update[col]= train_test_update_indices(train_indices_list[col], test_indices_list[col], nan_indices)

        else:
            # Just use the original test and train indices
            test_indices_list_update[col] = test_indices_list[col].copy()
            train_indices_list_update[col] = train_indices_list[col].copy()

    for t in tqdm(range(n_T)) if n_T > 1 & verbose==True else range(n_T):
        # If confounds exist, perform confound regression on the dependent variables
        D_t, R_t = deconfound_values(D_data[t, :],R_data, confounds)

        # Create test_statistics and pval_perms based on method
        test_statistics, reg_pinv = initialize_permutation_matrices(method, Nnull_samples, n_p, n_q, D_t, combine_tests, permute_beta, category_columns, n_cca_components=n_cca_components)
        
        # Calculate the beta coefficient of each session
        beta, _, _ = calculate_beta_session(reg_pinv, R_t, D_t, test_indices_list_update, train_indices_list_update)

        for perm in range(Nnull_samples):
            # Calculate the permutation distribution
            stats_results = test_statistics_calculations(D_t, R_t, perm, test_statistics, reg_pinv, method, category_columns, combine_tests, None, permute_beta, beta, test_indices_list_update, Nnull_samples=Nnull_samples, n_cca_components=n_cca_components)
            base_statistics[t, :] = stats_results["base_statistics"] if perm == 0 and stats_results["base_statistics"] is not None else base_statistics[t, :] 
            pval[t, :] = stats_results["pval_matrix"] if perm == 0 and stats_results["pval_matrix"] is not None else pval[t, :]
            F_stats_list[t, perm, :] = stats_results["F_stats"] if stats_results["F_stats"] is not None else F_stats_list[t, perm, :]
            t_stats_list[t, perm, :] = stats_results["t_stats"] if stats_results["t_stats"] is not None else t_stats_list[t, perm, :]
        
        if FLAG_parametric==0:
            # Calculate p-values based on permutations
            pval = get_pval(test_statistics, Nnull_samples, method, t, pval, FWER_correction)
            
            # Output test statistics if it is set to True can be hard for memory otherwise
            if return_base_statistics==True:
                test_statistics_list[t,:] = test_statistics
    
    if Nnull_samples >0 and combine_tests is not False:
        # Set all values to empty lists (except for cases like "all_columns")
        category_columns = {key: [] for key in category_columns}
        category_columns['z_score'] = combine_tests
  
    # Get f and t states when doing a multivariate test
    f_t_stats = get_f_t_stats(D_data, R_data, F_stats_list, t_stats_list, Nnull_samples, n_T) if method =='multivariate' else None
    # Create report summary
    test_summary =create_test_summary(R_data, base_statistics,pval, predictor_names, outcome_names, method, f_t_stats,n_T, n_N, n_p,n_q, Nnull_samples, category_columns, combine_tests)
    Nnull_samples = 0 if Nnull_samples==1 else Nnull_samples    
    category_columns = {key: value for key, value in category_columns.items() if value}
    if len(category_columns)==1:
        category_columns[next(iter(category_columns))]='all_columns'
    elif combine_tests is not False:
        category_columns['z_score'] ='all_columns'
    elif category_columns == {}:
        category_columns['f_reg_cols'] ='all_columns'

    if np.isscalar(pval) is True:
        print("Exporting the base statistics only")
    elif np.sum(np.isnan(pval))>0 & verbose:
        print("Warning: Permutation testing resulted in p-values equal to NaN.\n")
        print("This may indicate an issue with the input data. Please review your data.")
              
    # Return results
    result = {
        'pval': pval,
        'base_statistics': base_statistics,
        'null_stat_distribution': test_statistics_list,
        'test_type': test_type,
        'method': method,
        'combine_tests': combine_tests,
        'max_correction':FWER_correction,
        'statistical_measures': category_columns,
        'Nnull_samples': Nnull_samples,
        'test_summary':test_summary}
    if method =='multivariate' and Nnull_samples>1:
        result['pval_F_multivariate'] = f_t_stats['perm_p_values_F']
        result['pval_t_multivariate'] = f_t_stats['perm_p_values_t']
    return result

 
def test_across_state_visits(D_data, R_data , method="multivariate", Nnull_samples=0, confounds=None, 
                            comparison_statistic ="mean", state_comparison="larger",
                             FWER_correction=False, detect_categorical = False, category_limit=10,
                             vpath_surrogates=None, predictor_names=[], outcome_names=[], return_base_statistics=True, verbose=True):
    """
    Perform permutation testing across state visits.

    This test assesses whether variations in an external signal (`R_data`) 
    correspond to differences in state sequences (`D_data`) derived from the Viterbi path. 
    For example, it can test whether physiological signals (e.g., pupil size, 
    heart rate, or skin conductance in `R_data`) systematically vary depending on 
    the brain state (`D_data`) during rest or task engagement.

    Parameters:
    --------------
    D_data (numpy.ndarray):  
        Viterbi path data, provided as either a 1D or 2D array:  
        - 1D: `(n,)`, contains discrete state labels at each timepoint.  where `n` is the number of timepoints  and `p` is the number of predictors.  
        - 2D: `(n, K)`, One-hot encoded states at each timepoint.
        where `n` is the number of timepoints (or samples, `no. of timepoints × no. of sessions`), and `K` is the number of states.  
    R_data (numpy.ndarray):  
        Continuous physiological signal measurements  
        - 2D: `(n, q)`, where `n` is the number of timepoints (or samples, `no. of timepoints × no. of sessions`) and `q` is the number of recorded signals (e.g., pupil size, heart rate).  
    method (str, optional), default="multivariate":  
        The statistical method used to analyse the relationship between states from the Viterbi path and continuous signals. Valid options are:  
        - `"multivariate"`: Tests whether a states from the Viterbi path explains variation in the signals.  
        - `"univariate"`: Tests whether each state individually explains variation in the signals.  
        - `"cca"`: Stands for Canonical Correlation Analysis; Identifies patterns of shared variation between the Viterbi path and signals, returning a single p-value indicating whether they are linked overall.
        - `"osr"`: Stands for One-State-vs-the-Rest; Compares one state against all others combined to assess its influence on the signal.  
        - `"osa"` Stands for One-State-vs-Another-State; Compares differences between two selected states
    Nnull_samples (int), default=0:                
        Number of generated Viterbi path surrogates used to generate the null distribution, obtained via Monte Carlo resampling.
        Set `Nnull_samples=0` for parametric test output. 
    confounds (numpy.ndarray or None, optional), default=None: 
        The confounding variables to be controlled for, i.e. regressed out from R_data (signal measurements).
        The array should have a shape (n, c), where n is the number of trials and c is the number of confounding variables. 
        Each column represents a different confound to be controlled for in the analysis.
    comparison_statistic (str, optional, default="mean"):  
        Statistic for "osr" and "osa" methods:
        - "mean": Use the mean difference.
        - "median": Use the median difference.
    state_comparison (str, optional), default="larger":  
        Only applies to the "osr" test. Defines how the signal of a given state is compared to the average across other states:
        - `"larger"`: Tests whether the state has a larger signal than the rest.
        - `"smaller"`: Tests whether the state has a smaller signal than the rest.
    FWER_correction (bool, optional), default=False:  
        If `True`, applies family-wise error rate (FWER) correction using the MaxT method to reduce false positives.     
    detect_categorical (bool), default=False:  
        If `True`, automatically detects categorical columns in `R_data` and applies the appropriate statistical test:  
        - Binary categorical variables: Independent t-test.  
        - Multiclass categorical variables: MANOVA for multivariate tests, ANOVA (F-test) for univariate test.  
        - Continuous variables:  F-regression for multivariate tests, Pearson’s t-test for univariate tests.
    category_limit (int), default=10
        Maximum number of unique values allowed to decide whether to use F-ANOVA or F-regression in the F-test.
        Prevents integer-valued variables (e.g., age) from being misclassified as categorical.
    vpath_surrogates (numpy.ndarray, optional):  
        If provided, these surrogate sequences will be used instead of generating new ones.
        - 2D: `(n, q)`, where `n` is the number of timepoints (or samples, `no. of timepoints × no. of sessions`), and `Nnull_samples`is the number of surrogate sequences generated. 
    predictor_names (list of str, optional):  
        Names of the predictor variables in `D_data` (should match column order).  
    outcome_names (list of str, optional):  
        Names of the outcome variables in `R_data`.
    return_base_statistics (bool, optional), default=True:  
        If `True`, stores test statistics from all permutations.  
    verbose (bool, optional):  
        If `True`, prints progress messages.     

    Returns:
    ----------  
    result (dict): 
        The returned dictionary contains:
        'pval': P-values for each test:
            - `"multivariate"`: `(q,)`
            - `"univariate"`: `(p, q)`
            - `"cca"`: `(1)`
            - `"osa"`: `(q,)`
            - `"osr"`: `(p, p)`
        'base_statistics': Test statistics from original/observed data.
        `null_stat_distribution`: Distribution of test statistics under the null hypothesis, where the first row always corresponds to the observed test statistic.
            - `"multivariate"`: `(Nnull_samples, q)`
            - `"univariate"`: `(Nnull_samples, p, q)`
            - `"cca"`: `(Nnull_samples, 1)`
            - `"osa"`: `(Nnull_samples, len(pairwise_comparisons), q)`
            - `"osr"`: `(Nnull_samples, p, q)`
        'test_type': `"test_state_subjects"`.
        'method': Chosen statistical method (`"multivariate"`, `"univariate"`,`"cca"`, `"osa"` or `"osr"`).
        'max_correction': Marks whether FWER correction was applied (`True`/`False`).
        'Nnull_samples': Number of generated Viterbi path surrogates used to generate the null distribution, obtained via Monte Carlo resampling.
        'test_summary': Summary of test results.
        'pval_ftest_multivariate' *(only if `method="multivariate"` and `Nnull_samples > 1`)*: P-values from the F-test in the multivariate analysis.
        'pval_ttest_multivariate' *(only if `method="multivariate"` and `Nnull_samples > 1`)*:  P-values from the t-tests in the multivariate analysis.
    """
    # Initialize variables
    test_type = 'test_across_state_visits'
    method = method.lower()
    FLAG_parametric = 0
    if vpath_surrogates is not None:
        # Define Nnull_samples if vpath_surrogates is provided
        Nnull_samples = vpath_surrogates.shape[-1]

    # Ensure Nnull_samples is at least 1
    if Nnull_samples == 0:
        # Set flag for identifying categories without permutation
        detect_categorical = True
        if method == 'cca' or method =='osr' or method =='osa' and Nnull_samples == 1:
            raise ValueError("'cca', 'osr' and 'osa' does not support parametric statistics. The number of Monto Carlo samples ('Nnull_samples') cannot be set to 1. "
                            "Please increase the number of Monto Carlo samples to perform a valid analysis.")
    
        Nnull_samples +=1
        FLAG_parametric =1
        print("Applying parametric test")

    if not isinstance(detect_categorical,bool):
        raise TypeError("detect_categorical must be a boolean value (True or False)")
    
    if method == "cca" and FWER_correction:
        raise ValueError(
            "FWER correction using MaxT is not applicable when performing CCA, "
            "as only a single p-value is returned. Set FWER_correction=False."
        )         
    # Check if the Viterbi path is correctly constructed
    if vpath_check_2D(D_data) == False:
        raise ValueError(
            "'D_data' is not correctly formatted. Ensure that the Viterbi path is correctly positioned within the target matrix (R). "
            "The data should be one-hot encoded if it's 2D, or an array of integers if it's 1D. "
            "Please verify your input to the 'test_across_state_visits' function."
        )
    # Check validity of method
    valid_state_comparison = ["larger", "smaller"]
    validate_condition(state_comparison.lower() in valid_state_comparison, "Invalid option specified for 'state_comparison'. Must be one of: " + ', '.join(valid_state_comparison))
    
    # Check validity of method
    valid_methods = ["multivariate", "univariate", "cca", "osr", "osa"]
    validate_condition(method.lower() in valid_methods, "Invalid option specified for 'method'. Must be one of: " + ', '.join(valid_methods))
    
    valid_statistic = ["mean", "median"]
    validate_condition(comparison_statistic.lower() in valid_statistic, "Invalid option specified for 'statistic'. Must be one of: " + ', '.join(valid_statistic))
    
    # Convert vpath from matrix to vector
    vpath_array=generate_vpath_1D(D_data)
    if D_data.ndim == 1:
        vpath_bin = np.zeros((len(D_data), len(np.unique(D_data))))
        vpath_bin[np.arange(len(D_data)), D_data - 1] = 1
        D_data = vpath_bin.copy()
    
    # Number of states
    n_states = len(np.unique(vpath_array))
    
    # Get input shape information
    n_T, n_N, n_p, n_q, vpath_data, R_data= get_input_shape(D_data, R_data , verbose)  
    
    # Identify categorical columns in R_data
    category_columns = categorize_columns_by_statistical_method(R_data, method, Nnull_samples, detect_categorical, category_limit,comparison_statistic=comparison_statistic)
    
    # Identify categorical columns
    if category_columns["t_stat_independent_cols"]!=[] or category_columns["f_anova_cols"]!=[]:
        if FWER_correction and (len(category_columns.get('t_stat_independent_cols')) != vpath_data.shape[-1] or len(category_columns.get('f_anova_cols')) != vpath_data.shape[-1]):
            print("Warning: Cannot perform FWER_correction with different test statisticss.\nConsider to set detect_categorical=False")
            raise ValueError("Cannot perform FWER_correction")  

    # Initialize arrays based on shape of data shape and defined options
    pval, base_statistics, test_statistics_list, F_stats_list, t_stats_list, R2_stats_list = initialize_arrays(n_p, n_q,n_T, method, Nnull_samples, return_base_statistics)

    # Custom variable names
    predictor_names = [f"State {i+1}" for i in range(n_states)] if predictor_names==[] or len(predictor_names)!=n_states else predictor_names
    outcome_names = [f"Regressor {i+1}" for i in range(pval.shape[-1])] if outcome_names==[] or len(outcome_names)!=pval.shape[-1] else outcome_names

    # Print tqdm over n_T if there are more than one timepoint
    for t in tqdm(range(n_T)) if n_T > 1 & verbose==True else range(n_T):
        # Correct for confounds and center data_t
        data_t, _ = deconfound_values(R_data,None, confounds)

        # Removing rows that contain nan-values
        if method == "multivariate" or method == "cca":
            if vpath_surrogates is None:
                vpath_array, data_t, _ = remove_nan_values(vpath_array, data_t, method)
            else:
                vpath_surrogates, data_t, _ = remove_nan_values(vpath_surrogates, data_t, method)
        
        if method != "osa":
            ###################### Permutation testing for other tests beside state pairs #################################
            # Create test_statistics and pval_perms based on method
            test_statistics, reg_pinv = initialize_permutation_matrices(method, Nnull_samples, n_p, n_q, vpath_data[0,:,:], category_columns=category_columns)
            # Perform permutation testing
            for perm in tqdm(range(Nnull_samples)) if n_T == 1 & verbose==True else range(n_T):
                # Redo vpath_surrogate calculation if the number of states are not the same (out of 1000 permutations it happens maybe 1-2 times with this demo dataset)
                
                if vpath_surrogates is None:
                    while True:
                        # Create vpath_surrogate
                        vpath_surrogate = surrogate_state_time(perm, vpath_array, n_states)
                        if len(np.unique(vpath_surrogate)) == n_states:
                            break  # Exit the loop if the condition is satisfied
                else:
                    vpath_surrogate = vpath_surrogates[:,perm].astype(int)
                if method =="osr":
                    for state in range(1, n_states+1):
                        test_statistics[perm,state -1,] =calculate_baseline_difference(vpath_surrogate, data_t, state, comparison_statistic.lower(), state_comparison)
                    base_statistics = test_statistics[perm,:] if perm == 0 else base_statistics                 
                elif method =="multivariate":
                    # Make vpath to a binary matrix
                    vpath_surrogate_binary = np.zeros((len(vpath_surrogate), len(np.unique(vpath_surrogate))))
                    # Set the appropriate positions to 1
                    vpath_surrogate_binary[np.arange(len(vpath_surrogate)), vpath_surrogate - 1] = 1
                    stats_results = test_statistics_calculations(vpath_surrogate_binary,data_t, perm,test_statistics, reg_pinv, method, category_columns)
                    base_statistics[t, :] = stats_results["base_statistics"] if perm == 0 and stats_results["base_statistics"] is not None else base_statistics[t, :] 
                    pval[t, :] = stats_results["pval_matrix"] if perm == 0 and stats_results["pval_matrix"] is not None else pval[t, :]
                    if stats_results["t_stats"] is not None:
                        t_stats_list[t,perm,:] = stats_results["t_stats"]
                else:
                    # Univariate test
                    # Apply 1 hot encoding
                    vpath_surrogate_onehot = viterbi_path_to_stc(vpath_surrogate,n_states)
                    # Apply t-statistic on the vpath_surrogate
                    stats_results = test_statistics_calculations(vpath_surrogate_onehot, data_t , perm, test_statistics, reg_pinv, method, category_columns)
                    base_statistics[t, :] = stats_results["base_statistics"] if perm == 0 and stats_results["base_statistics"] is not None else base_statistics[t, :] 
                    pval[t, :] = stats_results["pval_matrix"] if perm == 0 and stats_results["pval_matrix"] is not None else pval[t, :]
            if FLAG_parametric==0: 
                pval = get_pval(test_statistics, Nnull_samples, method, t, pval, FWER_correction)
                pval = np.squeeze(pval, axis=0)
        ###################### Permutation testing for state pairs #################################
        elif method =="osa":
            # Run this code if it is "osa"
            # Generates all unique combinations of length 2 
            pairwise_comparisons = list(combinations(range(1, n_states + 1), 2))
            test_statistics = np.zeros((Nnull_samples, len(pairwise_comparisons), n_q))
            pval = np.zeros((n_states, n_states, n_q))
            # Iterate over pairwise state comparisons
            
            for idx, (state_1, state_2) in tqdm(enumerate(pairwise_comparisons), total=len(pairwise_comparisons), desc="Pairwise comparisons") if verbose ==True else enumerate(pairwise_comparisons):    
                # Generate surrogate state-time data and calculate differences for each permutation
                for perm in range(Nnull_samples):
                    
                    if vpath_surrogates is None:
                        while True:
                            # Create vpath_surrogate
                            vpath_surrogate = surrogate_state_time(perm, vpath_array, n_states)
                            if len(np.unique(vpath_surrogate)) == n_states:
                                break  # Exit the loop if the condition is satisfied
                    else:
                        vpath_surrogate = vpath_surrogates[:,perm].astype(int)
                    test_statistics[perm,idx] = calculate_statepair_difference(vpath_surrogate, data_t, state_1, 
                                                                               state_2, comparison_statistic)
                if FLAG_parametric==0:
                    p_val= np.sum(test_statistics[:,idx] >= test_statistics[0,idx], axis=0) / (Nnull_samples)
                    pval[state_1-1, state_2-1] = p_val
                    pval[state_2-1, state_1-1] = 1 - p_val
            # Fill numbers in base statistics
            if  np.sum(base_statistics[t, :])==0:
                base_statistics[t, :] =test_statistics[0,:]
                
        if return_base_statistics:
            test_statistics_list[t, :] = test_statistics

    # Remove the first dimension if it is 1
    pval =np.squeeze(pval, axis=0) if (pval.ndim >2 and pval.shape[0]==1) else np.squeeze(pval) if method=="osa" else pval
    base_statistics =np.squeeze(base_statistics, axis=(0,1)) if base_statistics.ndim>3 and base_statistics.shape[1] ==Nnull_samples else np.squeeze(base_statistics, axis=0) if base_statistics.shape[0]==1 else base_statistics
    test_statistics_list = np.squeeze(test_statistics_list, axis=(0,1)) if (test_statistics_list.ndim>3 and test_statistics_list.shape[1] == 1) else np.squeeze(test_statistics_list, axis=0)


    # Create report summary
    f_t_stats = get_f_t_stats(D_data, R_data, F_stats_list, t_stats_list, Nnull_samples, n_T) if method =='multivariate' else None

    test_summary =create_test_summary(R_data, base_statistics,pval, predictor_names, outcome_names, method, f_t_stats,n_T, n_N, n_p,n_q, Nnull_samples, category_columns)
    if method =="osr":
        test_summary["state_comparison"] = state_comparison 
    if method =="osa":
        test_summary["pairwise_comparisons"] =pairwise_comparisons
        
    Nnull_samples = 0 if Nnull_samples==1 else Nnull_samples
    category_columns = {key: value for key, value in category_columns.items() if value}
    if len(category_columns)==1:
        category_columns[next(iter(category_columns))]='all_columns'
    elif category_columns=={} and method=='univariate':
        category_columns['t_stat_independent_cols']='all_columns'


    if np.sum(np.isnan(pval))>0 & verbose:
        print("Warning: Permutation testing resulted in p-values equal to NaN.")
        print("This may indicate an issue with the input data. Please review your data.")
        
    # Return results
    result = {
        'pval': pval,
        'base_statistics': base_statistics,
        'null_stat_distribution': test_statistics_list,
        'statistical_measures': category_columns,
        'test_type': test_type,
        'method': method,
        'max_correction':FWER_correction,
        'Nnull_samples': Nnull_samples,
        'test_summary':test_summary}
    if method =='multivariate' and Nnull_samples>1:
        result['pval_F_multivariate'] = f_t_stats['perm_p_values_F']
        result['pval_t_multivariate'] = f_t_stats['perm_p_values_t']
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
        "multivariate", "univariate", "cca", "osr" or "osa". 
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
    nan_mask = None
    if R_data.ndim == 1:
        R_data = R_data.reshape(-1,1) 
    if method == "multivariate":
        # When applying "multivariate" we need to remove rows for our D_data, as we cannot use it as a predictor for
        # Check for NaN values and remove corresponding rows
        nan_mask = np.isnan(np.expand_dims(D_data,axis=1)).any(axis=1) if D_data.ndim==1 else np.isnan(D_data).any(axis=1)
        # nan_mask = np.isnan(D_data).any(axis=1)
        # Get indices or rows that have been removed
        # removed_indices = np.where(nan_mask)[0]
        D_data = D_data[~nan_mask]
        R_data = R_data[~nan_mask]
    elif method== "cca":
        # When applying cca we need to remove rows at both D_data and R_data
        # Check for NaN values and remove corresponding rows
        if D_data.ndim==1:
            D_data = np.expand_dims(D_data, axis=1)
        nan_mask = np.isnan(D_data).any(axis=1) | np.isnan(R_data).any(axis=1)
        # Remove nan indices
        D_data = D_data[~nan_mask]
        R_data = R_data[~nan_mask]
    
        nan_ratio = np.mean(nan_mask)  # Fraction of rows affected
        # if more than 30 % of the data is missing
        if nan_ratio > 0.30:
            warnings.warn(f"High proportion of missing data: {nan_ratio:.1%} of rows contain NaNs in both matrices.")

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
        # if D_data.ndim !=R_data.ndim:
        #     R_data = np.expand_dims(R_data, axis=0)
        n_T, n_ST, n_p = D_data.shape
        n_q = R_data.shape[-1]

    else:
        # Performing permutation testing per timepoint
        if verbose:
            print("performing permutation testing per timepoint")
        n_T, n_ST, n_p = D_data.shape

        # Tile the R_data if it doesn't match the number of timepoints in D_data
        n_q = R_data.shape[-1]
    
    
    return n_T, n_ST, n_p, n_q, D_data, R_data

def process_family_structure(dict_family, Nnull_samples):
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
    Nnull_samples (int): Number of permutations.

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
    # Nnull_samples: number of permutations

    default_values = {
        'file_location' : 'None',
        'M': 'None',
        'CMC': 'False',
        'EE': 'False',
        'nP': Nnull_samples
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
    dict_mfam['nP'] = Nnull_samples
    dict_mfam.setdefault('M', default_values['M'])
    dict_mfam.setdefault('CMC', default_values['CMC'])
    dict_mfam.setdefault('EE', default_values['EE'])
    
    return dict_mfam

def initialize_arrays(n_p, n_q, n_T, method, Nnull_samples, return_base_statistics, combine_tests=False, n_cca_components=None):
    """
    Initializes arrays for permutation testing based on the selected method and test parameters.

    Parameters:
    -----------
    n_p (int): 
        Number of features.
    n_q (int): 
        Number of predictions.
    n_T (int): 
        Number of timepoints.
    method (str): 
        Statistical testing method. Valid options: "multivariate", "cca", "univariate", "osa", "osr".
    Nnull_samples (int): 
        Number of permutations.
    return_base_statistics (bool): 
        If True, returns test statistics values.
    combine_tests (str or bool, optional):       
        Specifies how tests are combined. Valid options:
        - True: Combine across all dimensions.
        - "across_columns": Combine across features (n_p).
        - "across_rows": Combine across predictions (n_q).
        - Default is False (no combination).

    Returns:
    --------
    pval (numpy.ndarray): 
        p-values for the test. Shape varies based on `method` and `combine_tests`:
        - (n_T, 1) if `combine_tests=True`
        - (n_T, n_p) if `combine_tests="across_columns"`
        - (n_T, n_q) if `combine_tests="across_rows"`
        - (n_T, n_p, n_q) for univariate tests without combination
        - (n_T, 1) for CCA
        - (n_T, n_q) for multivariate
        - (n_T, n_p, n_p, n_q) for OSA
        - (n_T, 1, n_p, n_q) for OSR

    base_statistics (numpy.ndarray): 
        Base statistics of a given test, with the same shape as `pval`.

    test_statistics_list (numpy.ndarray or None): 
        Test statistic values, only returned if `return_base_statistics=True`. 
        - Shape (n_T, Nnull_samples, 1), (n_T, Nnull_samples, n_p), or (n_T, Nnull_samples, n_q) based on `combine_tests` for univariate tests.
        - Shape (n_T, Nnull_samples, n_q) for multivariate tests.
        - Shape (n_T, Nnull_samples, 1) for CCA.
        - Shape (n_T, Nnull_samples, len(pairwise_comparisons), n_q) for OSA.
        - Shape (n_T, Nnull_samples, n_p, n_q) for OSR.
        - None if `return_base_statistics=False`.

    F_stats_list (numpy.ndarray): 
        F-statistics for permutation tests. Shape: (n_T, Nnull_samples, n_q).

    t_stats_list (numpy.ndarray): 
        t-statistics for permutation tests. Shape: (n_T, Nnull_samples, n_p, n_q).

    R2_stats_list (numpy.ndarray or None): 
        R² statistics for permutation tests, only available for "multivariate" tests. 
        Shape: (n_T, Nnull_samples, n_q) if `method="multivariate"`, otherwise None.

    """
    
    # Initialize the arrays based on the selected method and data dimensions
    if  method == "multivariate":
        if combine_tests in [True, "across_columns"]: 
            pval = np.zeros((n_T, 1))
            if return_base_statistics:
                test_statistics_list = np.zeros((n_T, Nnull_samples, 1))
            else:
                test_statistics_list= None
            base_statistics= np.zeros((n_T, 1, 1))
        else:
            pval = np.zeros((n_T, n_q))
        
            if return_base_statistics:
                test_statistics_list = np.zeros((n_T, Nnull_samples, n_q))
            else:
                test_statistics_list= None
            base_statistics= np.zeros((n_T, 1, n_q))
        
    elif  method == "cca":
        pval = np.zeros((n_T, n_cca_components))
        if return_base_statistics:
            test_statistics_list = np.zeros((n_T, Nnull_samples, n_cca_components))
        else:
            test_statistics_list= None
        base_statistics= np.zeros((n_T, n_cca_components))        
    elif method == "univariate" :  
        if combine_tests in [True, "across_columns", "across_rows"]: 
            pval_shape = (n_T, 1) if combine_tests == True else (n_T, n_p) if combine_tests == "across_columns" else (n_T, n_q)
            pval = np.zeros(pval_shape)
            base_statistics = pval.copy()
            if return_base_statistics:
                test_statistics_list_shape = (n_T, Nnull_samples, 1) if combine_tests == True else (n_T, Nnull_samples, n_p) if combine_tests == "across_columns" else (n_T, Nnull_samples, n_q)
                test_statistics_list = np.zeros(test_statistics_list_shape)
            else:
                test_statistics_list = None
        else:    
            pval = np.zeros((n_T, n_p, n_q))
            base_statistics = pval.copy()
            if return_base_statistics:    
                test_statistics_list = np.zeros((n_T, Nnull_samples, n_p, n_q))
            else:
                test_statistics_list= None
    elif method == "osa":
        pval = np.zeros((n_T, n_p, n_p, n_q))
        pairwise_comparisons = list(combinations(range(1, n_p + 1), 2))
        if return_base_statistics:    
            test_statistics_list = np.zeros((n_T, Nnull_samples, len(pairwise_comparisons), n_q))
        else:
            test_statistics_list= None
        base_statistics= np.zeros((n_T, len(pairwise_comparisons), n_q))
    elif method == "osr":
        pval = np.zeros((n_T, n_p, n_q))
        if return_base_statistics:
            test_statistics_list = np.zeros((n_T, Nnull_samples, n_p, n_q))
        else:
            test_statistics_list= None
        base_statistics= np.zeros((n_T, n_p, n_q))

    # Create the data to store the t-stats
    t_stats_list = np.zeros((n_T, Nnull_samples, n_p, n_q))
    # Create the data to store the F-stats
    F_stats_list = np.zeros((n_T, Nnull_samples, n_q))
    # Create the data to store the R2-stats
    R2_stats_list = np.zeros((n_T, Nnull_samples, n_q)) if method == "multivariate" else None

    return pval, base_statistics, test_statistics_list, F_stats_list, t_stats_list, R2_stats_list

def handle_nan_values(D_data, R_data, confounds, permutation_matrix, method):
    """
    Check for NaN values at the first time point and update data and the permutation matrix if needed.

    Parameters:
    -----------
    D_data : numpy.ndarray
        Array of shape (n_T, n_samples, n_features) representing the dependent variable data.
    R_data : numpy.ndarray
        Array of shape (n_samples, n_regressors) representing the independent variable data.
    permutation_matrix : numpy.ndarray
        Permutation matrix used for statistical testing.
    method : str
        Statistical method being used. NaN handling is applied only if method is "multivariate" or "cca".

    Returns:
    --------
    D_data : numpy.ndarray
        Updated dependent variable data with NaNs removed if necessary.
    R_data : numpy.ndarray
        Updated independent variable data with NaNs removed if necessary.
    permutation_matrix_update : numpy.ndarray
        Updated permutation matrix if NaNs were found; otherwise, a copy of the original.
    FLAG_NAN : bool
        Indicates whether NaN handling is required in future iterations.
    """
    FLAG_NAN = method in {"multivariate", "cca"}  # Only relevant for these methods
    
    if FLAG_NAN:
        _, _, nan_mask = remove_nan_values(D_data[0, :], R_data, method)
        if np.any(nan_mask):
            permutation_matrix_update = update_permutation_matrix(permutation_matrix, nan_mask)
            # Remove NaN-affected data
            D_data = D_data[:, ~nan_mask, :]
            R_data = R_data[~nan_mask, :]
            confounds = confounds[~nan_mask, :] if confounds is not None else None
        else:
            permutation_matrix_update = permutation_matrix.copy()
            FLAG_NAN = False  # Disable NaN check for future time points
    else:
        permutation_matrix_update = permutation_matrix.copy()  # Default value

    return D_data, R_data, confounds, permutation_matrix_update

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
    Regress out confounds from the input data array D_data, and optionally from R_data if provided.
    This function deconfounds D_data by regressing out the effect of confounds. If R_data is provided, 
    it also deconfounds R_data. If confounds are not provided, it returns the centered versions of D_data 
    and R_data (if R_data is not None).

    Parameters:
    --------------
    D_data (numpy.ndarray): 
        The independent variable matrix. Shape: (n, p), where n is the number of observations and 
        p is the number of predictors.  
    R_data (numpy.ndarray or None): 
        The dependent variable matrix. Shape: (n, q), where q is the number of dependent variables.
        If None, only D_data will be deconfounded and returned.  
    confounds (numpy.ndarray or None, optional): 
        The confounds matrix. Shape: (n, k), where k is the number of confounding variables. Default is None.

    Returns:
    ----------
    D_t (numpy.ndarray): 
        The deconfounded D_data with confounds regressed out. Shape: (n, p).
        
    R_t (numpy.ndarray or None): 
        The deconfounded R_data with confounds regressed out, if R_data is provided. Shape: (n, q) or None 
        if R_data was not provided.
    """
    # Center D_data
    D_data_centered = D_data - np.nanmean(D_data, axis=0)

    if R_data is not None:
        # Center R_data
        R_data_centered = R_data - np.nanmean(R_data, axis=0)
    else:
        R_data_centered = None

    if confounds is not None:
        # Center confounds
        confounds_centered = confounds - np.nanmean(confounds, axis=0)

        # Check for NaNs in D_data and confounds
        nan_in_D = np.isnan(D_data).any()
        nan_in_confounds = np.isnan(confounds).any()

        # Fast path if no NaNs are present
        if not nan_in_D and not nan_in_confounds:
            D_t = D_data_centered - confounds_centered @ np.linalg.pinv(confounds_centered) @ D_data_centered
            if R_data_centered is not None:
                R_t = R_data_centered - confounds_centered @ np.linalg.pinv(confounds_centered) @ R_data_centered
            else:
                R_t = None
        else:
            # Initialize outputs with NaNs
            D_t = np.full_like(D_data, np.nan)
            R_t = np.full_like(R_data, np.nan) if R_data is not None else None

            # Column-wise regression for D_data
            for i in range(D_data.shape[1]):
                valid_indices = ~np.isnan(D_data[:, i]) & ~np.isnan(confounds).any(axis=1)
                if np.any(valid_indices):
                    confounds_valid = confounds_centered[valid_indices]
                    D_t[valid_indices, i] = (
                        D_data_centered[valid_indices, i]
                        - confounds_valid @ np.linalg.pinv(confounds_valid) @ D_data_centered[valid_indices, i]
                    )

            # Column-wise regression for R_data (if provided)
            if R_data_centered is not None:
                for i in range(R_data.shape[1]):
                    valid_indices = ~np.isnan(R_data[:, i]) & ~np.isnan(confounds).any(axis=1)
                    if np.any(valid_indices):
                        confounds_valid = confounds_centered[valid_indices]
                        R_t[valid_indices, i] = (
                            R_data_centered[valid_indices, i]
                            - confounds_valid @ np.linalg.pinv(confounds_valid) @ R_data_centered[valid_indices, i]
                        )
    else:
        # If confounds are not provided, return centered data
        D_t = D_data_centered
        R_t = R_data_centered

    return D_t, R_t

def deconfound_values(D_data, R_data, confounds=None):
    """
    Deconfound the variables R_data and D_data for permutation testing.

    Parameters:
    --------------
    D_data  (numpy.ndarray): 
        The input data array.
    R_data (numpy.ndarray or None): 
        The second input data array, default= None.
        If None, assumes we are working across visits, and R_data represents the Viterbi path of a sequence.
    confounds (numpy.ndarray or None): 
        The confounds array, default= None.

    Returns:
    ----------  
    D_t (numpy.ndarray): 
        D_data with confounds regressed out.
    R_t (numpy.ndarray): 
        R_data with confounds regressed out.
    """
    # Center D_data and R_data
    D_data_centered = D_data - np.nanmean(D_data, axis=0)

    if R_data is not None:
        # Center R_data
        R_data_centered = R_data - np.nanmean(R_data, axis=0)
    

    if confounds is not None:
        if confounds.ndim == 1:
            confounds = np.expand_dims(confounds, axis=1)
        # Center confounds
        confounds_centered = confounds - np.nanmean(confounds, axis=0)

        # Check for NaNs in D_data, R_data, or confounds
        nan_in_D = np.isnan(D_data).any()
        nan_in_confounds = np.isnan(confounds).any()
        nan_in_R = np.isnan(R_data).any() if R_data is not None else False

        if not nan_in_D and not nan_in_R and not nan_in_confounds:
            # Fast matrix operation when no NaNs are present
            D_t = D_data_centered - confounds_centered @ np.linalg.pinv(confounds_centered) @ D_data_centered
            # Center R_data
            if R_data is not None: 
                R_t = R_data_centered - confounds_centered @ np.linalg.pinv(confounds_centered) @ R_data_centered
            
        else:
            # Initialize outputs with NaNs
            D_t = np.full_like(D_data, np.nan)

            # Column-wise regression for D_data
            for i in range(D_data.shape[1]):
                valid_indices = ~np.isnan(D_data[:, i]) & ~np.isnan(confounds).any(axis=1)
                if np.any(valid_indices):  # Ensure valid indices exist
                    confounds_valid = confounds_centered[valid_indices]
                    D_t[valid_indices, i] = (
                        D_data_centered[valid_indices, i]
                        - confounds_valid @ np.linalg.pinv(confounds_valid) @ D_data_centered[valid_indices, i]
                    )
                    
                    
            if R_data is not None:
                # Initialize outputs with NaNs
                R_t = np.full_like(R_data, np.nan)

                # Column-wise regression for R_data
                for i in range(R_data.shape[1]):
                    valid_indices = ~np.isnan(R_data[:, i]) & ~np.isnan(confounds).any(axis=1)
                    if np.any(valid_indices):  # Ensure valid indices exist
                        confounds_valid = confounds_centered[valid_indices]
                        R_t[valid_indices, i] = (
                            R_data_centered[valid_indices, i]
                            - confounds_valid @ np.linalg.pinv(confounds_valid) @ R_data_centered[valid_indices, i]
                        )
            else:
                R_t = None
    else:
        # If confounds are not provided, return centered data
        D_t = D_data_centered
        R_t = R_data_centered if R_data is not None else R_data
    return D_t, R_t

def initialize_permutation_matrices(method, Nnull_samples, n_p, n_q, D_data, combine_tests=False, permute_beta=False, category_columns=None, n_cca_components=1):
    """
    Initializes the permutation matrices and prepare the regularized pseudo-inverse of D_data.

    Parameters:
    --------------
    method (str): 
        The method to use for permutation testing.
    Nnull_samples (int): 
        The number of permutations.
    n_p (int): 
        The number of features.
    n_q (int): 
        The number of predictions.
    D_data (numpy.ndarray): 
        The independent variable.
    combine_tests (str), default=False:       
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
        if combine_tests in [True, "across_columns", "across_rows"]: 
            test_statistics_shape = (Nnull_samples, 1) if combine_tests == True else (Nnull_samples, n_p) if combine_tests == "across_columns" else (Nnull_samples, n_q)
            test_statistics = np.zeros(test_statistics_shape)
        else:
            # Initialize test statistics output matrix based on the selected method
            test_statistics = np.zeros((Nnull_samples, n_p, n_q))
        if permute_beta or category_columns['f_reg_cols']!=[]:
            if np.isnan(D_data).any():
                # Impute NaN values with column-wise mean
                col_means = np.nanmean(D_data, axis=0)
                nan_indices = np.where(np.isnan(D_data))
                D_data_nan = D_data.copy()
                D_data_nan[nan_indices] = np.take(col_means, nan_indices[1])
                # Set the regularization parameter
                regularization = 0.001
                # Create a regularization matrix (identity matrix scaled by the regularization parameter)
                regularization_matrix = regularization * np.eye(D_data_nan.shape[1])
                # Compute the regularized pseudo-inverse of D_data
                reg_pinv = np.linalg.inv(D_data_nan.T @ D_data_nan + regularization_matrix) @ D_data_nan.T

            else:
                # Set the regularization parameter
                regularization = 0.001
                # Create a regularization matrix (identity matrix scaled by the regularization parameter)
                regularization_matrix = regularization * np.eye(D_data.shape[1])  # Regularization term for Ridge regression
                # Compute the regularized pseudo-inverse of D_data
                reg_pinv = np.linalg.inv(D_data.T @ D_data + regularization_matrix) @ D_data.T 


    elif method =="cca":
        # Initialize test statistics output matrix based on the selected method
        test_statistics = np.zeros((Nnull_samples, n_cca_components))
        # Define regularization parameter
        regularization_parameter = 0.001
        # Create a regularization matrix (identity matrix scaled by the regularization parameter)
        regularization_matrix = regularization_parameter * np.eye(D_data.shape[1]) 
        # Compute the regularized pseudo-inverse
        reg_pinv = np.linalg.inv(D_data.T @ D_data + regularization_matrix) @ D_data.T  

    elif method =="osr":
        # Initialize test statistics output matrix based on the selected method
        test_statistics = np.zeros((Nnull_samples, n_p, n_q))
    else:
        # multivariate
        if combine_tests in [True, "across_columns", "across_rows"]:
            test_statistics = np.zeros((Nnull_samples, 1))
        else:
            # Regression got a N by q matrix 
            test_statistics = np.zeros((Nnull_samples, n_q))

        # Define regularization parameter
        regularization_parameter = 0.001
        # Create a regularization matrix (identity matrix scaled by the regularization parameter)
        regularization_matrix = regularization_parameter * np.eye(D_data.shape[1]) 
        # Compute the regularized pseudo-inverse
        reg_pinv = np.linalg.inv(D_data.T @ D_data + regularization_matrix) @ D_data.T  
        

    return test_statistics, np.array(reg_pinv)

def permutation_matrix_across_subjects(Nnull_samples, R_data):
    """
    Generates a normal permutation matrix with the assumption that each index is independent across subjects. 

    Parameters:
    --------------
    Nnull_samples (int): 
        The number of permutations.
    R_data (numpy.ndarray): 
        R-matrix at timepoint 't'
        
    Returns:
    ----------  
    permutation_matrix (numpy.ndarray): 
        Permutation matrix of subjects it got a shape (n_ST, Nnull_samples)
    """
    # Count the maximum column with non-Nan values
    non_nan_count = np.sum(~np.isnan(R_data), axis=0)
    max_length_idx = np.argmax(non_nan_count) 
    max_length = non_nan_count[max_length_idx]
    permutation_matrix = np.zeros((max_length,Nnull_samples), dtype=int)
    for perm in range(Nnull_samples):
        if perm == 0:
            permutation_matrix[:,perm] = np.arange(max_length)
        else:
            permutation_matrix[:,perm] = np.random.permutation(max_length)
    return permutation_matrix

def get_pval(test_statistics, Nnull_samples, method, t, pval, FWER_correction=False):
    """
    Computes p-values from the test statistics.
    # Ref: https://github.com/OHBA-analysis/HMM-MAR/blob/master/utils/testing/permtest_aux.m

    Parameters:
    --------------
    test_statistics (numpy.ndarray): 
        The permutation array.
    pval_perms (numpy.ndarray): 
        The p-value permutation array.
    Nnull_samples (int): 
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
    if method == "multivariate" or method == "osr":
        if FWER_correction:
            # Perform family-wise permutation correction
            # Compute the maximum statistic for each permutation (excluding the first row)
            max_test_statistics = np.max(test_statistics[1:], axis=1)  # Shape: (Nnull_samples,)

            # Count how many times MaxT statistics exceed or equal each observed statistic
            # Adding 1 to numerator and denominator for bias correction
            pval[t, :] = (np.sum(max_test_statistics[:, np.newaxis] >= test_statistics[0, :], axis=0) + 1) / (Nnull_samples + 1)
            
        else:
            # Count how many times test_statistics exceed or equal each observed statistic
            # Adding 1 for bias correction
            pval[t, :] = (np.sum(test_statistics[:] >= test_statistics[0,:], axis=0)) / (Nnull_samples)
        
    elif method == "univariate" or method =="cca":
        if FWER_correction:
            # Perform family-wise permutation correction
            # Calculate the MaxT statistics for each permutation (excluding the observed)
            # The empirical distribution of the maximum test statistics does not include the observed statistics
            maxT_statistics = np.max(np.abs(test_statistics[1:, :, :]), axis=(1, 2))  # Shape: (Nnull_samples - 1,)

            # Extract the observed test statistics (first row)
            observed_test_stats = np.abs(test_statistics[0, :, :])  # Shape: (p_dim, q_dim)

            # Use broadcasting to compare observed statistics against MaxT statistics
            # Expand dimensions for broadcasting
            observed_expanded = observed_test_stats[np.newaxis, :, :]  # Shape: (1, p_dim, q_dim)
            maxT_expanded = maxT_statistics[:, np.newaxis, np.newaxis]  # Shape: (Nnull_samples - 1, 1, 1)

            # Count how many times MaxT statistics exceed or equal each observed statistic
            # Adding 1 to numerator and denominator for bias correction
            pval[t, :, :] = (np.sum(maxT_expanded >= observed_expanded, axis=0) + 1) / (Nnull_samples + 1)  # Shape: (p_dim, q_dim)
            
        else:    
            # Count how many times test_statistics exceed or equal each observed statistic
            # Adding 1 for bias correction
            pval[t, :] = (np.sum(test_statistics[:] >= test_statistics[0,:], axis=0)) / (Nnull_samples)
    
    return pval

def get_f_t_stats(D_data, R_data, F_stats_list, t_stats_list, Nnull_samples, n_T):
    """
    Compute F and t statistics along with permutation-based p-values and confidence intervals.

    This function calculates F and t statistics from input data and, if permutations are used, 
    estimates p-values and confidence intervals by comparing observed statistics to permuted values.

    Parameters:
    --------------
    D_data (numpy.ndarray):  
        Predictor data of shape `(n, p)`, where:  
        - `n` is the number of samples (e.g., trials, subjects).  
        - `p` is the number of predictor variables.  
    R_data (numpy.ndarray):  
        Outcome data of shape `(n, q)`, where:  
        - `n` is the number of samples.  
        - `q` is the number of outcome variables.  
    F_stats_list (numpy.ndarray):  
        Array of F-statistics across permutations, with shape `(n_T, Nnull_samples + 1, q)`.  
        The first index (`[:, 0, :]`) contains observed statistics, while remaining indices contain permuted values.  
    t_stats_list (numpy.ndarray):  
        Array of t-statistics across permutations, with shape `(n_T, Nnull_samples + 1, p, q)`.  
        The first index (`[:, 0, :, :]`) contains observed statistics, while remaining indices contain permuted values.  
    Nnull_samples (int):  
        Number of permutations used in the test.  
    n_T (int):  
        Number of timepoints or conditions.  

    Returns:
    --------------  
    f_t_stats (dict):  
        A dictionary containing the computed statistics:  
        - `'perm_p_values_F'` (numpy.ndarray): Permutation-based p-values for F-statistics,  
          with shape `(n_T, q)`.  
        - `'perm_p_values_t'` (numpy.ndarray): Permutation-based p-values for t-statistics,  
          with shape `(n_T, p, q)`.  
        - `'perm_ci_lower'` (numpy.ndarray): Lower bound of the 95% confidence interval for t-statistics,  
          with shape `(n_T, p, q)`.  
        - `'perm_ci_upper'` (numpy.ndarray): Upper bound of the 95% confidence interval for t-statistics,  
          with shape `(n_T, p, q)`.  
        - `'F_stats'` (numpy.ndarray): Observed F-statistics, with shape `(n_T, q)`.  
        - `'t_stats'` (numpy.ndarray): Observed t-statistics, with shape `(n_T, p, q)`.  
    """
    if Nnull_samples >1:
        n_q = R_data.shape[-1]
        n_p = D_data.shape[-1]
        if n_T>1 and (n_p==1 or n_q==1) :
            perm_p_values_F = np.zeros((n_T,n_q))
            perm_p_values_t = np.zeros((n_T,n_p,n_q))
            perm_ci_lower = np.zeros((n_T,n_p,n_q))
            perm_ci_upper = np.zeros((n_T,n_p,n_q))
            t_stats = np.zeros((n_T,n_p,n_q))
            F_stats =[]
            t_stats =[]
            for t in range(n_T):
                perm_p_values_F[t,:] = (np.sum(F_stats_list[t,:] >= F_stats_list[t,0,:], axis=0)) / (Nnull_samples)
                perm_p_values_t[t,:] = (np.sum(np.abs(t_stats_list[t,:]) >= np.abs(t_stats_list[t,0,:]), axis=0)) / (Nnull_samples)
                perm_ci_lower[t,:] = np.percentile(t_stats_list[t,1:], 2.5, axis=0)
                perm_ci_upper[t,:] = np.percentile(t_stats_list[t,1:], 97.5, axis=0)
                F_stats.append(F_stats_list[t,0])
                t_stats.append(t_stats_list[t,0])
            F_stats = np.squeeze(np.array(F_stats))
            t_stats = np.array(t_stats)
        else: 
            perm_p_values_F = (np.sum(F_stats_list[0,1:] >= F_stats_list[0,0], axis=0)) / (Nnull_samples)
            perm_p_values_t = (np.sum(np.abs(t_stats_list[0,1:]) >=np.abs(t_stats_list[0,0]), axis=0)) / (Nnull_samples)
            perm_ci_lower = (np.percentile(t_stats_list[0,1:], 2.5, axis=0))
            perm_ci_upper = (np.percentile(t_stats_list[0,1:], 97.5, axis=0))
            F_stats = F_stats_list[0,0]
            t_stats = t_stats_list[0,0]

        f_t_stats ={
            'perm_p_values_F': perm_p_values_F, 
            'perm_p_values_t':perm_p_values_t, 
            'perm_ci_lower': perm_ci_lower, 
            'perm_ci_upper': perm_ci_upper, 
            'F_stats': F_stats, 
            't_stats' : t_stats
            }
    else:
        # Parametric values
        f_t_stats ={
            'F_stats': np.squeeze(F_stats_list,axis=1), 
            't_stats' : np.squeeze(t_stats_list,axis=1)
            }

    return f_t_stats

def permutation_matrix_across_trials_within_session(Nnull_samples, R_data, idx_array, trial_timepoints=None):
    """
    Generates permutation matrix of within-session across-trial data based on given indices.

    Parameters:
    --------------
    Nnull_samples (int): 
        The number of permutations.
    R_data (numpy.ndarray): 
        The preprocessed data array.
    idx_array (numpy.ndarray): 
        The indices array.
    trial_timepoints (int): 
        Number of timepoints for each trial, default = None

    Returns:
    ----------  
    permutation_matrix (numpy.ndarray): 
        Permutation matrix of subjects it got a shape (n_ST, Nnull_samples)
    """
    # Perform within-session between-trial permutation based on the given indices

    R_t = R_data.copy()
    
    permutation_matrix = np.zeros((R_t.shape[0], Nnull_samples), dtype=int)
    idx_array = idx_array-1 if np.min(idx_array)==1 else idx_array
    unique_indices = np.unique(idx_array)

    for perm in range(Nnull_samples):
        if perm == 0:
            permutation_matrix[:,perm] = np.arange(R_t.shape[0])
        else:
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

def permute_subject_trial_idx(idx_array, unique_values=None, value_counts=None):
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
    # # Get unique values and their counts
    # unique_values, value_counts = np.unique(idx_array, return_counts=True)
    if np.all(np.diff(value_counts) ==0):
        # Permute the unique values
        permuted_unique_values = np.random.permutation(unique_values)

        # Repeat each unique value according to its original count
        permuted_array = np.repeat(permuted_unique_values, value_counts)

    else:
        # Identify groups of unique values that share the same count
        count_to_values = {}
        for unique_val, count in zip(unique_values, value_counts):
            count_to_values.setdefault(count, []).append(unique_val)

        # Permute within each group of the same value count
        permuted_array = np.zeros_like(idx_array)
        for count, values in count_to_values.items():
            permuted_values = np.random.permutation(values)  # Shuffle only within groups of the same count
            mask = np.isin(idx_array, values)  # Identify positions of these values
            permuted_array[mask] = np.repeat(permuted_values, count)[:np.sum(mask)]  # Assign permuted values

    # # Verify the counts remain the same after permutation
    # original_counts = dict(zip(unique_values, value_counts))
    # permuted_unique_values, permuted_counts = np.unique(permuted_idx_array, return_counts=True)
    # permuted_counts_dict = dict(zip(permuted_unique_values, permuted_counts)
    return permuted_array


def permutation_matrix_within_subject_across_sessions(Nnull_samples, R_data, idx_array):
    """
    Generates permutation matrix of within-session across-session data based on given indices.

    Parameters:
    --------------
    Nnull_samples (int): 
        The number of permutations.
    R_data (numpy.ndarray): 
        The preprocessed data array.
    idx_array (numpy.ndarray): 
        The indices array.


    Returns:
    ----------  
    permutation_matrix (numpy.ndarray): 
        The within-session continuos indices array.
    """
    # Count the maximum column with non-Nan values
    non_nan_count = np.sum(~np.isnan(R_data), axis=0)
    max_length_idx = np.argmax(non_nan_count) 
    max_length = non_nan_count[max_length_idx]
    permutation_matrix = np.zeros((max_length,Nnull_samples), dtype=int)
    idx_array_update = idx_array[~np.isnan(R_data[:,max_length_idx])] -1 if np.min(idx_array)==1 else idx_array[~np.isnan(R_data[:,max_length_idx])]
    # Get unique values and their counts
    unique_values, value_counts = np.unique(idx_array_update, return_counts=True)

    # Identify unique groups per count category
    count_to_values = {}
    for unique_val, count in zip(unique_values, value_counts):
        count_to_values.setdefault(count, []).append(unique_val)

    # Compute log factorial sum safely without computing factorial directly
    log_total_permutations = sum(sum(math.log(i) for i in range(1, len(values) + 1)) for values in count_to_values.values())

    # Extract the exponent
    exponent = int(log_total_permutations)

    # Compute the coefficient (mantissa) for better readability
    coefficient = 10 ** (log_total_permutations - exponent)

    # Format output professionally
    exp_notation = f"{coefficient:.2f} × 10^{exponent}"

    # Print the result in a clean, professional format
    print(f"Total possible permutations: {exp_notation}")


    for perm in range(Nnull_samples):
        if perm == 0:
            permutation_matrix[:,perm] = np.arange(max_length)
        else:
            idx_array_perm = permute_subject_trial_idx(idx_array_update, unique_values, value_counts)
            unique_indices = np.unique(idx_array_perm)
            positions_permute = [np.where(np.array(idx_array_perm) == i)[0] for i in unique_indices]
            permutation_matrix[:,perm] = np.concatenate(positions_permute,axis=0)
    return permutation_matrix

def permutation_matrix_within_and_between_groups(Nnull_samples, R_data, idx_array):
    """
    Generates a permutation matrix with permutations within and between groups.
    
    Parameters:
    --------------
    Nnull_samples (int): 
        The number of permutations.
    R_data (numpy.ndarray): 
        The R_data array used for generating the within-group permutation matrix.
    idx_array (numpy.ndarray): 
        The indices array that groups the R_data (e.g., session or subject identifiers).
    
    Returns:
    ----------
    permutation_matrix (numpy.ndarray): 
        The matrix with both within- and between-group permutations.
    """

    # Count the maximum column with non-Nan values
    non_nan_count = np.sum(~np.isnan(R_data), axis=0)
    max_length_idx = np.argmax(non_nan_count) 
    max_length = non_nan_count[max_length_idx]
    permutation_matrix = np.zeros((max_length,Nnull_samples), dtype=int)
    idx_array_update = idx_array[~np.isnan(R_data[:,max_length_idx])] -1 if np.min(idx_array)==1 else idx_array[~np.isnan(R_data[:,max_length_idx])]
    unique_indices = np.unique(idx_array_update)

    R_t = R_data[~np.isnan(R_data)[:,max_length_idx],:] # Getting the max rows when encounting for NaN values
    # Generate the within-group permutation matrix
    permutation_matrix_within_group = permutation_matrix_across_trials_within_session(
        Nnull_samples, R_t, idx_array_update)
    
    # Initialize the permutation matrix for within and between groups
    permutation_within_and_between_group = permutation_matrix_within_group.copy()
    # Get unique values and their counts
    unique_values, value_counts = np.unique(idx_array_update, return_counts=True)
    for perm in range(Nnull_samples):
        if perm == 0:
            # The first column is just the identity permutation (no change)
            permutation_within_and_between_group[:, perm] = np.arange(R_t.shape[0])
        else:
            # Permute the idx_array to shuffle the groupings
            idx_array_perm = permute_subject_trial_idx(idx_array_update, unique_values, value_counts)
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

    return vpath_array.astype(np.int8)



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
            viterbi_path_surrogate = np.squeeze(viterbi_path.copy().astype(np.int8))
        else:
            viterbi_path_surrogate = viterbi_path.copy().astype(np.int8)
            
    else:
        viterbi_path_surrogate = surrogate_viterbi_path(viterbi_path, n_states)
    return viterbi_path_surrogate


def surrogate_state_time_matrix(Nnull_samples, vpath_data, n_states):
    """
    Generate a matrix of surrogate Viterbi paths by permuting state assignments 
    while ensuring that all states are represented in each permutation.

    Parameters:
    --------------
    Nnull_samples (int):
        The number of surrogate permutations to generate.
    vpath_data (numpy.ndarray):
        The original Viterbi path data, representing the state sequence.
    n_states (int):
        The total number of states.

    Returns:
    ----------  
    vpath_surrogates (numpy.ndarray):
        A 2D array of shape (len(vpath_data), Nnull_samples), where each column is a 
        surrogate Viterbi path that maintains the segment structure and 
        ensures all states are included.
    """  
    vpath_array=generate_vpath_1D(vpath_data)
    vpath_surrogates = np.zeros((len(vpath_array),Nnull_samples), dtype=np.int8)
    for perm in tqdm(range(Nnull_samples)):
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
    stc = np.zeros((len(viterbi_path), n_states), dtype=np.int8)
    if np.min(viterbi_path)==0:
        stc[np.arange(len(viterbi_path)), viterbi_path] = 1
    else:
        stc[np.arange(len(viterbi_path)), viterbi_path-1] = 1
    return stc

def surrogate_viterbi_path(viterbi_path, n_states):
    """
    Generate a surrogate Viterbi path while keeping the segment 
    structure intact. Each segment (continuous run of the same state) is 
    reassigned to a new state, ensuring that no two consecutive segments 

    Parameters:
    --------------
    viterbi_path (numpy.ndarray):   
        1D array representing the original Viterbi path with unique state segments.
    n_states (int):                
        The total number of states.

    Returns:
    ----------  
    viterbi_path_surrogate (numpy.ndarray): 
        A 1D array with the same segmentation pattern but reassigned states, ensuring 
        that no segment is mapped to the same state as the previous one.
    """
    # Detect segment boundaries (where the state changes)
    viterbi_path = np.squeeze(viterbi_path) if np.ndim(viterbi_path) == 2 and viterbi_path.shape[1] ==1 else viterbi_path
    segment_start_indices = np.where(np.diff(viterbi_path) != 0)[0] + 1
    segment_start_indices = np.insert(segment_start_indices, 0, 0)  # Include the first index

    # Extract unique states and shuffle them for reassignment
    original_states = np.unique(viterbi_path)
    # Assign states ensuring no consecutive segments have the same value
    viterbi_path_surrogate = np.zeros_like(viterbi_path, dtype=np.int8)
    last_state = None
    available_states = original_states.copy()
    # shuffled_states = original_states.copy()
    
    # # Ensure shuffled states are different from the original sequence
    # while np.any(shuffled_states == original_states):
    #     np.random.shuffle(shuffled_states)

    # available_states = shuffled_states.copy()
    
    for i, start in enumerate(segment_start_indices):
        end = segment_start_indices[i + 1] if i + 1 < len(segment_start_indices) else len(viterbi_path)

        # Choose a new state that is different from the last assigned state
        possible_states = available_states[available_states != last_state]
        new_state = np.random.choice(possible_states)
        
        # Assign the new state to the segment
        viterbi_path_surrogate[start:end] = new_state
        last_state = new_state

    return viterbi_path_surrogate
    
def calculate_baseline_difference(vpath_array, R_data, state, comparison_statistic, state_comparison):
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
    comparison_statistic (str)             
        The chosen statistic to be calculated. Valid options are "mean" or "median".

    Returns:
    ----------  
    difference (float)            
        The calculated difference between the specified state and all other states combined.
    """
    if comparison_statistic == 'median':
        # Calculate the median for the specific state
        state_R_data = np.nanmedian(R_data[vpath_array == state])
        # Calculate the median for all other states
        other_R_data = np.nanmedian(R_data[vpath_array != state])
    elif comparison_statistic == 'mean':
        # Calculate the mean for the specific state
        state_R_data = np.nanmean(R_data[vpath_array == state], axis=0)
        # Calculate the mean for all other states
        other_R_data = np.nanmean(R_data[vpath_array != state], axis=0)
    else:
        raise ValueError("Invalid stat value")
    # Detect any difference
    # difference = np.abs(state_R_data) - np.abs(other_R_data)
    if state_comparison=="larger":
        difference = state_R_data - other_R_data
    else:
        difference =  other_R_data - state_R_data
    
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
        state_1_R_data = np.nanmean(R_data[vpath_array == state_1], axis=0)
        state_2_R_data = np.nanmean(R_data[vpath_array == state_2], axis=0)
    elif stat == 'median':
        state_1_R_data = np.nanmedian(R_data[vpath_array == state_1], axis=0)
        state_2_R_data = np.nanmedian(R_data[vpath_array == state_2], axis=0)
    else:
        raise ValueError("Invalid stat value")
    # Detect any difference
    difference = state_1_R_data - state_2_R_data
    return difference

def compute_max_permutations(block_indices= None, permute_within_blocks = False, permute_between_blocks = False, Nnull_samples = None, permute_beta = False, verbose=True):
    """
    Compute the maximum number of possible permutations based on the selected permutation strategy.

    Parameters:
    --------------
    block_indices (numpy.ndarray), default=None:
        A 1D array indicating block membership for each sample.
        Required when using block-based permutations.
    permute_within_blocks (bool), default=False:
        If `True`, subjects can permute within their respective blocks.
    permute_between_blocks (bool), default=False:
        If `True`, blocks of the same size can swap positions.
    Nnull_samples (int), default=None:
        The total number of independent samples for raw permutation.
        Required when `permute_within_blocks=False` and `permute_between_blocks=False`.
    permute_beta (bool), default=False:
        If True, permutations are performed at the beta level (e.g., shuffling session-level betas).
        Assumes one beta per unique block label.
    verbose (bool), default=True:
        If True, prints the total number of possible permutations and how many will be used.


    Returns:
    --------------
    log_permutations (float):
        The logarithm of the total number of possible permutations.
    max_permutations (float):
        The total number of possible permutations in exponential form 
        (if computable without overflow).
    
    Notes:
    --------------
    - If all permutation flags are False, the function assumes raw permutation across `Nnull_samples`.
    - If `permute_within_blocks` is True, it computes the number of permutations possible within each block.
    - If `permute_between_blocks` is True, it computes permutations between blocks of equal size.
    - If both `permute_within_blocks` and `permute_between_blocks` are True, both effects are combined.
    - If `permute_beta` is True, it treats each unique entry in `block_indices` as a permutable beta unit.
    - Output formatting uses standard scientific notation (e.g., '1.30e+12') and truncates mantissas instead of rounding.
    """

    # Raw Permutation (Independent Samples)
    if not permute_within_blocks and not permute_between_blocks and not permute_beta:
        if Nnull_samples is None:
            raise ValueError("Nnull_samples must be provided for raw permutation.")

        # Compute log(N!) to prevent overflow
        log_permutations = sum(math.log(k) for k in range(1, Nnull_samples + 1))

    elif block_indices is not None:

        if permute_beta:
            # Permutation across beta coefficients
            log_permutations = sum(math.log(k) for k in range(1, len(np.unique(block_indices)) + 1))

        else:
            # Count occurrences of each block index to get block sizes
            block_sizes = Counter(block_indices)

            # Initialize log_permutations
            log_permutations = 0
            # Permutation Within Blocks
            if permute_within_blocks:
                log_permutations += sum(math.log(math.factorial(size)) for size in block_sizes.values())

            # Permutation Between Blocks
            if permute_between_blocks:
                size_counts = Counter(block_sizes.values())  # Count of blocks per size
                num_sessions =sum(size_counts.values())
                log_permutations += math.log(math.factorial(num_sessions))

    else:
        raise ValueError("block_indices must be provided when using block-based permutations.")

    # Convert Log to Exponential if Feasible
    try:
        possible_permutations = math.exp(log_permutations)
        if possible_permutations < 1e6:
            formatted_possible = f"{int(possible_permutations):,}"
        else:
            log10_perm = math.log10(possible_permutations)
            mantissa = 10 ** (log10_perm - int(log10_perm))
            truncated_mantissa = int(mantissa * 100) / 100
            formatted_possible = f"{truncated_mantissa:.2f}e+{int(log10_perm)}"
    except OverflowError:
        # If overflow, format using base-10 exponent as scientific notation
        log10_perm = log_permutations / math.log(10)
        base10_approx = 10 ** (log10_perm - int(log10_perm))  # get the mantissa
        truncated_mantissa = int(base10_approx * 100) / 100
        formatted_possible = f"{truncated_mantissa:.2f}e+{int(log10_perm)}"
        possible_permutations = float('inf')  # Use inf for internal logic

    # Step Ensure the number of requested permutations does not exceed possible permutations
    if Nnull_samples is not None:
        final_permutations = min(possible_permutations, Nnull_samples)
    else:
        final_permutations = possible_permutations

    # Optional: verbose output
    if verbose:
        # Handle Cases Where Only One Permutation is Possible
        if final_permutations == 1:
            print("Warning: Only using the unpermuted data.")

        print(f"Total possible permutations: {formatted_possible}")
        print(f"Running number of permutations: {int(final_permutations) if final_permutations < float('inf') else formatted_possible}")

    return

def test_statistics_calculations(Din, Rin, perm, test_statistics, reg_pinv, method, category_columns=[], combine_tests=False, idx_data=None, permute_beta=False, beta = None, test_indices=None, Nnull_samples=None, n_cca_components = 1):
    """
    Computes test statistics and p-values based on permutation testing.

    Parameters:
    --------------
    Din (numpy.ndarray): 
        Independent variable matrix (predictors).
    Rin (numpy.ndarray): 
        Dependent variable matrix (responses).
    perm (int): 
        Current permutation index.
    test_statistics (numpy.ndarray): 
        Array storing computed test statistics for each permutation.
    reg_pinv (numpy.ndarray or None):  
        Regularized pseudo-inverse of `Din`, if applicable.
    method (str): 
        The permutation testing method used.
    category_columns (dict), optional:
        A dictionary marking the columns where t-test ("t_stat_independent_cols") and F-test ("f_anova_cols") have been applied.
    combine_tests (str), default=False:       
        Specifies the method for combining test results.
        Valid options: "True", "across_columns", "across_rows".
    idx_data (numpy.ndarray), default=None: 
        Indexing array for session-based data.
    permute_beta (bool), default=False: 
        If `True`, permutes beta coefficients.
    beta (numpy.ndarray), default=None:
        Beta coefficient matrix with shape `(num_sessions, p, q)`, where:
        - `num_sessions`: Number of sessions.
        - `p`: Number of predictor features.
        - `q`: Number of dependent variables.
    test_indices (numpy.ndarray), default=None:
        Indices marking test-set data points for each session.
    Nnull_samples (int), default=None:
        Total number of permutations (if needed for normalization).
    n_cca_components (int), default=1:
        Number of Canonical Correlation Analysis (CCA) components to compute. 
        Increasing this allows for capturing additional modes of covariance between `Din` and `Rin`.

    Returns:
    ----------  
    test_statistics (numpy.ndarray): 
        Updated array storing computed test statistics.
    base_statistics (numpy.ndarray): 
        Raw statistics before permutation-based p-value computation.
    pval_matrix (numpy.ndarray): 
        Parametric P-values derived from the different tests
    """
    # Account for univariate tests
    t_stats, F_stats = None, None       
    if permute_beta:
        test_indices_con = np.concatenate(test_indices[0], axis=0) if len(test_indices) == 1 else None
        idx_data =get_indices_from_list(test_indices[0]) if len(test_indices) == 1 else idx_data
        if len(test_indices) > 1:
            n, p = Din.shape
            q = Rin.shape[-1]
            R2_stats, F_stats = np.zeros(q), np.zeros(q)
            pval_matrix = np.zeros(q) if method=="multivariate" else None
            t_stats = np.zeros((p, q))  
    else:
        test_indices_con = None
    # Look if combined test is performed
    combine_tests_flag = combine_tests in [True, "across_columns", "across_rows"]

    if method == 'multivariate':
        nan_values = np.isnan(Rin).any() 
        if category_columns["t_stat_independent_cols"]==[] and category_columns["f_manova_cols"]==[] and category_columns["f_anova_cols"]==[] and category_columns["f_reg_cols"]==[] or category_columns["f_reg_cols"]=="all_columns":
            if nan_values:
                # NaN values are detected
                if combine_tests_flag:
                    if permute_beta:
                        if len(test_indices) ==1:
                            R2_stats, F_stats, t_stats, pval_matrix =ols_regression_sessions(Rin[test_indices_con,:], Din[test_indices_con,:], idx_data, beta[0], perm, permute_beta, regression_statistics=True, Nnull_samples=Nnull_samples)
                        else:
                            for col in range(len(test_indices)):
                                idx_data =get_indices_from_list(test_indices[col])
                                test_indices_con = np.concatenate(test_indices[col], axis=0)
                                # calculate beta coefficients
                                _, _, _, pval_matrix[col] =ols_regression_sessions(Rin[test_indices_con,col], Din[test_indices_con,:], idx_data, beta[col], perm, permute_beta, regression_statistics=True, Nnull_samples=Nnull_samples, nan_values=nan_values,no_t_stats=True)
                        base_statistics, pval_matrix = calculate_combined_z_scores(pval_matrix, combine_tests)
                        test_statistics[perm] =abs(base_statistics) 
                    else: 
                        # Calculate F-statitics with no NaN values.
                        _, _, _, p_value =calculate_regression_statistics(Din, Rin, reg_pinv, nan_values, no_t_stats=True)
                        # Get the base statistics and store p-values as z-scores to the test statistic
                        base_statistics, pval_matrix = calculate_combined_z_scores(p_value, combine_tests)
                        test_statistics[perm] =abs(base_statistics) 
                elif permute_beta:
                    # Nan values
                    base_statistics = np.zeros((Rin.shape[-1]))
                    for col, test_index in enumerate(test_indices):
                        idx_data =get_indices_from_list(test_index)
                        test_indices_con = np.concatenate(test_index, axis=0)
                        _, F_stats[col], t_stat, pval_matrix[col] =ols_regression_sessions(Rin[test_indices_con,col], Din[test_indices_con,:], idx_data, beta[col], perm, permute_beta, regression_statistics=True, Nnull_samples=Nnull_samples, nan_values=nan_values)
                        t_stats[:,col] = t_stat.flatten() if t_stat.ndim==2 else t_stat 
                        base_statistics[col] = F_stats[col].copy()
                    test_statistics[perm] =(base_statistics)       
                else:
                    # Calculate the explained variance if R got NaN values.
                    R2_stats, F_stats, t_stats, pval_matrix=calculate_regression_statistics(Din, Rin, reg_pinv, nan_values)
                    base_statistics = F_stats #r_squared
                    test_statistics[perm,:] =F_stats           
            else:
                # No- NaN values are detected
                if idx_data is not None and permute_beta==True:
                    # Calculate predicted values with no NaN values
                    if permute_beta:
                        if len(test_indices) ==1:
                            # calculate beta coefficients
                            R2_stats, F_stats, t_stats, pval_matrix =ols_regression_sessions(Rin[test_indices_con,:], Din[test_indices_con,:], idx_data, beta[0], perm, permute_beta, regression_statistics=True, Nnull_samples=Nnull_samples)
                        else:
                            for col in range(len(test_indices)):
                                idx_data =get_indices_from_list(test_indices[col])
                                test_indices_con = np.concatenate(test_indices[col], axis=0)
                                # calculate beta coefficients
                                R2_stats[col], F_stats[col], t_stat, pval_matrix[col] =ols_regression_sessions(Rin[test_indices_con,col], Din[test_indices_con,:], idx_data, beta[col], perm, permute_beta, regression_statistics=True, Nnull_samples=Nnull_samples)
                                t_stats[:,col] = t_stat.flatten() if t_stat.ndim==2 else t_stat
                    base_statistics = F_stats #r_squared
                else:
                    # Calculate statistics with zero Nan values
                    R2_stats, F_stats, t_stats, pval_matrix =calculate_regression_statistics(Din, Rin, reg_pinv)
                if combine_tests in [True, "across_columns"]:
                    # Calculate the degrees of freedom for the model and residuals
                    df1 = Din.shape[1]  # Number of predictors 
                    df2 = Din.shape[0] - df1

                    pval = 1 - f.cdf(F_stats, df1, df2)
                    # Get the base statistics and store p-values as z-scores to the test statistic
                    base_statistics, pval_matrix = calculate_combined_z_scores(pval, combine_tests)
                    test_statistics[perm] =(base_statistics) 
                else:
                    # Store the R^2 values in the test_statistics array
                    base_statistics = F_stats
                    test_statistics[perm] = (base_statistics)
        elif category_columns["f_manova_cols"]=="all_columns" and nan_values == False:
            # Then we need to calculate beta
            F_stats, t_stats, pval_matrix=calculate_manova_f_test(Din, Rin, nan_values)
            # Store statistics
            #t_stats =t_stat.flatten() if t_stat.ndim==2 and t_stat.shape[1]!=1 else t_stat
            base_statistics = F_stats.copy()
            test_statistics[perm] =(base_statistics) 

        elif category_columns["f_anova_cols"]=="all_columns" and nan_values == False:
            if permute_beta: 
                if len(test_indices) ==1: 
                    # calculate beta coefficients
                    F_stats, t_stats, pval_matrix =calculate_anova_f_test(Din[test_indices_con,:], Rin[test_indices_con,:], idx_data, permute_beta, perm, nan_values, beta=beta[0], Nnull_samples=Nnull_samples)  
                    base_statistics = F_stats
                    test_statistics[perm] =(base_statistics) 
                else:
                    for col in range(len(test_indices)):
                        idx_data =get_indices_from_list(test_indices[col])
                        test_indices_con = np.concatenate(test_indices[col], axis=0)
                        F_stats[col], t_stat, pval_matrix[col] =calculate_anova_f_test(Din[test_indices_con,:], Rin[test_indices_con,col], idx_data, permute_beta, perm, nan_values, beta=beta[col], Nnull_samples=Nnull_samples)  
                        t_stats[:,col] = t_stat.flatten() if t_stat.ndim==2 else t_stat
                    base_statistics = F_stats
                    test_statistics[perm] =(base_statistics) 
            else:
                F_stats, t_stats, pval_matrix =calculate_anova_f_test(Din, Rin, idx_data, permute_beta, perm, nan_values)   
                base_statistics = F_stats
                test_statistics[perm] =(base_statistics)    
        else:
            # Now we have to do t- or f-statistics
            # If we are doing combine_testss, we need to calculate f-statistics on every column
            if combine_tests_flag and category_columns["t_stat_independent_cols"]==[] and category_columns["f_anova_cols"]==[] and category_columns["f_reg_cols"]==[]:
                nan_values = np.isnan(Rin).any()
                # Calculate the explained variance if R got NaN values.
                _, _, _, p_value =calculate_regression_statistics(Din, Rin, reg_pinv, nan_values, no_t_stats=True)
                # Get the base statistics and store p-values as z-scores to the test statistic
                base_statistics, pval_matrix = calculate_combined_z_scores(p_values, combine_tests)
                test_statistics[perm] =(base_statistics) 
            else:
                # If we are not perfomring combine_tests, we need to perform a columnwise operation.  
                # We perform f-test if category_columns has flagged categorical columns otherwise it will be R^2 
                # Initialize variables 
                n, p = Din.shape
                q = Rin.shape[-1]
                R2_stats = np.zeros(q) 
                F_stats = np.zeros(q) 
                t_stats = np.zeros((p, q))  # Initialize t-statistics matrix 
                base_statistics =np.zeros_like(F_stats) if combine_tests_flag!=False else np.zeros_like(test_statistics[0,:])
                pval_matrix =np.zeros_like(base_statistics) 
                
                for col in range(q):
                    # Get the R_column
                    R_column = Rin[:, col]
                    # Calculate f-statistics of columns of interest 
                    nan_values = np.sum(np.isnan(Rin[:,col]))>0 
                    if col in category_columns["f_manova_cols"]:
                        # Then we need to calculate beta
                        F_stats[col], t_stat, pval_matrix[col]=calculate_manova_f_test(Din, R_column, True)
                        # Store statistics
                        t_stats[:,col] =t_stat.flatten() if t_stat.ndim==2 else t_stat
                        base_statistics[col] = F_stats[col].copy()
                        if not combine_tests_flag: # Only assign if combine_tests_flag is False
                            test_statistics[perm,col] = base_statistics[col]    

                    if col in category_columns["f_anova_cols"]:
                        # Nan values
                        if permute_beta:
                            # Calculate base statistics per column
                            idx_data =get_indices_from_list(test_indices[col])
                            test_indices_con = np.concatenate(test_indices[col], axis=0)
                            F_stats[col], t_stat, pval_matrix[col] =calculate_anova_f_test(Din[test_indices_con,:], Rin[test_indices_con,col], idx_data, permute_beta, perm, nan_values, beta=beta[col], Nnull_samples=Nnull_samples)  
                        else:
                            # Then we need to calculate beta
                            F_stats[col], t_stat, pval_matrix[col]=calculate_anova_f_test(Din, R_column, idx_data, permute_beta, perm, nan_values)
                        # Store statistics
                        t_stats[:,col] =t_stat.flatten() if t_stat.ndim==2 else t_stat
                        base_statistics[col] = F_stats[col].copy()
                        if not combine_tests_flag: # Only assign if combine_tests_flag is False
                            test_statistics[perm,col] = base_statistics[col]          
                    elif col in category_columns["f_reg_cols"]:
                        # Nan values
                        if permute_beta:
                            # Calculate base statistics per column
                            idx_data =get_indices_from_list(test_indices[col])
                            test_indices_con = np.concatenate(test_indices[col], axis=0)
                            _, F_stats[col], t_stat, pval_matrix[col] =ols_regression_sessions(Rin[test_indices_con,col], Din[test_indices_con,:], idx_data, beta[col], perm, permute_beta, regression_statistics=True, Nnull_samples=Nnull_samples, nan_values=nan_values)
                            # Calculate f statistics
                        else:
                            R2_stats[col], F_stats[col], t_stat, pval_matrix[col] = calculate_regression_statistics(Din, R_column, reg_pinv, nan_values=nan_values)
                        # Store statistics
                        t_stats[:,col] = t_stat.flatten() if t_stat.ndim==2 else t_stat 
                        base_statistics[col] = F_stats[col].copy()    
                        if not combine_tests_flag: # Only assign if combine_tests_flag is False
                            test_statistics[perm,col] = base_statistics[col]         
                if combine_tests_flag:                
                    base_statistics, pval_matrix = calculate_combined_z_scores(pval_matrix, combine_tests)
                    test_statistics[perm] =(base_statistics) 
    # Calculate for univariate tests              
    elif method == "univariate":
        nan_values = np.isnan(Din).any() or np.isnan(Rin).any()
        if category_columns["t_stat_independent_cols"]==[] and category_columns["f_anova_cols"]==[]and category_columns["f_reg_cols"]==[]:

            # Only calcuating the correlation matrix, since there is no need for t- or f-test
            if combine_tests_flag: 
                if permute_beta:
                    # Columnwise
                    if len(test_indices) ==1:
                        # calculate beta coefficients
                        _, p_value =calculate_regression_f_stat_univariate(Din, Rin, idx_data, beta[0], perm, reg_pinv, permute_beta, test_indices=test_indices_con)
                        base_statistics, pval_matrix =calculate_combined_z_scores(p_value, combine_tests)
                        test_statistics[perm] =abs(base_statistics) 
                    else:
                        base_statistics = np.zeros((Din.shape[-1],Rin.shape[-1]))
                        p_values = np.zeros((Din.shape[-1],Rin.shape[-1]))
                        for col, test_index in enumerate(test_indices):
                            idx_data =get_indices_from_list(test_index)
                            test_indices_con = np.concatenate(test_index, axis=0)
                            # Calculate geometric mean of p-values
                            _, p_value =calculate_regression_f_stat_univariate(Din, Rin[:,col], idx_data, beta[col], perm, reg_pinv, permute_beta, test_indices=test_indices_con)
                            p_values[:, col] = p_value.squeeze()
                        base_statistics, pval_matrix =calculate_combined_z_scores(p_values, combine_tests)
                        test_statistics[perm,:] =abs(base_statistics) 
                else: 
                    # Return parametric p-values   
                    corr_matrix, base_statistics, p_values  = compute_correlation_tstats(Din, Rin, pval_parametric=True)
                    # base statistics => Z_score
                    base_statistics, pval_matrix =calculate_combined_z_scores(p_values, combine_tests)
                    test_statistics[perm] =abs(base_statistics) 
            elif permute_beta:
                if len(test_indices) ==1:
                    # calculate beta coefficients
                    base_statistics, pval_matrix =calculate_regression_f_stat_univariate(Din, Rin, idx_data, beta[0], perm, reg_pinv, permute_beta, test_indices=test_indices_con)
                    # Ensure F_stats is at least 2D
                    base_statistics =np.expand_dims(base_statistics,axis=1) if base_statistics.ndim==1 else base_statistics
                    pval_matrix =np.expand_dims(pval_matrix,axis=1) if pval_matrix.ndim==1 else pval_matrix
                    test_statistics[perm, :, :] = np.abs(base_statistics)
                else:
                    base_statistics = np.zeros((Din.shape[-1],Rin.shape[-1]))
                    pval_matrix = np.zeros((Din.shape[-1],Rin.shape[-1]))
                    for col, test_index in enumerate(test_indices):
                        idx_data =get_indices_from_list(test_index)
                        test_indices_con = np.concatenate(test_index, axis=0)
                        # calculate beta coefficients
                        base_stats, p_values =calculate_regression_f_stat_univariate(Din, Rin[:,col], idx_data, beta[col], perm, reg_pinv, permute_beta, test_indices=test_indices_con)
                        base_statistics[:, col] = base_stats.squeeze()
                        pval_matrix[:, col] = p_values.squeeze()
                        test_statistics[perm, :, col] = np.abs(base_statistics[:,col] )
            elif perm==0:
                # Compute correlation, p-values, and t-statistics
                corr_matrix, base_statistics, pval_matrix  = compute_correlation_tstats(Din, Rin, pval_parametric=True)
                # base statistics => t-statistics
                test_statistics[perm, :, :] = np.abs(base_statistics)

            else:
                # Calculate correlation coeffcients without NaN values
                corr_matrix, base_statistics, pval_matrix  = compute_correlation_tstats(Din, Rin)
                test_statistics[perm, :, :] = np.abs(base_statistics)
        elif "all_columns" in category_columns.values() and nan_values == False:
            # No need for a columnwise operation
            if category_columns["t_stat_independent_cols"]=="all_columns":
                base_statistics, pval_matrix = calculate_nan_t_test(Din, Rin, nan_values=nan_values)
                test_statistics[perm,:] =abs(base_statistics)
            elif category_columns["f_anova_cols"]=="all_columns":
                base_statistics, t_stats, pval_matrix = calculate_anova_f_test(Din, Rin, method=method)
                test_statistics[perm] =base_statistics
        else: 
            pval_matrix = np.zeros((Din.shape[-1],Rin.shape[-1]))
            base_statistics = np.zeros((Din.shape[-1],Rin.shape[-1]))
            for col in range(Rin.shape[1]):
                # Column-wise NaN check
                nan_values = True if np.sum(np.isnan(Din))>0 or np.sum(np.isnan(Rin))>0 else False
                if col in category_columns["t_stat_independent_cols"]:
                    # Perform  t-statistics per column if nan_values=True 
                    base_stat, pval = calculate_nan_t_test(Din, Rin[:, col], nan_values=nan_values) 
                    base_stat = abs(base_stat)   
                elif col in category_columns["f_anova_cols"]:
                    if permute_beta==False:
                        # Perform f-statistics
                        base_stat, _, pval =calculate_anova_f_test(Din, Rin[:, col], method=method) 
                    else:
                        idx_data =get_indices_from_list(test_indices[col])
                        test_indices_con = np.concatenate(test_indices[col], axis=0)
                        base_stat, _, pval =calculate_anova_f_test(Din[test_indices_con,:], Rin[test_indices_con,col], idx_data, permute_beta, perm, nan_values, beta=beta[col], method=method)  
                elif col in category_columns["f_reg_cols"]:
                    if permute_beta == False:
                        # Perform f-statistics
                        base_stat, pval =calculate_regression_f_stat_univariate(Din, Rin[:, col], idx_data, beta, perm, reg_pinv, permute_beta, test_indices=test_indices_con)  
                    else:
                        idx_data =get_indices_from_list(test_indices[col])
                        test_indices_con = np.concatenate(test_indices[col], axis=0)
                        # calculate beta coefficients
                        base_stat, pval =calculate_regression_f_stat_univariate(Din, Rin[:,col], idx_data, beta[col], perm, reg_pinv, permute_beta, test_indices=test_indices_con)
                else:
                    # Perform correlation analysis and handle NaN values
                    corr_array, t_stat_corr, pval =compute_correlation_tstats(Din, Rin[:, col], True)
                    base_stat = np.squeeze(t_stat_corr)
                # Store p-values
                pval_matrix[:, col] = np.squeeze(pval)
                base_statistics[:, col] = np.squeeze(base_stat)

                # Store test statistics if combine_tests is False
                if not combine_tests:
                    test_statistics[perm, :, col] = np.abs(base_statistics[:, col])
                          
            if combine_tests_flag:
                #base_statistics = calculate_combined_z_scores(base_statistics_com, combine_tests) if perm==0 else base_statistics_com
                base_statistics, pval_matrix = calculate_combined_z_scores(pval_matrix, combine_tests) 
                test_statistics[perm,:] =abs(base_statistics) 
                #pval_matrix =geometric_pvalue(pval_matrix, combine_tests)


    elif method =="cca":
        pval_matrix = None
        # Get the base statistics and test statistics 
        base_statistics, test_statistics =  compute_cca_statistics(Din, Rin, test_statistics, perm, permute_beta,test_indices_con, idx_data, beta, n_cca_components)


    # Check if perm is 0 before returning the result
    stats_results = {"null_stat_distribution":test_statistics,
                     "base_statistics":base_statistics,
                     "pval_matrix": pval_matrix,
                     "F_stats": F_stats,
                     "t_stats": t_stats}

    return stats_results

def define_predictor_outcome_names(method, combine_tests, predictor_names, outcome_names, n_p, n_q):
    """
    Define predictor and outcome names based on the selected statistical method and test combination settings.

    Parameters:
    --------------
    method (str):
        The statistical method used for analysis. Valid options:
        - 'multivariate': Uses multiple predictors together to find patterns.
        - 'univariate': Tests each predictor separately.
        - 'cca': Canonical Correlation Analysis, finding a single pattern across two data sets.
    combine_test (bool or str):
        Determines how test results are combined:
        - `True`: Produces a single p-value for the entire test (`1 × 1`).
        - "across_rows": Produces one p-value per outcome (`1 × q`).
        - "across_columns": Produces one p-value per predictor (`1 × p`).
        - `False`: No test combination, default.
    predictor_names (list of str):
        Names of predictor variables. If empty or mismatched with `n_p`, default names are assigned.
    outcome_names (list of str):
        Names of outcome variables. If empty or mismatched with `n_q`, default names are assigned.
    n_p (int):
        Number of predictor variables.
    n_q (int):
        Number of outcome variables.

    Returns:
    ----------
    predictor_name (str or list):
        Assigned name(s) for the predictor variable(s).
    outcome_name (str or list):
        Assigned name(s) for the outcome variable(s).
    """
    # Default predictor and outcome names if not provided
    predictor_names = predictor_names if predictor_names and len(predictor_names) == n_p else [f"State {i+1}" for i in range(n_p)]
    outcome_names = outcome_names if outcome_names and len(outcome_names) == n_q else [f"Regressor {i+1}" for i in range(n_q)]
    
    # Assign names based on method and test combination
    if method == "cca" or combine_tests is True:
        predictor_name = "States"
        outcome_name = "Regressors"
    elif combine_tests == "across_columns":
        predictor_name = predictor_names  # Keep predictor names as they are
        outcome_name = "Regressors"
    elif combine_tests == "across_rows":
        predictor_name = "States"
        outcome_name = outcome_names  # Keep outcome names as they are
    else:
        predictor_name = predictor_names
        outcome_name = outcome_names
    
    return predictor_name, outcome_name


def calculate_combined_z_scores(p_values, combine_tests=None):
    """
    Calculate test statistics of z-scores converted from p-values based on the specified combination.

    Parameters:
    --------------
    p_values (numpy.ndarray):  
        Matrix of p-values.
    combine_tests (str):       
        Specifies the combination method.
        Valid options: "True", "across_columns", "across_rows".
        Default is "True".

    Returns:
    ----------  
    result (numpy.ndarray):       
        Test statistics of z-scores converted from p-values.
    """
    # Cap p-values slightly below 1 to avoid infinite z-scores
    epsilon = 1e-15
    adjusted_pval = np.clip(p_values, epsilon, 1 - epsilon) #  restricts the values in pval_matrix to lie within the range [epsilon, 1 - epsilon]
    if combine_tests == True:        
        pval = np.squeeze(np.exp(np.mean(np.log(adjusted_pval))))                   
        z_scores = norm.ppf(1 - np.array(pval))
        test_statistics = z_scores
    elif combine_tests == "across_columns" or combine_tests == "across_rows":
        axis = 1 if combine_tests == "across_columns" else 0
        # Apply the geoemtric mean based on it is a 2D or 1D array
        pval = np.squeeze(np.exp(np.mean(np.log(adjusted_pval), axis=axis))) if p_values.ndim==2 else np.squeeze(np.exp(np.mean(np.log(adjusted_pval))))
        z_scores = norm.ppf(1 - np.array(pval))
        test_statistics = z_scores
    else:
        pval = adjusted_pval.copy()
        z_scores = norm.ppf(1 - np.array(adjusted_pval))
        test_statistics = np.squeeze(z_scores)

    return test_statistics, pval

# Define the inverse Fisher z-transformation function
def inverse_fisher_z(z_matrix):
    """
    Convert z-scores back to correlation coefficients using the inverse Fisher z-transformation.
    
    Parameters:
    z_matrix (ndarray): 
        A matrix of z-scores.
    
    Returns:
    ndarray: 
        A matrix of correlation coefficients.
    """
    return (np.exp(2 * z_matrix) - 1) / (np.exp(2 * z_matrix) + 1)

def compute_cca_statistics(Din, Rin, test_statistics, perm, permute_beta=False, 
                           test_indices_con=None, idx_data=None, beta=None, n_cca_components=1):
    """
    Compute test statistics using Canonical Correlation Analysis (CCA).

    Parameters:
    --------------
    Din (numpy.ndarray):  
        Input data matrix (predictors).
    Rin (numpy.ndarray):  
        Target data matrix (responses).
    test_statistics (numpy.ndarray):  
        Array storing computed statistics for permutations.
    perm (int):  
        Permutation index.
    permute_beta (bool, optional):  
        Whether to permute beta coefficients. Default is False.
    test_indices_con (numpy.ndarray or None, optional):  
        Concatenated test indices for permutation. Default is None.
    idx_data (numpy.ndarray or None, optional):  
        Indices for data selection. Default is None.
    beta (numpy.ndarray or None, optional):  
        Beta coefficients for regression if permuting. Default is None.
    n_cca_components (int, optional):  
        Number of CCA components to compute. Default is 1.

    Returns:
    ----------  
    base_statistics (numpy.ndarray):       
        Computed CCA correlation values for each component.
    test_statistics (numpy.ndarray):       
        Updated test statistics   
    """
    if permute_beta:
        # Predict R using the beta coefficients
        R_pred = ols_regression_sessions(Rin[test_indices_con], Din[test_indices_con], idx_data, beta[0], perm, permute_beta)
        D_residual, R_residual = Din[test_indices_con].copy(), R_pred.copy()
    else:
        D_residual, R_residual = Din.copy(), Rin.copy()

    base_statistics = np.zeros(n_cca_components)

    for cca_component in range(n_cca_components):
        cca = CCA(n_components=1)  # Compute one component at a time
        cca.fit(D_residual, R_residual)
        D_c, R_c = cca.transform(D_residual, R_residual)

        # Compute the correlation for this component
        base_statistics[cca_component] = np.corrcoef(D_c[:, 0], R_c[:, 0])[0, 1]
        test_statistics[perm, cca_component] = np.abs(base_statistics[cca_component])

        # Remove explained variance from D and R
        D_residual -= D_c @ cca.x_weights_.T
        R_residual -= R_c @ cca.y_weights_.T  

    return base_statistics, test_statistics

def compute_correlation_tstats(Din, Rin, pval_parametric=False):
    """
    Computes Pearson correlation coefficients and t-statistics between independent variables (Din) 
    and dependent variables (Rin), with automatic NaN handling.
    
    If no NaN values exist, it calls `compute_correlation_and_tstats` directly for fast vectorized computation.
    Otherwise, it processes each column pair individually while ignoring NaNs.

    Parameters:
    --------------
    Din (numpy.ndarray): 
        Input D-matrix for the independent variables.
    Rin (numpy.ndarray): 
        Input R-matrix for the dependent variables.
    pval_parametric (bool, optional): 
        If True, calculates parametric p-values alongside the t-statistics. Default is False.

    Returns:
    ----------  
    correlation_matrix (numpy.ndarray): 
        Pearson correlation coefficients for each predictor-response pair.
    t_statistics (numpy.ndarray): 
        t-statistics for each predictor-response pair.
    pval_matrix (numpy.ndarray): 
        p-values for each predictor-response pair. If `pval_parametric=False`, this will be NaN.
    """
    # Convert 1D arrays to 2D (if needed)
    if Din.ndim == 1:
        Din = Din[:, np.newaxis]
    if Rin.ndim == 1:
        Rin = Rin[:, np.newaxis]

    # Ensure both arrays have the same number of samples
    if Din.shape[0] != Rin.shape[0]:
        raise ValueError("Din and Rin must have the same number of samples (rows).")

    # If no NaNs, use the optimized function
    if not np.isnan(Din).any() and not np.isnan(Rin).any():
        return calculate_correlation_and_tstats(Din, Rin, pval_parametric)

    # Otherwise, handle NaNs column-by-column
    p, q = Din.shape[1], Rin.shape[1]
    correlation_matrix = np.full((p, q), np.nan)
    t_statistics = np.full((p, q), np.nan)
    pval_matrix = np.full((p, q), np.nan)

    for p_i in range(p):
        D_column = Din[:, p_i]
        for q_j in range(q):
            R_column = Rin[:, q_j]
            # Find non-NaN indices
            valid_indices = ~np.isnan(D_column) & ~np.isnan(R_column)
            if np.any(valid_indices):  # Ensure at least one valid pair
                correlation_matrix[p_i, q_j], t_statistics[p_i, q_j], pval_matrix[p_i, q_j] = \
                    calculate_correlation_and_tstats(D_column[valid_indices], R_column[valid_indices], pval_parametric)

    return correlation_matrix, t_statistics, pval_matrix

def calculate_correlation_and_tstats(Din, Rin, pval_parametric=False):
    """
    Calculates Pearson correlation coefficients and t-statistics 
    for each pair of columns between Din and Rin. Optionally calculates p-values.

    Parameters:
    --------------
    Din (numpy.ndarray): 
        Input data vector or matrix with shape (n_samples, n_predictors).
    Rin (numpy.ndarray): 
        Input data vector or matrix with shape (n_samples, n_targets).
    pval_parametric (bool, optional): 
        If True, calculates parametric p-values. Default is False.

    Returns:
    --------------
    correlation_matrix (numpy.ndarray or float): 
        Pearson correlation coefficient(s) for each predictor-target pair.
    t_statistics (numpy.ndarray or float): 
        t-statistic(s) for each predictor-target pair, computed from the correlation coefficients.
    pval_matrix (numpy.ndarray or float, optional): 
        Only returned if `pval_parametric=True`. Permutation-based p-values for each predictor-target pair.
    """
    # Convert 1D arrays to 2D (if needed)
    if Din.ndim == 1:
        Din = Din[:, np.newaxis]  # Convert to shape (n, 1)
    if Rin.ndim == 1:
        Rin = Rin[:, np.newaxis]  # Convert to shape (n, 1)

    # Ensure both arrays have the same number of samples (n)
    if Din.shape[0] != Rin.shape[0]:
        raise ValueError("Din and Rin must have the same number of samples (rows).")

    # Number of samples
    n = Din.shape[0]

    # Center the data
    Din_centered = Din - Din.mean(axis=0) 
    Rin_centered = Rin - Rin.mean(axis=0)

    # Compute correlation coefficients via dot product
    numerator = Din_centered.T @ Rin_centered
    denom_x = np.sqrt(np.sum(Din_centered**2, axis=0))[:, None]
    denom_y = np.sqrt(np.sum(Rin_centered**2, axis=0))
    correlation_matrix = numerator / (denom_x @ denom_y[None, :])

    # Calculate t-statistics using correlation coefficients
    t_statistics = correlation_matrix * np.sqrt(n - 2) / np.sqrt(1 - correlation_matrix**2)

    # Compute p-values if pval_parametric is True, else set to NaN
    if pval_parametric:
        df = n - 2  # Degrees of freedom
        pval_matrix = 2 * (1 - t.cdf(np.abs(t_statistics), df))
    else:
        pval_matrix = np.full_like(correlation_matrix, np.nan)

    return correlation_matrix, t_statistics, pval_matrix

def pval_correction(result_dic=None, pval=None, method='fdr_bh', alpha=0.05, include_nan=True, nan_diagonal=False):
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
        Significance level, default= 0.05.
    include_nan, default=True: 
        Include NaN values during the correction of p-values if True. Exclude NaN values if False.
    nan_diagonal, default=False: 
        Add NaN values to the diagonal if True.

    Returns:
    ---------- 
    pval_corrected (numpy.ndarray): 
        numpy array of corrected p-values.
        - 2D: `(p, q)`, where `p` is the number of predictors and `q` is the number of outcome variables.
        - 3D: `(T, p, q)`, where `p` is te number of timepoints, `p` is the number of predictors and `q` is the number of outcome variables.

    significant (numpy.ndarray): 
        numpy array of boolean values indicating significant p-values.
    """
    # Use the dictionary values if provided
    if result_dic is not None:
        pval = result_dic["pval"]
        #pval = np.squeeze(result_dic["pval"], axis=0) if result_dic["test_summary"]["Timepoints"] ==1 else result_dic["pval"]
        

    if pval is None:
        raise ValueError("Missing required parameters: pval")
    
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

def pval_FWER_correction(result_dic=None, test_statistics=None, Nnull_samples=None, method=None):
    """
    Compute Family-Wise Error Rate (FWER) corrected p-values for multivariate or univariate methods.

    Parameters:
    --------------
    result_dic (dict, default None:
        A dictionary containing "null_stat_distribution", "Nnull_samples", and "method".
    test_statistics (numpy.ndarray), default None:
        The permutation array, where the first row/element contains observed statistics.
    Nnull_samples (int), default None:
        The number of permutations.
    method (str), default None:
        The method used for permutation testing. Can be "multivariate" or "univariate".

    Returns:
    ----------
    pval_FWER (numpy.ndarray):
        FWER-corrected p-values.
    """
    # Use the dictionary values if provided
    if result_dic is not None:
        test_statistics = result_dic["null_stat_distribution"]
        Nnull_samples = result_dic['Nnull_samples']
        method = result_dic['method']

    if test_statistics is None or Nnull_samples is None or method is None:
        raise ValueError("Missing required parameters: test_statistics, Nnull_samples, or method.")

    if method == "multivariate":
        if test_statistics.shape[0] == Nnull_samples:  # Case 1: Without timepoints
            test_statistics = np.expand_dims(test_statistics, axis=1) if test_statistics.ndim == 1 else test_statistics
            # Compute the maximum statistic for each permutation
            max_stats = np.max(test_statistics[1:], axis=1)  # Shape: (Nnull_samples - 1,)

            # Get the observed (unpermuted) statistics (first permutation)
            observed_stats = test_statistics[0, :]  # Shape: (F,)

            # Compute FWER-corrected p-values for each feature
            pval_FWER = (np.sum(max_stats[:, np.newaxis] >= observed_stats, axis=0) + 1) / (Nnull_samples + 1)  # Shape: (F,)
        else:  # Case 2: With timepoints
            n_T = test_statistics.shape[0]
            test_statistics = np.expand_dims(test_statistics, axis=2) if test_statistics[0, :].ndim == 1 else test_statistics
            # Compute the maximum statistic for each timepoint and permutation
            max_stats = np.max(test_statistics[:, 1:, :], axis=2)  # Shape: (T, Nnull_samples - 1)

            # Get the observed (unpermuted) statistics (first permutation for each timepoint)
            observed_stats = test_statistics[:, 0, :]  # Shape: (T, F)

            # Compute FWER-corrected p-values for each timepoint and feature
            pval_FWER = (np.sum(max_stats[:, :, np.newaxis] >= observed_stats[:, np.newaxis, :], axis=1) + 1) / (Nnull_samples + 1)  # Shape: (T, F)

    elif method == "univariate":
        if test_statistics.shape[0] == Nnull_samples:  # Case 1: Without timepoints
            test_statistics = np.expand_dims(test_statistics, axis=2) if test_statistics[0, :].ndim == 1 else test_statistics
            maxT_statistics = np.max(np.abs(test_statistics[1:, :, :]), axis=(1, 2))  # Shape: (Nnull_samples - 1,)
            observed_test_stats = np.abs(test_statistics[0])  # Shape: (p, q)

            # Use broadcasting to compare observed statistics against MaxT statistics - Get a comparison for every combination of permutation, feature, and outcome.
            pval_FWER = (np.sum(maxT_statistics[:, np.newaxis, np.newaxis] >= observed_test_stats, axis=0) + 1) / (Nnull_samples + 1)  # Shape: (p, q)
        else:  # Case 2: With timepoints
            n_T = test_statistics.shape[0]
            test_statistics = np.expand_dims(test_statistics, axis=3) if test_statistics[0, :].ndim == 2 else test_statistics
            pval_FWER = np.zeros((n_T, test_statistics.shape[-2], test_statistics.shape[-1]))  # Shape: (T, p, q)

            for t in range(n_T):
                maxT_statistics = np.max(np.abs(test_statistics[t, 1:, :, :]), axis=(1, 2))  # Shape: (Nnull_samples - 1,)
                observed_test_stats = np.abs(test_statistics[t, 0])  # Shape: (p, q)

                # Use broadcasting to compare observed statistics against MaxT statistics - Get a comparison for every combination of permutation, feature, and outcome.
                pval_FWER[t, :, :] = (np.sum(maxT_statistics[:, np.newaxis, np.newaxis] >= observed_test_stats, axis=0) + 1) / (Nnull_samples + 1)

    else:
        raise ValueError("Invalid method. Must be 'multivariate' or 'univariate'.")

    return np.squeeze(pval_FWER) if pval_FWER.ndim > 2 else pval_FWER

def pval_cluster_based_correction(result_dic = None, test_statistics=[], pval=None, alpha=0.05, individual_feature=False):
    """
    Perform cluster-based correction on test statistics using the output from permutation testing.
    The function corrects p-values by using the test statistics and p-values obtained from permutation testing.
    It converts the test statistics into z-scores, thresholds them to identify clusters, and uses the cluster sizes to adjust the p-values.
        
    Parameters:
    ------------
    test_statistics (numpy.ndarray): 
        3D or 4D array of test statistics.
        For a 3D array, it should have a shape of (timepoints, permutations, p).
        For a 4D array, it should have a shape of (timepoints, permutations, p, q), 
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
    # Use the dictionary values if provided
    if result_dic is not None:
        test_statistics = result_dic.get("null_stat_distribution", test_statistics)
        pval = result_dic.get("pval", pval)

    if test_statistics is [] or pval is None:
        raise ValueError("Missing required parameters: test_statistics or pval.\n"
                         "Remember to set 'return_base_statistics=True' to export the test_statistics when running the test")

    if individual_feature:
        p_size = test_statistics.shape[-2]
        p_values = np.zeros_like(pval)
        for p_i in range(p_size):
            # Compute mean and standard deviation under the null hypothesis
            mean_h0 = np.squeeze(np.mean(test_statistics[:,:,p_i], axis=1))
            std_h0 = np.std(test_statistics[:,:,p_i], axis=1)

            # Initialize array to store maximum cluster sums for each permutation
            Nnull_samples = test_statistics[:,:,p_i].shape[1]
            # Not including the first permuation
            max_cluster_sums = np.zeros(Nnull_samples-1)

            # Define zval_thresh threshold based on alpha
            zval_thresh = norm.ppf(1 - alpha)
            
            # Iterate over permutations to find maximum cluster sums
            for perm in range(Nnull_samples-1):
                # Take each permutation map and transform to Z
                thresh_Nnull_samples = (test_statistics[:,perm+1,p_i])
                if np.sum(thresh_Nnull_samples)!=0:
                    #thresh_Nnull_samples = permmaps[perm, :]
                    thresh_Nnull_samples = (thresh_Nnull_samples - np.mean(thresh_Nnull_samples)) / np.std(thresh_Nnull_samples)
                    # Threshold line at p-value
                    thresh_Nnull_samples[np.abs(thresh_Nnull_samples) < zval_thresh] = 0

                    # Find clusters
                    cluster_label = label(thresh_Nnull_samples > 0)
                    if len(np.unique(cluster_label)) > 1:
                        temp_cluster_sums = [np.sum(thresh_Nnull_samples[cluster_label == label]) for label in range(1, len(np.unique(cluster_label)))]
                        if temp_cluster_sums:
                            max_cluster_sums[perm] = max(temp_cluster_sums)
                        else:
                            max_cluster_sums[perm] = 0  # No clusters
            # Calculate cluster threshold
            cluster_thresh = np.percentile(max_cluster_sums, 100 - (100 * alpha))

            # Convert p-value map calculated using permutation testing into z-scores
            pval_zmap = norm.ppf(1 - pval[:,p_i])
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
            p_values[:,p_i] = 1 - norm.cdf(pval_zmap)
            mask = (p_values[:, p_i, 0] == 0.5)  # ensures we correctly extract a 1D array
            p_values[mask, p_i, 0] = 1 
    
    else:    
        # Compute mean and standard deviation under the null hypothesis
        mean_h0 = np.squeeze(np.mean(test_statistics, axis=1))
        std_h0 = np.std(test_statistics, axis=1)

        # Initialize array to store maximum cluster sums for each permutation
        Nnull_samples = test_statistics.shape[1]
        # Not including the first permuation
        max_cluster_sums = np.zeros(Nnull_samples-1)

        # Define zval_thresh threshold based on alpha
        zval_thresh = norm.ppf(1 - alpha)
        
        # Iterate over permutations to find maximum cluster sums
        for perm in range(Nnull_samples-1):
            # when test_statistics is 4D
            if test_statistics.ndim==4:
                thresh_Nnull_samples = np.squeeze(test_statistics[:, perm+1, :])
                thresh_Nnull_samples = (thresh_Nnull_samples - mean_h0) / std_h0

                # Threshold image at p-value
                thresh_Nnull_samples[np.abs(thresh_Nnull_samples) < zval_thresh] = 0

                # Find clusters using connected components labeling
                cluster_label = label(thresh_Nnull_samples > 0)
                regions = regionprops(cluster_label, intensity_image=thresh_Nnull_samples)
                if regions:
                    # Sum values inside each cluster
                    temp_cluster_sums = [np.sum(region.intensity_image) for region in regions]
                    if temp_cluster_sums:
                        # Store the sum of values for the biggest cluster
                        max_cluster_sums[perm] = max(temp_cluster_sums)
                    else:
                        max_cluster_sums[perm] = 0  # No clusters
                    
            # Otherwise it is a 3D matrix
            else: 
                # Take each permutation map and transform to Z
                thresh_Nnull_samples = (test_statistics[:,perm+1])
                if np.sum(thresh_Nnull_samples)!=0:
                    #thresh_Nnull_samples = permmaps[perm, :]
                    thresh_Nnull_samples = (thresh_Nnull_samples - np.mean(thresh_Nnull_samples)) / np.std(thresh_Nnull_samples)
                    # Threshold line at p-value
                    thresh_Nnull_samples[np.abs(thresh_Nnull_samples) < zval_thresh] = 0

                    # Find clusters
                    cluster_label = label(thresh_Nnull_samples > 0)

                    if len(np.unique(cluster_label)>0) or np.sum(cluster_label)==0:
                        # Sum values inside each cluster
                        temp_cluster_sums = [np.sum(thresh_Nnull_samples[cluster_label == label]) for label in range(1, len(np.unique(cluster_label)))]
                        if temp_cluster_sums:
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


def get_indices_range(size, step):
    """
    Create a 2D matrix of start and end indices with a fixed step size.

    Parameters:
    --------------
    size (int): 
        The total size of the data to generate indices for.
    step (int): 
        The step size for each range.

    Returns:
    ----------  
    indices (ndarray): 
        A 2D NumPy array where each row represents the start and end indices.
    
    Example:
    ----------
    >>> size = 1000
    >>> step = 250
    >>> get_indices_range(size, step)
    array([[   0,  250],
           [ 250,  500],
           [ 500,  750],
           [ 750, 1000]])
    """
    # Generate start and end indices
    start_values = np.arange(0, size, step)
    end_values = np.arange(step, size + step, step)
    end_values[-1] = size  # Ensure the last value is exactly the size

    # Combine into a 2D array
    indices = np.column_stack((start_values, end_values))

    return indices


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

def get_concatenate_data_memmap(D_raw, filename="D_con.dat", dtype=np.float32):
    """
    Saves a list of NumPy arrays (D_raw) into a memory-mapped file to optimize RAM usage.
    
    Parameters:
    -----------
    D_raw : list of np.ndarray
        List containing session-wise NumPy arrays with the same number of columns.
    filename : str, optional
        Name of the memory-mapped file to store the concatenated dataset (default is "D_con.dat").
    
    Returns:
    --------
    np.memmap
        Memory-mapped NumPy array containing the concatenated data.
    """
    if not D_raw:
        raise ValueError("D_raw cannot be empty.")
    
    # Define the shape dynamically based on D_raw
    total_samples = sum(d.shape[0] for d in D_raw)
    num_features = D_raw[0].shape[1]
    
    # Create a memory-mapped file
    D_con = np.memmap(filename, dtype=dtype, mode="w+", shape=(total_samples, num_features))
    
    # Write data incrementally
    start = 0
    for d in D_raw:
        d = d.astype(dtype, copy=False)  # Safely cast to float32 without extra copy
        end = start + d.shape[0]
        D_con[start:end] = d
        start = end

    D_con.flush()

    return D_con 

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


def reconstruct_concatenated_to_3D(D_con, D_original=None, n_timepoints=None, n_entities=None, n_features = None):
    """
    Reshape a concatenated 2D matrix back into its original 3D format (timepoints, trials, channels).
    
    This function converts a concatenated 2D matrix `D_con` (e.g., from HMM Gamma values)
    back into its original 3D shape. If the original session matrix `D_original` is provided,
    the function will infer the number of timepoints, trials, and channels from its shape. 
    Otherwise, the user must provide the correct dimensions.
    
    Parameters:
    ------------
    D_con (numpy.ndarray): 
        A 2D concatenated D-matrix of shape ((n_timepoints * n_entities), n_features).
    D_original (numpy.ndarray, optional): 
        A 3D array containing the original D-matrices for each session, with shape (n_timepoints, n_entities, n_features).
    n_timepoints (int, optional): 
        A number of timepoints per trial, is required if `D_original` is not provided.
    n_entities (int, optional): 
        A number of e.g. trials or subjects per session, is required if `D_original` is not provided.
    n_features (int, optional): 
        Number of features (e.g. channels), required if `D_original` is not provided.

    Returns:
    ---------
    D_reconstruct (numpy.ndarray): 
        A 3D array containing the reconstructed D-matrix for each session, with shape (n_timepoints, n_entities, n_features).

    Raises:
    --------
    ValueError: 
        If `D_original` is provided and is not a 3D numpy array, or if the provided dimensions do not match the shape of `D_con`.
        If `n_timepoints`, `n_trials`, or `n_features` are not provided when `D_original` is missing.
        If the shape of `D_con` does not match the expected dimensions based on the input parameters.
    """
    # Input validation and initialization
    if D_original is not None and len([arg for arg in [n_timepoints, n_entities, n_features] if arg is not None]) == 0:
        if not isinstance(D_original, np.ndarray) or D_original.ndim != 3:
            raise ValueError("Invalid input: D_original must be a 3D numpy array.")
        n_timepoints, n_entities, n_features = D_original.shape
        D_reconstruct = np.zeros_like(D_original)
    else:
        if None in [n_timepoints, n_entities, n_features]:
            raise ValueError("Invalid input: n_timepoints, n_trials, and n_features must be provided if D_original is not provided.")
        D_reconstruct = np.zeros((n_timepoints, n_entities, n_features))
    
    # Check if the shape of D_con matches the expected shape
    if D_con.shape != (n_timepoints * n_entities, n_features):
        raise ValueError("Invalid input: D_con does not match the expected shape.")

    # Assign values from D_con to D_reconstruct
    for i in range(n_entities):
        start_idx = i * n_timepoints
        end_idx = (i + 1) * n_timepoints
        D_reconstruct[:, i, :] = D_con[start_idx:end_idx, :]
    return D_reconstruct



def pad_vpath(vpath, lag_val, indices_tde=None):
    """
    Pad the Viterbi path with repeated first and last rows.

    This function adds padding to the beginning and end of a given Viterbi path (vpath) from a Hidden Markov Model 
    by repeating the first and last rows a specified number of times (lag_val). This is useful for maintaining 
    boundary conditions in scenarios such as sequence alignment or signal processing where the state 
    transitions need to be preserved.

    Parameters:
    ------------
    vpath (numpy.ndarray): 
        A 2D array representing the Viterbi path, where each row corresponds to a specific state in the HMM 
        and each column represents different features or observations.
    lag_val (int): 
        The number of times to repeat the first and last rows for padding.
    indices_tde (list of tuples, optional): 
        A list of tuples, where each tuple contains the start and end indices of individual sequences 
        within the vpath. If provided, padding is applied to each sequence separately.
    indices_tde (numpy.ndarray): 
        Is a 2D array where each row represents the start and end index for a session for the TDE-HMM dataset.
        
    Returns:
    ---------
    numpy.ndarray: 
        A new 2D array containing the padded Viterbi path, with shape ((lag_val + n_rows + lag_val), n_features), 
        where n_rows is the original number of rows in vpath and n_features is the number of features.

    Raises:
    --------
    ValueError: 
        If `lag_val` is not a positive integer, or if `vpath` is not a 2D numpy array.
    """
    # Input validation
    if not isinstance(vpath, np.ndarray) or vpath.ndim != 2:
        raise ValueError("Invalid input: vpath must be a 2D numpy array.")
    if not isinstance(lag_val, int) or lag_val <= 0:
        raise ValueError("Invalid input: lag_val must be a positive integer.")
    if indices_tde is not None:
        if not isinstance(indices_tde, np.ndarray) or vpath.ndim != 2 or indices_tde[-1][-1]!=len(vpath):
            raise ValueError("Invalid input: indices_tde does not match with vpath.")
    if indices_tde is None:
        # Get the first and last rows
        first_row = vpath[0]
        last_row = vpath[-1]

        # Create padding for the beginning and the end
        beginning_padding = np.tile(first_row, (lag_val, 1))
        end_padding = np.tile(last_row, (lag_val, 1))

        # Concatenate the padding with the original vpath
        vpath_pad = np.vstack((beginning_padding, vpath, end_padding))
    else:
        # Multiple sequence padding based on indices_tde
        vpath_list=[]
        for start, end in indices_tde:
            # Get the first and last rows
            first_row = vpath[start]
            last_row = vpath[end-1]

            # Create padding for the beginning and the end
            beginning_padding = np.tile(first_row, (lag_val, 1))
            end_padding = np.tile(last_row, (lag_val, 1))

            # Append each session to the empty list
            vpath_list.append(np.vstack((beginning_padding, vpath[start:end], end_padding)) )
            # Concatenate the padding with the original vpath
            vpath_pad = np.concatenate(vpath_list,axis=0)
    return vpath_pad

def get_event_epochs(input_data, index_data, filtered_R_data, event_markers, 
                               fs, fs_target=None, ms_before_stimulus=0, epoch_window_tp=None):
    """
    Extract time-locked data epochs based on stimulus events.

    This function processes 2D input data to extract epochs aligned to specific stimulus events. 
    The epochs are extracted based on provided event files and are resampled to the target rate. 
    The function also returns relevant indices and concatenates filtered R data across sessions.

    Parameters:
    ------------
    input_data (numpy.ndarray): 
        2D array containing gamma values for the session, structured as ((number of timepoints * number of trials), number of states).
    index_data (numpy.ndarray): 
        2D array containing preprocessed indices for the session.
    filtered_R_data (list): 
        List of filtered R data arrays for each session based on the events.
    event_markers (list): 
        List of event information for each session.
    fs (int, optional): 
        The original sampling frequency in Hz. Defaults to 1000 Hz.
    fs_target (int, optional): 
        The target sampling frequency in Hz after resampling. Defaults to 250 Hz.
    ms_pre_stimulus (int, optional): 
        Time in milliseconds to offset the start of the epoch before the stimulus onset. Defaults to 0 ms.
    epoch_window_tp
        Epoch window length in time points. If None, a default duration of 1 second (equal to fs_target) is used.

    Returns:
    ---------
    epoch_data (numpy.ndarray): 
        3D array of extracted data epochs, structured as (number of timepoints, number of trials, number of states).
    epoch_indices (numpy.ndarray): 
        Array of indices corresponding to the extracted epochs for each session.
    concatenated_R_data (numpy.ndarray): 
        Concatenated array of R data across all sessions.
    """
    if fs_target== None:
        fs_target = fs.copy()
    # Calculate the downsampling factor
    downsampling_factor = fs / fs_target
    # Set default duration to 1 second if None
    epoch_window_tp = fs_target if epoch_window_tp is None else epoch_window_tp 
    # Calculate the shift for the stimulus onset
    stimulus_shift = ms_before_stimulus / downsampling_factor if ms_before_stimulus != 0 else 0

    # Initialize lists to store gamma epochs, filtered R data, and index data
    data_epochs_list = []  
    filtered_R_data_list = []  
    valid_epoch_counts = []  

    # Iterate over each event file corresponding to a session
    for idx, events in enumerate(event_markers):
        # Extract data values for the specific session using preprocessed indices
        data_session = input_data[index_data[idx, 0]:index_data[idx, 1], :]


        # Downsample the event time indices
        downsampled_events = (events[:, 0] / downsampling_factor).astype(int)

        # Calculate differences between consecutive events
        event_differences = np.diff(downsampled_events, axis=0)

        # Identify valid events that are sufficiently spaced apart
        valid_event_indices = (event_differences >= epoch_window_tp)

        # Ensure the first event is included if it meets the downsample condition
        if event_differences[0] >= epoch_window_tp:
            valid_event_indices = np.concatenate(([True], valid_event_indices))

        # Filter events that meet the downsample condition
        valid_event_indices &= (len(data_session) - downsampled_events >= epoch_window_tp)

        # Select filtered event indices based on the downsample condition
        filtered_event_indices = downsampled_events[valid_event_indices]

        # Counter for the number of valid trials
        trial_count = 0  

        # Iterate over each filtered event
        for event_index in filtered_event_indices:
            start_index = int(event_index + stimulus_shift)  # Adjust start index to include time before stimulus
            end_index = int(start_index + epoch_window_tp)  # Define end index for the epoch

            # Append the data for this epoch to the data_epochs_list
            data_epochs_list.append(data_session[start_index:end_index, :])

            trial_count += 1  # Increment the trial counter

        # Append the filtered R data to the filtered_R_data_list
        filtered_R_data_list.append(filtered_R_data[idx][valid_event_indices])

        # Store the count of valid epochs for this session in the valid_epoch_counts
        valid_epoch_counts.append(np.sum(valid_event_indices))

    # Convert the data_epochs_list to a NumPy array and transpose it for correct dimensions
    epoch_data = np.transpose(np.array(data_epochs_list), (1, 0, 2))

    # Concatenate all filtered R data along the first axis
    epoch_R_data = np.concatenate(filtered_R_data_list, axis=0)

    # Calculate the indices for the epoch data using a custom function
    epoch_indices = get_indices_from_list(valid_epoch_counts, count_timestamps=False)

    # Return the processed data
    return epoch_data, epoch_indices, epoch_R_data

def categorize_columns_by_statistical_method(R_data, method, Nnull_samples, detect_categorical=False, category_limit=None,permute_beta=False, comparison_statistic=False):
    """
    Detects categorical columns in R_data and categorizes them for later statistical testing (t-tests, F-tests, etc.).
    This function helps identify which columns are binary or categorical, and applies permutation inference based on the test statistics.

    Parameters:
    -----------
    R_data : numpy.ndarray 
        The 3D array (e.g., N x T x q) containing the data where categorical values need to be detected.
    method : str, optional
        The statistical method applied to the columns. Supported values are:
        "univariate", "multivariate", "osr", "osa".
    detect_categorical : bool, list, or numpy.ndarray, optional (default=False)
        If True, automatically identify categorical columns. 
        If a list or ndarray, the provided column indices are used for categorization.
    category_limit : int or None, optional (default=None)
        The maximum allowed number of unique categories for an F-test. Used to prevent misidentifying 
        continuous variables (like age) as categorical.
    permute_beta : bool, optional (default=False)
        Determines whether to use permutation testing on regression beta values.
    comparison_statistic : str, optional (default="mean")
        The statistic used in pairwise comparisons for methods like "osr" or "osa". 
        Supported values are "mean" or "median".

    Returns:
    -----------
    category_columns : dict
        A dictionary with the following keys:
        - 't_stat_independent_cols': Columns where t-tests are applied (binary variables).
        - 'f_anova_cols': Columns where F-tests (ANOVA) are applied (categorical variables).
        - 'f_reg_cols': Columns to apply F-regression on (continuous variables).
        - Other keys depending on method, such as 'r_squared', 'corr_coef', or 'z_score' for different tests.
    """
    category_columns = {'t_stat_independent_cols': [], 'f_anova_cols': [], 'f_manova_cols': [], 'f_reg_cols': [], 'z_score': []} 
    idx_cols =np.arange(R_data.shape[-1])
    # Perform categorical detection based on detect_categorical input
    # This checks if detect_categorical is either True or a list/ndarray
    if method=="cca":
        pass
    elif detect_categorical == True:
        # Initialize variable
        if method!="multivariate" and not permute_beta:
            category_columns["t_stat_independent_cols"] = [col for col in range(R_data.shape[-1]) if np.unique(R_data[~np.isnan(R_data[:, col]), col]).size == 2]
        if category_limit != None:
            # If method is not "multivariate" or permute_beta is True, identify binary columns in R_data
            #if method == "multivariate" :
            #if method == "multivariate" and not permute_beta:    
            if method == "multivariate" and not permute_beta:
                category_columns["f_manova_cols"] = [
                                    col for col in range(R_data.shape[-1]) 
                                    if (np.unique(R_data[~np.isnan(R_data[:, col]), col]).size > 2)  # Ignore NaNs
                                    and (np.unique(R_data[~np.isnan(R_data[:, col]), col]).size < category_limit)
                                ]
                
            else:
                category_columns["f_anova_cols"] = [col for col in range(R_data.shape[-1]) 
                                                if np.unique(R_data[~np.isnan(R_data[:, col]), col]).size > 2  # Check if more than 2 unique values
                                                and np.unique(R_data[~np.isnan(R_data[:, col]), col]).size < category_limit] # Check if the data type is above category_limit
            # idx_test is a list of all binary and categorical columns
            idx_test = category_columns["t_stat_independent_cols"]+category_columns["f_anova_cols"]+category_columns["f_manova_cols"]
            # The remaining columns, which are not binary or categorical, are treated as continuous
            if permute_beta or Nnull_samples==1 and method=="multivariate":
                category_columns["f_reg_cols"] = idx_cols[~np.isin(idx_cols,idx_test)].tolist()
            elif method=="multivariate" and all(not v for v in category_columns.values())==False:
                category_columns["f_reg_cols"] = idx_cols[~np.isin(idx_cols,idx_test)].tolist()
            elif permute_beta is False and method=="univariate":
                category_columns["t_stat_pearson_cols"] = idx_cols[~np.isin(idx_cols,idx_test)].tolist() 

        else:
            unique_counts = [np.unique(R_data[0, :, col]).size for col in range(R_data.shape[-1])]
            if max(unique_counts) > category_limit:
                warnings.warn(
                    f"Detected more than {category_limit} unique numbers in column {idx_cols[np.array(unique_counts)>category_limit]} dataset. "
                    f"If this is not intended as categorical data, you can ignore this warning. "
                    f"Otherwise, consider defining 'category_limit' to set the maximum allowed categories or specifying the indices of categorical columns."
                )
                 
    # Handling cases where no categorical detection is requested
    else:
        # If method is either osr or osa, apply the chosen comparison_statistic (mean or median)
        if method == "osr" or method =="osa":
            category_columns[comparison_statistic] = 'all_columns'
    
    # Count non-empty keys
    non_empty_keys = [key for key, value in category_columns.items() if value]
    # If exactly one key has values, set it to "all_columns"
    if len(non_empty_keys) == 1:
        category_columns[non_empty_keys[0]] = "all_columns"
        if category_columns['f_reg_cols'] =='all_columns' and permute_beta:
            category_columns['f_reg_cols'] = [] # that is the default for across session test, so no need for that
    return category_columns

def calculate_regression_statistics(Din, Rin, reg_pinv, nan_values=False, no_t_stats= False):
    """
    Calculate the R-squared values for the regression of each dependent variable 
    in Rin on the independent variables in Din, while handling NaN values column-wise.

    Parameters:
    --------------
    Din (numpy.ndarray): 
        Input data matrix for the independent variables.
    Rin (numpy.ndarray): 
        Input data matrix for the dependent variables.
    reg_pinv (numpy.ndarray), default None: 
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
    nan_values (bool, optional): 
        A flag indicating there are NaN values.

    Returns:
    ----------  
        R2_stats (numpy.ndarray): Array of R-squared values for each regression.
    """
    # Dimension of D-matrix
    n, p = Din.shape
    if nan_values:
        Rin = np.expand_dims(Rin, axis=1) if Rin.ndim==1 else Rin
        
        q = Rin.shape[-1]
        R2_stats = np.zeros(q) 
        F_stats = np.zeros(q) 
        p_value = np.zeros(q) 
        t_stats = np.zeros((p, q))  # Initialize t-statistics matrix 
        df1 =p     
        df2_list = []  # To store df2 for each regression
        
        # Calculate t-statistic for each pair of columns (D_column, R_data)
        for q_i in range(q):
            R_column = np.expand_dims(Rin[:, q_i],axis=1)
            valid_indices = np.all(~np.isnan(R_column), axis=1)
            # Calculate beta using the regularized pseudo-inverse of D_data
            beta = reg_pinv[:,valid_indices] @ R_column[valid_indices]  # Calculate regression_coefficients (beta)
            # Calculate the predicted values
            R_pred = Din[valid_indices] @ beta
            n_valid =sum(valid_indices)
            df2 = n_valid - p  # Compute df2 for the current regression
            df2_list.append(df2)  # Store df2 
            # Calculate the total sum of squares (tss)
            tss = np.sum((R_column[valid_indices] - np.mean(R_column[valid_indices], axis=0))**2, axis=0)
            # Calculate the residual sum of squares (rss)
            rss = np.sum((R_column[valid_indices]-R_pred)**2, axis=0)
            # Calculate R^2 for the current dependent variable
            R2_stats[q_i] = 1 - (rss / tss)
            # Calculate F_stats
            F_stats[q_i] = (R2_stats[q_i] / df1) / ((1 - R2_stats[q_i]) / df2)
            # Calculate parametric p-value
            p_value[q_i] = 1 - f.cdf(F_stats[q_i], df1, df2)

            if no_t_stats== True:
                t_stats = None
            else:
                # Expand Rin, if it is 1D
                if R_column.ndim == 1:
                    R_column = np.expand_dims(R_column, axis=1) 
                # Center the data
                Din_centered = Din[valid_indices] - Din[valid_indices].mean(axis=0) 
                Rin_centered = R_column[valid_indices] - R_column[valid_indices].mean(axis=0) 
                # Compute correlation coefficients efficiently
                numerator = Din_centered.T @ Rin_centered
                denom_x = np.sqrt(np.sum(Din_centered**2, axis=0))[:, np.newaxis]
                denom_y = np.sqrt(np.sum(Rin_centered**2, axis=0))
                correlation_matrix = numerator / (denom_x @ denom_y[np.newaxis, :])
                # Convert correlation coefficients to t-statistics
                t_stats[:, q_i] = np.squeeze(correlation_matrix * np.sqrt(n - 2) / np.sqrt(1 - correlation_matrix**2))
      
    else:
        # Calculate regression_coefficients (beta)
        beta= reg_pinv @ Rin  
        # Calculate the predicted values
        R_pred = Din @ beta   
        # Calculate the residual sum of squares (rss)
        rss = np.sum((Rin-R_pred)**2, axis=0)
        # Calculate the total sum of squares (tss)
        tss = np.sum((Rin - np.mean(Rin, axis=0))**2, axis=0)
        # Calculate R^2 for the current dependent variable
        R2_stats = 1 - (rss / tss)
        # Degress of freedom
        df1 = p  
        df2 = n - p
        # Calculate F_stats
        F_stats = (R2_stats / df1) / ((1 - R2_stats) / df2)
        # Calculate parametric p-value
        p_value = 1 - f.cdf(F_stats, df1, df2)
        if no_t_stats== True:
            t_stats = None
        else:
            # Expand Rin, if it is 1D
            if Rin.ndim == 1:
                Rin = np.expand_dims(Rin, axis=1) 

            # Center the data
            Din_centered = Din - Din.mean(axis=0)
            Rin_centered = Rin - Rin.mean(axis=0)

            # Compute correlation coefficients efficiently
            numerator = Din_centered.T @ Rin_centered
            denom_x = np.sqrt(np.sum(Din_centered**2, axis=0))[:, np.newaxis]
            denom_y = np.sqrt(np.sum(Rin_centered**2, axis=0))
            correlation_matrix = numerator / (denom_x @ denom_y[np.newaxis, :])

            # Convert correlation coefficients to t-statistics
            t_stats = correlation_matrix * np.sqrt(n - 2) / np.sqrt(1 - correlation_matrix**2)
    #if len(F_stats) == Rin.shape[-1] and F_stats.ndim==1:
    if F_stats.ndim==0:
        F_stats = np.expand_dims(F_stats,axis=0)
        R2_stats = np.expand_dims(R2_stats,axis=0)
        p_value = np.expand_dims(p_value,axis=0)
    return R2_stats, F_stats, t_stats, p_value

def ols_regression_stats(Din, Rin, R_pred, Nnull_samples, no_t_stats, valid_indices=None):
    """
    Compute regression statistics (R-squared, F-statistics, and t-statistics) for ordinary least squares (OLS) regression 
    across multiple sessions.

    Parameters:
    --------------
    Din (numpy.ndarray): 
        The design matrix (D-matrix) containing independent variables. Shape: (n, p), 
        where n is the number of observations and p is the number of predictors.   
    Rin (numpy.ndarray): 
        The dependent variable matrix (R-matrix) containing observed values. Shape: (n, q), 
        where q is the number of dependent variables.  
    R_pred (numpy.ndarray): 
        The predicted response matrix. Shape: (n, q).
    Nnull_samples (numpy.ndarray): 
        Number of permutations
    Returns:
    ----------
    R2_stats (numpy.ndarray): 
        R-squared values for each dependent variable. Shape: (q,).
    F_stats (numpy.ndarray): 
        F-statistics for each dependent variable. Shape: (q,).
    t_stats (numpy.ndarray): 
        The calculated t-statistics for each predictor-outcome pair. Shape: (p, q).

    Notes:
    ----------
    This approach is based on the statistical relationship between Pearson correlation coefficients 
    and t-distributions. By relying on R_pred rather than beta coefficients, the permutation process 
    directly influences the t-statistics.
    """

    n, p = Din.shape
    Rin = np.expand_dims(Rin,axis=1) if Rin.ndim==1 else Rin.copy()
    Rin = Rin if valid_indices is None else Rin[valid_indices].copy()
    Din = Din if valid_indices is None else Din[valid_indices].copy()
    R_pred = np.expand_dims(R_pred,axis=1) if R_pred.ndim==1 else R_pred
    #q = Rin.shape[-1] 
    # Initialize matrices for statistics
    rss = np.sum((Rin - R_pred) ** 2, axis=0)  # Residual sum of squares (shape: q)
    tss = np.sum((Rin - np.nanmean(Rin, axis=0)) ** 2, axis=0)  # Total sum of squares (shape: q)

    # Calculate R^2 for each dependent variable
    R2_stats = 1 - (rss / tss)

    # Degrees of freedom
    df1 = p
    df2 = n - p

    # Calculate F-statistics for each dependent variable
    F_stats = (R2_stats / df1) / ((1 - R2_stats) / df2)
    
    # Calculate parametric p-value
    p_value = 1 - f.cdf(F_stats, df1, df2)
    # Calculate permuted t-statistics based on correlations (p x q)
    if no_t_stats== True:
        t_stats = None
    else:
        # Expand Rin, if it is 1D
        if R_pred.ndim == 1:
            R_pred = np.expand_dims(R_pred, axis=1) 

        # Center the data
        Din_centered = Din - Din.mean(axis=0)
        R_centered = R_pred - R_pred.mean(axis=0) if Nnull_samples >1 else Rin - Rin.mean(axis=0)

        # Compute correlation coefficients efficiently
        numerator = Din_centered.T @ R_centered
        denom_x = np.sqrt(np.sum(Din_centered**2, axis=0))[:, np.newaxis]
        denom_y = np.sqrt(np.sum(R_centered**2, axis=0))
        correlation_matrix = numerator / (denom_x @ denom_y[np.newaxis, :])

        # Convert correlation coefficients to t-statistics
        t_stats = correlation_matrix * np.sqrt(n - 2) / np.sqrt(1 - correlation_matrix**2)

    return R2_stats, F_stats, t_stats, p_value


def preprocess_response(Rin):
    """
    Converting R_in into to dummy variables, and centering the data around zero.

    Parameters:
    --------------
    Rin (numpy.ndarray or pandas.Series): 
        Input array representing the dependent variable. Can be 1D or 2D. 
        If 1D, it will be flattened and converted to dummy variables.

    Returns:
    ----------
    Rin_centered (numpy.ndarray): 
        A 2D array of centered dummy variables based on the input response.
    """
    # Ensure Rin is a 2D array
    Rin = np.atleast_2d(Rin)

    # Handle the case where Rin was originally a 1D array
    if Rin.shape[0] == 1:
        Rin = Rin.T  # Transpose to make it a column vector if necessary

    # Convert to dummy variables and center
    Rin = pd.get_dummies(Rin.flatten(), drop_first=False).values
    #Rin_centered = Rin - np.mean(Rin, axis=0)  # Center the response data around 0

    return Rin

def calculate_anova_f_test(Din, Rin, idx_data=None, permute_beta=False, perm=0, nan_values=False, beta= None, no_t_stats=False, Nnull_samples=False, method=None):
    """
    Calculate the f-test values for the regression of each dependent variable
    in Rin on the independent variables in Din, while handling NaN values column-wise.

    Parameters:
    --------------
    Din (numpy.ndarray): 
        Input data matrix for the independent variables.
    Rin (numpy.ndarray): 
        Input data matrix for the dependent variables.
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
    # Initialize variables
    n, p = Din.shape
    # Identify unique groups in the first column of Y
    unique_groups = np.unique(Rin[~np.isnan(Rin)])

    if permute_beta and method != "univariate":
        # Valid indices for the dependent variable
        valid_indices_R = ~np.isnan(Rin)
        valid_indices_D = np.all(~np.isnan(Din), axis=1)
        valid_indices = valid_indices_R & valid_indices_D if np.any(valid_indices_D) else valid_indices_R
        # Calculate the predicted values using permuted beta
        R_pred =ols_regression_sessions(Rin[valid_indices], Din, idx_data, beta, perm, permute_beta, nan_values=~np.all(valid_indices),  valid_indices=valid_indices, method=method)
        # Compute observed F-ANOVA statistic based on groups
        groups = [R_pred[Rin[valid_indices] == group] for group in unique_groups]
        # Compute the global F-ANOVA statistic
        f_statistic, p_value = f_oneway(*groups)
        if no_t_stats== True:
            t_stats = None
        else:
            # Expand Rin, if it is 1D
            if R_pred.ndim == 1:
                R_pred = np.expand_dims(R_pred, axis=1) 
            # Center the data
            Din_centered = Din - Din.mean(axis=0) 
            R_centered = R_pred - R_pred.mean(axis=0) if Nnull_samples >1 else Rin[valid_indices][:,np.newaxis]  - Rin[valid_indices][:,np.newaxis].mean(axis=0)
            if len(Din_centered)!=len(R_centered): 
                Din_centered= Din_centered[valid_indices]

            # Compute correlation coefficients efficiently
            numerator = Din_centered.T @ R_centered 
            denom_x = np.sqrt(np.sum(Din_centered**2, axis=0))[:, np.newaxis]
            denom_y = np.sqrt(np.sum(R_centered**2, axis=0))
            correlation_matrix = numerator / (denom_x @ denom_y[np.newaxis, :])
            # Convert correlation coefficients to t-statistics
            t_stats = correlation_matrix * np.sqrt(n - 2) / np.sqrt(1 - correlation_matrix**2)
    elif permute_beta and method == "univariate":
        # Perform columnwise 
        if Rin.ndim == 1:
            Rin = Rin[:, np.newaxis]  # Ensure Y is (N, q)

        p = Din.shape[1]  # Number of continuous variables
        q = Rin.shape[1]  # Number of categorical variables
        f_statistic = np.zeros((p, q))
        p_value = np.zeros((p, q))
        t_stats = None

        # Valid indices for the dependent variable
        valid_indices_R = np.squeeze(~np.isnan(Rin))  # Boolean mask for non-NaN values
        nan_values = np.any(np.isnan(Din))  # Check for NaNs in Din
        
        # Calculate the predicted values using permuted beta
        R_pred =ols_regression_sessions(Rin[valid_indices_R], Din[valid_indices_R,:], idx_data, beta, perm, permute_beta, nan_values,  valid_indices=None, method=method)
        for p_i in range(p):  # Loop over D columns
            valid_indices = ~np.isnan(Din[:, p_i]) & valid_indices_R  # Boolean mask for non-NaN values
            # Compute observed F-ANOVA statistic based on groups
            groups = [R_pred[p_i][Rin[valid_indices] == group] for group in unique_groups]
            # Compute the global F-ANOVA statistic
            f_statistic[p_i,], p_value[p_i,] = f_oneway(*groups)

    elif method == 'univariate':
        # Perform columnwise 
        if Rin.ndim == 1:
            Rin = Rin[:, np.newaxis]  # Ensure Y is (N, q)

        p = Din.shape[1]  # Number of continuous variables
        q = Rin.shape[1]  # Number of categorical variables
        f_statistic = np.zeros((p, q))
        p_value = np.zeros((p, q))
        t_stats = None

        for p_i in range(p):  # Loop over D columns
            Din_column = Din[:, p_i]
            for q_j in range(q):  # Loop over each categorical variable in R
                Rin_column = Rin[:, q_j]
                # Remove NaNs from both D and R
                valid_indices = ~np.isnan(Din_column) & ~np.isnan(Rin_column)
                Din_filtered = Din_column[valid_indices]
                Rin_filtered = Rin_column[valid_indices]
                # Get unique group labels from Rin
                unique_groups = np.unique(Rin_filtered)
                # Group Din values according to Rin categories
                groups = [Din_filtered[Rin_filtered == group] for group in unique_groups]
                f_statistic[p_i, q_j], p_value[p_i, q_j] = f_oneway(*groups)
    else:
        # Valid indices for the dependent variable
        valid_indices = np.squeeze(~np.isnan(Rin))  # for multicolumn Rin (one-hot)
        # Compute observed F-ANOVA statistic based on groups
        groups = [Rin[valid_indices] == group for group in unique_groups]
        # Compute the global F-ANOVA statistic
        f_statistic, p_value = f_oneway(*groups)
        if no_t_stats== True:
            t_stats = None
        else:
            # Expand Rin, if it is 1D
            if Rin.ndim == 1:
                Rin = np.expand_dims(Rin, axis=1) 
            # Center the data - already centered
            # Din_centered = Din[valid_indices] - Din[valid_indices].mean(axis=0)
            # Rin_centered = Rin[valid_indices] - Rin[valid_indices].mean(axis=0)
            # Compute correlation coefficients efficiently
            numerator = Din.T @ Rin
            denom_x = np.sqrt(np.sum(Din**2, axis=0))[:, np.newaxis]
            denom_y = np.sqrt(np.sum(Rin**2, axis=0))
            correlation_matrix = numerator / (denom_x @ denom_y[np.newaxis, :])
            # Convert correlation coefficients to t-statistics
            t_stats = correlation_matrix * np.sqrt(n - 2) / np.sqrt(1 - correlation_matrix**2)
    return f_statistic, t_stats, p_value

def calculate_manova_f_test(Din, Rin, nan_values, no_t_stats=False):
    """
    Calculate the f-test values for the regression of each dependent variable
    in Rin on the independent variables in Din, while handling NaN values column-wise.

    Parameters:
    --------------
    Din (numpy.ndarray): 
        Input data matrix for the independent variables.
    Rin (numpy.ndarray): 
        Input data matrix for the dependent variables.
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
        Rin = np.squeeze(Rin)
        if Rin.ndim == 1:
            Rin = Rin[:, np.newaxis]  # shape (n, q)
        n, p = Din.shape
        q = Rin.shape[1]  # number of categorical variables
        p_values = np.zeros(q)
        f_statistics = np.zeros(q)
        t_stats = np.zeros((p,q))
        for q_i in range(q):
            # Filter out NaNs in Rin
            valid_indices = np.squeeze(~np.isnan(Rin[:,q_i]))  # for multicolumn Rin (one-hot)
            Rin_valid = Rin[valid_indices, q_i]
            Din_valid = Din[valid_indices]
            # One-hot encode categorical variable
            unique_vals, Rin_numeric = np.unique(Rin_valid, return_inverse=True)
            Rin_onehot = np.eye(len(unique_vals))[Rin_numeric]
            q_one = Rin_onehot.shape[1] # shape of the one-hot encoded matrix
            # ⚠️ NOTE:
            # In MANOVA, we model the continuous data (Din) as the **dependent variable**,
            # and the group labels (Rin_onehot) as the **independent variable**.
            # So we are fitting: Din = Rin_onehot @ beta + error

            # Fit model (predict Din from group membership)
            beta = np.linalg.pinv(Rin_onehot) @ Din_valid  # shape (q, p)
            Din_pred = Rin_onehot @ beta  # shape (n, p)

            # Compute residual and hypothesis matrices
            E = Din_valid - Din_pred                          # residuals
            H = Din_pred - Din_valid.mean(axis=0)            # model (hypothesis) component

            E_cov = E.T @ E  # Error SSCP
            H_cov = H.T @ H  # Hypothesis SSCP

            # Wilks’ lambda
            wilks_lambda = np.linalg.det(E_cov) / np.linalg.det(E_cov + H_cov)

            # Convert to approximate F-statistic and p-value
            df1 = q_one * p
            df2 = n - 0.5 * (p + q_one + 1)
            f_statistics[q_i] = ((1 - wilks_lambda) / wilks_lambda) * (df2 / df1)
            p_values[q_i] = f.sf(f_statistics[q_i], df1, df2)
            

            if not no_t_stats:
                Rin_column = Rin[:,q_i][:, np.newaxis]
                # Center the data
                ## Rember that Din and Rin are already centered, so no need to center it again
                # Compute correlation coefficients
                numerator = Din.T @ Rin_column
                denom_x = np.sqrt(np.sum(Din**2, axis=0))[:, np.newaxis]  # shape (p, 1)
                denom_y = np.sqrt(np.sum(Rin_column**2, axis=0))                    # shape (q,)
                correlation_matrix = numerator / (denom_x @ denom_y[np.newaxis, :])  # shape (p, q)

                # Convert to t-stats
                t_stat= correlation_matrix * np.sqrt(n - 2) / np.sqrt(1 - correlation_matrix**2)
                t_stats[:,q_i] =t_stat.ravel()
            else:
                t_stats= None
    else:
        Rin = np.squeeze(Rin)
        if Rin.ndim == 1:
            Rin = Rin[:, np.newaxis]  # shape (n, q)
        n, p = Din.shape
        q = Rin.shape[1]  # number of categorical variables
        p_values = np.zeros(q)
        f_statistics = np.zeros(q)
        t_stats = np.zeros((p,q))

        for q_i in range(q):
            col = Rin[:, q_i]
            unique_vals, encoded = np.unique(col, return_inverse=True)
            Rin_onehot = np.zeros((n, len(unique_vals)))
            Rin_onehot[np.arange(n), encoded] = 1
            q_one = Rin_onehot.shape[1]

            # ⚠️ NOTE:
            # In MANOVA, we model the continuous data (Din) as the **dependent variable**,
            # and the group labels (Rin_onehot) as the **independent variable**.
            # So we are fitting: Din = Rin_onehot @ beta + error
            # Fit model (predict Din from group membership)
            beta = np.linalg.pinv(Rin_onehot) @ Din
            Din_pred = Rin_onehot @ beta

            # Residual and hypothesis matrices
            E = Din - Din_pred  # residuals
            H = Din_pred - Din.mean(axis=0) # model (hypothesis) component
            E_cov = E.T @ E
            H_cov = H.T @ H

            # Add small regularization to avoid singular matrices
            eps = np.finfo(float).eps
            E_cov += eps * np.eye(E_cov.shape[0]) # Error SSCP
            H_cov += eps * np.eye(H_cov.shape[0]) # Hypothesis SSCP

            # Wilks’ lambda
            wilks_lambda = np.linalg.det(E_cov) / np.linalg.det(E_cov + H_cov)
            df1 = q_one * p
            df2 = n - 0.5 * (p + q_one + 1)
            f_statistics[q_i] = ((1 - wilks_lambda) / wilks_lambda) * (df2 / df1)
            p_values[q_i] = f.sf(f_statistics[q_i], df1, df2)

            # Optional t-stats
            if not no_t_stats:
                # Center the data
                ## Rember that Din and Rin are already centered, so no need to center it again
                # Compute correlation coefficients
                Rin_column = Rin[:,q_i][:, np.newaxis]
                numerator = Din.T @ Rin_column
                denom_x = np.sqrt(np.sum(Din**2, axis=0))[:, np.newaxis]  # shape (p, 1)
                denom_y = np.sqrt(np.sum(Rin_column**2, axis=0))                    # shape (q,)
                correlation_matrix = numerator / (denom_x @ denom_y[np.newaxis, :])  # shape (p, q)
                # Convert to t-stats
                t_stat= correlation_matrix * np.sqrt(n - 2) / np.sqrt(1 - correlation_matrix**2)
                t_stats[:,q_i] =t_stat.ravel()
            else:
                t_stats= None

    return f_statistics, t_stats, p_values

def calculate_beta_session(reg_pinv, Rin, Din, test_indices_list, train_indices_list):
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
    category_limit : int or None, optional, default=10
        Maximum allowed number of categories for F-test. Acts as a safety measure for columns 
        with integer values, like age, which may be mistakenly identified as multiple categories.  
    """

    # detect nan values
    nan_values = np.any(np.isnan(Rin)) or np.any(np.isnan(Din))

    # Do it columnwise if NaN values are detected
    if nan_values:
        Rin = np.expand_dims(Rin, axis=1) if Rin.ndim==1 else Rin
        q = Rin.shape[-1]
        p = Din.shape[-1]
        beta = []
        if len(train_indices_list)!=Rin.shape[-1]:
            # Calculate beta coefficients for each session
            beta_hat = np.zeros((len(train_indices_list[0]),p,q))
            for q_i in range(q):
                # Go through each column of R
                for p_i in range(p):
                    # Indentify columns with NaN values
                    R_column = np.expand_dims(Rin[:, q_i],axis=1)
                    nan_mask = np.isnan(np.expand_dims(Din,axis=1)).any(axis=1) if Din.ndim==1 else np.isnan(Din).any(axis=1)
                    valid_indices = np.all(~np.isnan(R_column), axis=1) & ~nan_mask
                    
                    for session, indices in enumerate(train_indices_list[0]):
                        #session_beta[idx,p_i] = np.expand_dims(reg_pinv[p_i,indices][valid_indices[indices]] @ Rin[indices, q_i][valid_indices[indices]],axis=1)
                        beta_hat[session,p_i, q_i] = reg_pinv[p_i,indices][valid_indices[indices]] @ Rin[indices, q_i][valid_indices[indices]]
            beta.append(np.array(beta_hat))
        else:
            # Precompute NaN masks
            D_expanded = np.expand_dims(Din, axis=1) if Din.ndim == 1 else Din
            nan_mask = np.isnan(D_expanded).any(axis=1)
            indices_range = np.arange(len(nan_mask))
            # Calculate beta coefficients for each session for every predictor
            for col in range(Rin.shape[-1]):
                beta_hat = []
                
                R_column = Rin[:, col, np.newaxis]  # Avoid redundant expand_dims calls
                valid_indices = ~np.isnan(R_column).any(axis=1) & ~nan_mask
                nan_indices = indices_range[~valid_indices]

                for indices in train_indices_list[col]:
                    idx_ses = indices_range[indices]
                    valid_ses = idx_ses[~np.isin(idx_ses, nan_indices)]  # Filter out NaN indices
                    session_beta = reg_pinv[:, valid_ses] @ Rin[valid_ses, col, np.newaxis]
                    beta_hat.append(session_beta)

                beta.append(np.array(beta_hat))

    else:
        # permute beta and calulate predicted values
        #beta, test_indices_list, train_indices_list = calculate_ols_beta(reg_pinv, Rin, idx_data, category_limit, test_indices_list, train_indices_list)
        beta = []
        if len(train_indices_list)!=Rin.shape[-1]:
            beta_hat = []
            for indices in train_indices_list[0]:
                session_beta = reg_pinv[:,indices] @ Rin[indices, :]
                beta_hat.append(session_beta)
            beta.append(np.array(beta_hat))
        else:
            for col in range(Rin.shape[-1]):
                beta_hat = []
                for indices in train_indices_list[col]:
                    session_beta = np.expand_dims(reg_pinv[:,indices] @ Rin[indices, col],axis=1)
                    beta_hat.append(session_beta)
                beta.append(np.array(beta_hat))
    return beta, test_indices_list, train_indices_list
        
def train_test_indices(Rin, Din, idx_data, method, category_limit= 10):
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
    
    seed =0
    np.random.seed(seed)  # Set seed for reproducibility
    # detect nan values
    nan_values = np.sum(np.isnan(Rin))>0
    # Expand dimension of Rin if it is 1D
    Rin = np.expand_dims(Rin, axis=1) if Rin.ndim==1 else Rin
    # Count unique non-NaN values per column
    uniq_counts = np.array([len(np.unique(Rin[~np.isnan(Rin[:, col]), col])) for col in range(Rin.shape[-1])]) # Count unique non-NaN values per column
    cat_cols = np.where(uniq_counts < category_limit)[0] # Find categorical columns
    cat_col_idx = cat_cols[0] if cat_cols.size == 1 else (None if cat_cols.size > 1 else None)

    FLAG_CAT = 1 if cat_col_idx is not None and nan_values==False  else 0 # Store categorical column index or default to 0
    # Initialise flags and index lists
    FLAG_CONTINUOUS = False  # Flag to track if a continuous split has been created
    shared_train_indices = None  # Store train indices for continuous values
    shared_test_indices = None  # Store test indices for continuous values
    indices_list = []
    # Initialise lists to store train-test splits for each column
    train_indices_list = [[] for _ in range(Rin.shape[-1])] if FLAG_CAT==0 and method !='cca' else [[] for _ in range(1)]
    test_indices_list = [[] for _ in range(Rin.shape[-1])] if FLAG_CAT ==0 and method !='cca' else [[] for _ in range(1)]

    if nan_values and method!='cca':
        num_cols = Rin.shape[-1]
        for col in range(num_cols):
            # Handle NaN values by identifying columns with NaN
            indices = np.all(~np.isnan(Rin[:, col]), axis=1) if Rin[:, col].ndim > 1 else ~np.isnan(Rin[:, col])
            indices_range = np.arange(len(indices))
            nan_indices = indices_range[~indices]
            bool_data = ~np.isin(indices_range, nan_indices)
            idx_data_update = update_indices(~indices, idx_data)

            # Make a train-test split, which we are going to estimate the beta's from for each session.
            for start, end in idx_data_update:
                idx_range = np.arange(start, end)
                # Find NaN values within the range
                matches = np.isin(nan_indices, idx_range)
                matched_values = nan_indices[matches]
                # Calculate the range length considering matched_values
                range_length = end - start - len(matched_values) if len(matched_values) > 0 else end - start
                # Extract valid values for stratification
                valid_values = Rin[idx_range[bool_data[idx_range]], col]
                unique_values, counts = np.unique(valid_values, return_counts=True)

                if len(unique_values) < category_limit:
                    if np.min(counts) == 1:
                        raise ValueError("Test aborted: At least one group has fewer than 2 unique values.")
                    # Create the train-test split
                    train_indices, test_indices = train_test_split(np.arange(range_length), test_size=0.5, stratify=valid_values, random_state=seed)
                else:
                    # Perform random split for continuous values
                    train_indices, test_indices = train_test_split(np.arange(end - start), test_size=0.5, random_state=seed)

                # Adjust the indices so they account for the NaN values
                for value in matched_values:
                    index_increase_train = train_indices >= value
                    train_indices = train_indices[index_increase_train] + 1
                    index_increase_test = test_indices >= value
                    test_indices = test_indices[index_increase_test] + 1  

                # Adjust train-test indices for the start position
                train_indices.sort()
                test_indices.sort()
                train_indices += start
                test_indices += start

                # Append results to the correct column
                train_indices_list[col].append(train_indices.copy())
                test_indices_list[col].append(test_indices.copy())

    elif FLAG_CAT or method=='cca':
        # Handle None by setting to 0, ensure array-like input, then extract first value
        col_index = 0 if cat_cols is None else np.atleast_1d(cat_cols)[0]
        if nan_values:
            valid_indices = ~np.isnan(Din[0,:]).any(axis=1) & ~np.isnan(Rin).any(axis=1)
            indices_range = np.arange(len(valid_indices))
            nan_indices = indices_range[~valid_indices]
            bool_data = ~np.isin(indices_range, nan_indices)
            idx_data_update = update_indices(~valid_indices, idx_data)

            # Make a train-test split, which we are going to estimate the beta's from for each session.
            for start, end in idx_data_update:
                idx_range = np.arange(start, end)
                # Find NaN values within the range
                matches = np.isin(nan_indices, idx_range)
                matched_values = nan_indices[matches]
                # Calculate the range length considering matched_values
                range_length = end - start - len(matched_values) if len(matched_values) > 0 else end - start
                # Extract valid values for stratification
                valid_values = Rin[idx_range[bool_data[idx_range]], col_index]
                unique_values, counts = np.unique(valid_values, return_counts=True)

                if len(unique_values) < category_limit:
                    if np.min(counts) == 1:
                        raise ValueError("Test aborted: At least one group has fewer than 2 unique values.")
                    # Create the train-test split
                    train_indices, test_indices = train_test_split(np.arange(range_length), test_size=0.5, stratify=valid_values, random_state=seed)
                else:
                    # Perform random split for continuous values
                    train_indices, test_indices = train_test_split(np.arange(end - start), test_size=0.5, random_state=seed)

                # Adjust the indices so they account for the NaN values
                for value in matched_values:
                    index_increase_train = train_indices >= value
                    train_indices = train_indices[index_increase_train] + 1
                    index_increase_test = test_indices >= value
                    test_indices = test_indices[index_increase_test] + 1 
                    
                # Adjust train-test indices for the start position
                train_indices.sort()
                test_indices.sort()
                train_indices += start
                test_indices += start
                test_indices_list[0].append(test_indices)
                train_indices_list[0].append(train_indices)

        else:
            for start, end in idx_data:
                 # Use cat_cols if provided; otherwise, default to column 0
                unique_values, counts = np.unique(Rin[start:end, col_index], return_counts=True)
                if len(unique_values) < category_limit:
                    if np.min(counts) == 1:
                        raise ValueError("Test aborted: At least one group has fewer than 2 unique values.")
                    train_indices, test_indices = train_test_split(
                        np.arange(end - start), test_size=0.5, stratify=Rin[start:end, col_index], random_state=seed
                    )
                else:
                    # Perform random split for continuous values
                    train_indices, test_indices = train_test_split(
                        np.arange(end - start), test_size=0.5, random_state=seed
                    )
                train_indices.sort()
                test_indices.sort()
                train_indices += start
                test_indices += start
                test_indices_list[0].append(test_indices)
                train_indices_list[0].append(train_indices)
       
    else: 
        # Perform columnwise operation when no NaN values are detected
        for start, end in idx_data:
            train_set_length = None  # Keep track of expected train set length
            for col in range(Rin.shape[-1]):
                unique_values, counts = np.unique(Rin[start:end,col], return_counts=True)
                if len(unique_values)<category_limit:
                    if np.min(counts)==1:
                        raise ValueError("Test aborted: At least one group has fewer than 2 unique values.")
                    train_indices, test_indices = train_test_split(np.arange(end-start), test_size=0.5, stratify=Rin[start:end,col], random_state=seed)
                    # Adjust indices to match the original dataset
                    train_indices = np.sort(train_indices + start)  
                    test_indices = np.sort(test_indices + start)  
                elif not FLAG_CONTINUOUS:
                    # Perform random split for continuous values
                    train_indices, test_indices = train_test_split(np.arange(end-start), test_size=0.5, random_state=seed)
                    FLAG_CONTINUOUS = True
                    shared_train_indices = train_indices.copy()
                    shared_test_indices = test_indices.copy()
                    Flag_shared = 1
                    # Adjust indices to match the original dataset
                    train_indices = np.sort(train_indices + start)  
                    test_indices = np.sort(test_indices + start)  
                else:
                    # Use the same split for all continuous columns
                    train_indices = shared_train_indices.copy()
                    test_indices = shared_test_indices.copy()
                # Ensure consistent train-test sizes across all columns
                if train_set_length is None:
                    train_set_length = len(train_indices)  # Set reference length
                elif len(train_indices) != train_set_length:
                    # Swap train and test sets if inconsistent
                    train_indices, test_indices = test_indices, train_indices
                # Append results to the correct column
                train_indices_list[col].append(train_indices.copy())
                test_indices_list[col].append(test_indices.copy())
                # Update shared train-test indices if they exist
                if shared_train_indices is not None:
                    shared_train_indices = np.array(train_indices)
                    shared_test_indices = np.array(test_indices)
            FLAG_CONTINUOUS = False
            Flag_shared == 0

    for col in range(Rin.shape[-1]):
        indices_list.append(~(~np.isnan(Rin[:,col])))
    np.random.seed(None)
    return train_indices_list, test_indices_list, indices_list

def train_test_update_indices(train_indices_list, test_indices_list, nan_indices):
    """
    Update train and test indices after removing specified NaN indices and re-indexing the remaining values.

    Parameters:
    --------------
    train_indices_list (list of list of int): 
        A list where each element is a list of sorted train indices for different segments.
    test_indices_list (list of list of int): 
        A list where each element is a list of sorted test indices for different segments.
    nan_indices (numpy.ndarray): 
        A list of indices to be removed from both train and test lists.
        
    Returns:
    ----------  
    train_indices_list_update (list of list of int): 
        The updated train indices with specified NaN indices removed and remaining indices re-indexed.
    test_indices_list_update (list of list of int): 
        The updated test indices with specified NaN indices removed and remaining indices re-indexed.
    """
    # Combine all indices from train and test, and sort them
    all_indices = sorted(set([idx for sublist in train_indices_list + test_indices_list for idx in sublist]))
    # Remove specified indices
    updated_all_indices = [idx for idx in all_indices if idx not in nan_indices]
    # Create a mapping from old indices to new indices
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(updated_all_indices)}
    # Update each sublist in train_indices and test_indices
    train_indices_list_update = [
        [index_map[idx] for idx in sublist if idx not in nan_indices]
        for sublist in train_indices_list
    ]
    test_indices_list_update = [
        [index_map[idx] for idx in sublist if idx not in nan_indices]
        for sublist in test_indices_list
    ]
    return train_indices_list_update, test_indices_list_update

def ols_regression_sessions(Rin, Din, idx_data, beta, perm, permute_beta=False, nan_values=False,  valid_indices=None, regression_statistics=False, Nnull_samples=False, no_t_stats= False, method=None):
    """
    Calculate predictions for ordinary least squares regression for across sessions test.

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
    regression_statistics (bool, optional): 
        Flag indicating whether to compute regression statistics (R-squared, F-statistics, and t-statistics). Default is False.

    Returns:
    --------
    R_pred (numpy.ndarray): 
        Predicted values from the OLS regression when `regression_statistics` is False.
        
    R2_stats (numpy.ndarray), F_stats (numpy.ndarray), t_stats (numpy.ndarray): 
        If `regression_statistics` is True, returns R-squared, F-statistics, and t-statistics for each session.
 
    """
    if nan_values and method =="univariate":
        
        R_pred = []
        # Permute rows only (sessions)
        beta_perm = beta[np.random.permutation(beta.shape[0]), :] if permute_beta and perm != 0 else beta
        for col in range(Din.shape[-1]):
            valid_indices = ~np.isnan(Din[:, col])
            invalid_indices = np.where(~valid_indices)[0]  # Get invalid indices efficiently
            D_column = Din[:, col][:, np.newaxis]
            # Use list to collect results per session (avoiding preallocation issues)
            R_pred_session = [
                (D_column[start:end][~np.isin(np.arange(start, end), invalid_indices)] @ beta_perm[idx, col])
                for idx, (start, end) in enumerate(idx_data)
            ]
            # Stack predictions directly
            R_pred.append(np.concatenate(R_pred_session, axis=0)[:,  np.newaxis])
        # Convert to NumPy array for efficiency
     
        return R_pred
    
    elif nan_values:
        R_pred = []
        valid_indices = ~np.isnan(Rin) if valid_indices is None else valid_indices
        valid_indices_range = np.arange(len(valid_indices))
        invalid_indices = valid_indices_range[~valid_indices]
        # Permute rows only (sessions)
        beta_perm = beta[np.random.permutation(beta.shape[0]), :] if permute_beta and perm != 0 else beta
        
        for idx, (start, end) in enumerate(idx_data):
            idx_range = np.arange(start, end)
            bool_data = ~np.isin(idx_range, invalid_indices)
            R_pred.append(np.dot(Din[idx_range[bool_data], :], beta_perm[idx]))
        R_pred =np.concatenate(R_pred, axis=0)
        if regression_statistics == False:
            return R_pred
        else:
            R2_stats, F_stats, t_stats, p_value = ols_regression_stats(Din, Rin, R_pred, Nnull_samples, no_t_stats, valid_indices)
            return R2_stats, F_stats, t_stats, p_value

    if method =="univariate":
        R_pred = []
        # Permute rows only (sessions)
        beta_perm = beta[np.random.permutation(beta.shape[0]), :] if permute_beta and perm != 0 else beta
        for col in range(Din.shape[-1]):
            R_pred_session = []
            D_column = Din[:, col][:, np.newaxis]
            for idx, (start, end) in enumerate(idx_data):
                idx_range = np.arange(start, end)
                R_pred_session.append(np.dot(D_column[idx_range,], beta_perm[idx, col]))
            R_pred.append(np.concatenate(R_pred_session, axis=0)[:,np.newaxis])
     
        return R_pred
        
    elif Rin.ndim==2: 
        R_pred = np.zeros((len(Din), Rin.shape[-1]))
        # Permute rows only (sessions)
        beta_perm = beta[np.random.permutation(beta.shape[0]), :] if permute_beta and perm != 0 else beta
        # Ensure beta_perm is 2D
        beta_perm = beta_perm[:, np.newaxis] if beta_perm.ndim == 1 else beta_perm
        # Iterate over idx_data without unnecessary checks
        for idx, (start, end) in enumerate(idx_data):
            if beta_perm[idx, :].ndim ==1:
                # Reshape beta_perm[idx, :] to a 2D array with a single row or column based on Din[start:end, :].shape
                beta_reshaped = beta_perm[idx, :].reshape(-1, 1) if Din[start:end, :].shape[-1] != 1 else beta_perm[idx, :].reshape(1, -1)
                # Perform the matrix multiplication
                R_pred[start:end, :] = Din[start:end, :] @ beta_reshaped
            else:
                R_pred[start:end,:] = Din[start:end, :] @ beta_perm[idx, :]   
        if regression_statistics == False:
            return R_pred
        else:
            R2_stats, F_stats, t_stats, p_value = ols_regression_stats(Din, Rin, R_pred, Nnull_samples, no_t_stats)
            return R2_stats, F_stats, t_stats, p_value
    else:
        # Column wise calculation
        R_pred = []
         # Permute rows only (sessions)
        beta_perm = beta[np.random.permutation(beta.shape[0]), :, :] if permute_beta and perm != 0 else beta
        for idx, (start, end) in enumerate(idx_data):
            R_pred.append(np.dot(Din[start:end, :], beta_perm[idx]))
        R_pred= np.concatenate(R_pred, axis=0)
        if regression_statistics == False:
            return R_pred
        else:
            R2_stats, F_stats, t_stats, p_value = ols_regression_stats(Din, Rin, R_pred, Nnull_samples, no_t_stats)
            return R2_stats, F_stats, t_stats, p_value


def calculate_nan_correlation_matrix(D_data, R_data, pval_parametric=False):
    """
    Calculate the correlation matrix between independent variables (D_data) and dependent variables (R_data),
    while handling NaN values column by column of dimension p without  without removing entire rows.
    
    Parameters:
    --------------
    D_data (numpy.ndarray): 
        Input D-matrix for the independent variables.
    R_data (numpy.ndarray): 
        Input R-matrix for the dependent variables.
    pval_parametric (bool). Default is False: 
        Flag to mark if parametric p-values should calculated alongside the correlation cofficients or not.

    Returns:
    ----------  
    correlation_matrix (numpy.ndarray): 
        correlation matrix between columns in D_data and R_data.
    """
    # Initialize a matrix to store correlation coefficients
    p, q = D_data.shape[1], (R_data.shape[1] if R_data.ndim > 1 else 1)
    correlation_matrix = np.full((p, q), np.nan)
    pval_matrix = np.full((p, q), np.nan)
    t_statistics = np.full((p, q), np.nan)
    for p_i in range(p):
        D_column = D_data[:, p_i]
        for q_j in range(q):
            # Do it column by column if R_data got more than 1 column
            R_column = R_data[:, q_j] if R_data.ndim>1 else R_data
            # Find non-NaN indices for both D_column and R_column
            valid_indices = ~np.isnan(D_column) & ~np.isnan(R_column)           
            # calculate the correlation coefficient
            correlation_matrix[p_i, q_j], t_statistics[p_i, q_j], pval_matrix[p_i, q_j] =calculate_correlation_and_tstats(D_column[valid_indices],R_column[valid_indices], pval_parametric)
    return t_statistics, pval_matrix

def geometric_pvalue(p_values, combine_tests):
    """
    Calculate the geometric combination of p-values.
    
    Parameters:
    --------------
    p_values (numpy.ndarray): 
        A 2D array representing the parametric p-values between variables.
        
    combine_tests (bool or str): 
        Specifies the method for combining the p-values:
        - True: Compute the geometric mean of all p-values (1-by-1).
        - "across_columns": Compute the geometric mean for each column, returning an array of shape (1-by-q).
        - "across_rows": Compute the geometric mean for each row, returning an array of shape (1-by-p).
        
    Returns:
    ----------
    corr_combination (numpy.ndarray): 
        The combined p-values based on the specified combine_tests method.
    """
    # Calculate the geometric mean
    if combine_tests== True:
        corr_combination=np.nanmean(p_values)
    elif combine_tests== "across_columns":
        corr_combination=np.nanmean(p_values, axis=0)
    elif combine_tests== "across_rows":
        corr_combination = np.nanmean(p_values, axis=1)
    return corr_combination


def calculate_nan_t_test(Din, Rin, nan_values=False):
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
        p, q = Din.shape[1], Rin.shape[1]
        t_stats = np.zeros((p, q))
        p_values = np.zeros((p, q))

        for q_i in range(q):
            R_column = Rin[:, q_i]
            group0, group1 = np.unique(R_column)

            for p_i in range(p):
                D_column = Din[:, p_i]

                # Mask for valid (non-NaN) data
                valid_mask = ~np.isnan(D_column) & ~np.isnan(R_column)

                group0_vals = D_column[(R_column == group0) & valid_mask]
                group1_vals = D_column[(R_column == group1) & valid_mask]

                t_stat, pval = ttest_ind(group0_vals, group1_vals, nan_policy='omit')
                # Store the t-statistic in the matrix
                t_stats[p_i, q_i] = t_stat
                p_values[p_i, q_i] = pval
    else:
        t_stats = np.zeros((Din.shape[1], Rin.shape[1]))
        p_values = np.zeros((Din.shape[1], Rin.shape[1]))
        for p_i in range(Rin.shape[1]):
            R_column = Rin[:, p_i]
            # Get the t-statistic if there are no NaN values
            t_test_group = np.unique(R_column)
            # Get the t-statistic
            t_test, pval_array = ttest_ind(Din[R_column == t_test_group[0]], Din[R_column == t_test_group[1]]) 
            t_stats[:, p_i] = t_test.ravel()
            p_values[:, p_i] = pval_array.ravel()
    return t_stats, p_values


def calculate_regression_f_stat_univariate(Din, Rin, idx_data, beta, perm, reg_pinv, permute_beta=False, test_indices=None):
    """
    Calculate F-statistics for each feature of Din against categories in R_data, while handling NaN values column by column without removing entire rows.

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
    test_indices_list (numpy.ndarray), default=None:
        Indices for data points that belongs to the test-set for each session.
    Returns:
    ----------  
    f_statistic (numpy.ndarray): 
        F-statistics for each feature in Din against the categories in R_data.
    pval_matrix (numpy.ndarray): 
        parametric p-values estimated from the F-statistic
    """
    p_dim = Din.shape[1]
    q_dim =Rin.ndim if Rin.ndim==1 else Rin.shape[-1]
    f_statistics = np.zeros((p_dim, q_dim))
    pval_array = np.zeros((p_dim, q_dim))
    valid_indices_D = np.all(~np.isnan(Din), axis=1).any()

    if permute_beta:
        for q_i in range(q_dim):
            # Identify columns with NaN values
            R_column =Rin[test_indices][:,np.newaxis] if Rin.ndim==1 else Rin[test_indices, q_i][:,np.newaxis]
            valid_indices_R = np.all(~np.isnan(R_column), axis=1)

            for p_j in range(p_dim):

                #D_column =np.expand_dims(Din[test_indices[0]], axis=1) if len(test_indices)==1 else np.expand_dims(Din[test_indices[0], q_i], axis=1)
                #reg_pinv_column =np.expand_dims(reg_pinv[test_indices[0]], axis=1) if Rin.ndim==1 else np.expand_dims(reg_pinv[test_indices[0], q_i], axis=1)
                D_column = Din[test_indices, p_j][:,np.newaxis]
                beta_column = beta[:,p_j,q_i][:,np.newaxis]# beta coefficeints for each session at a specific column
                nan_values = np.any(np.isnan(valid_indices_R)) or np.any(np.isnan(D_column))
                valid_indices = valid_indices_R & ~np.isnan(D_column[:,0]) if valid_indices_D else valid_indices_R
                R_pred =ols_regression_sessions(R_column, D_column, idx_data, beta_column, perm, permute_beta, nan_values,  valid_indices)
                R_pred = R_pred[:,np.newaxis] if R_pred.ndim==1 else R_pred
                # Calculate the residual sum of squares (rss)
                rss = np.sum((R_column[valid_indices]-R_pred)**2, axis=0)
                # Calculate the total sum of squares (tss)
                tss = np.sum((R_column[valid_indices,] - np.mean(R_column[valid_indices,], axis=0))**2, axis=0)
                ess = tss - rss
                # Calculate the degrees of freedom for the model and residuals
                df1 = D_column.shape[1]  # Number of predictors
                df2 = D_column.shape[0] - df1
                # Calculate the mean squared error (MSE) for the model and residuals
                MSE_model = ess / df1
                MSE_resid = rss / df2
                # Calculate the F-statistic
                f_statistic = MSE_model / MSE_resid
                # Calculate the p-value for the F-statistic
                pval = 1 - f.cdf(f_statistic, df1, df2)
                # Store the p-value
                f_statistics[p_j, q_i] =f_statistic
                pval_array[p_j, q_i] = pval 
    else:
        for q_i in range(q_dim):
            # Identify columns with NaN values
            R_column = Rin[:,np.newaxis] if Rin.ndim==1 else Rin
            valid_indices_R = np.all(~np.isnan(R_column), axis=1)

            for p_j in range(p_dim):
                D_column =  Din[:, p_j][:,np.newaxis]
                reg_pinv_column = reg_pinv[p_j,:][:,np.newaxis] #np.expand_dims(reg_pinv, axis=1) if reg_pinv.ndim==1 else reg_pinv
                valid_indices = valid_indices_R & ~np.isnan(D_column[:,0]) if valid_indices_D else valid_indices_R
                # Calculate beta coefficients using regularized pseudo-inverse of D_data
                beta = reg_pinv_column[valid_indices].T @ R_column[valid_indices] if reg_pinv_column.shape[-1]==1 else  reg_pinv_column[valid_indices] @ R_column[valid_indices]  # Calculate regression_coefficients (beta)
                # Predicted R
                R_pred = D_column @ beta
                # Calculate the residual sum of squares (rss)
                rss = np.sum((R_column[valid_indices,q_i][:,np.newaxis]-R_pred)**2, axis=0)
                # Calculate the total sum of squares (tss)
                tss = np.sum((R_column[valid_indices,q_i] - np.nanmean(R_column[valid_indices,q_i], axis=0))**2, axis=0)
                ess = tss - rss
                # Calculate the degrees of freedom for the model and residuals
                df1 = D_column.shape[1]  # Number of predictors
                df2 = D_column.shape[0] - df1
                # Calculate the mean squared error (MSE) for the model and residuals
                MSE_model = ess / df1
                MSE_resid = rss / df2
                # Calculate the F-statistic
                f_statistic = MSE_model / MSE_resid
                # Calculate the p-value for the F-statistic
                pval = 1 - f.cdf(f_statistic, df1, df2)
                # Store the p-value
                pval_array[p_j, q_i] = pval       
                f_statistics[p_j, q_i] =f_statistic  
        
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
    
def squeeze_first_dim(array):
    """
    Conditionally squeeze a 3D numpy array if its first dimension has size 1.

    Parameters:
    ------------
    array (numpy.ndarray or None): 
        A numpy array that may be 3-dimensional. Can also be None.

    Returns:
    ----------
    numpy.ndarray or None: 
        - If the input array is 3D and its first dimension has size 1, 
          the array is squeezed along the first dimension and the result is returned.
        - If the input array is not 3D, or its first dimension is not of size 1, 
          the array is returned as is.
        - If the input is None, None is returned.
    """
    if array is not None and array.ndim == 3:
        return np.squeeze(array)
    return array


def update_indices(nan_mask, idx_data):
    """
    Updates the index array to account for removed NaN values.

    Parameters:
    nan_mask (np.ndarray): 
        A boolean array where True indicates NaN positions.
    idx_data (np.ndarray): 
        A 2D array of shape (n, 2) where each row contains [start, end] indices.

    Returns:
    idx_data_update (np.ndarray): 
        A 2D array of updated indices after removing NaN values.
    """
    # Get valid indices after removing NaNs
    valid_indices = np.where(~nan_mask)[0]

    # Initialize updated idx_data
    idx_data_update = np.zeros_like(idx_data)

    # Update idx_data based on valid indices
    for i, (start, end) in enumerate(idx_data):
        valid_start = np.searchsorted(valid_indices, start)
        valid_end = np.searchsorted(valid_indices, end - 1, side='right')
        idx_data_update[i, 0] = valid_start
        idx_data_update[i, 1] = valid_end

    return idx_data_update


def create_test_summary(Rin, base_statistics,pval, predictor_names, outcome_names, method, f_t_stats ,n_T, n_N, n_p, n_q, Nnull_samples, category_columns=None, combine_tests=False, test_indices_list=None):
    """
    Create a summary report for the test.

    Parameters:
    --------------
    Rin (numpy.ndarray): 
        Input data matrix for the dependent variables (shape: n_samples x n_outcomes).
    base_statistics (numpy.ndarray): 
        Array of R² or correlation coefficients, depending on the method used.
    pval (numpy.ndarray): 
        Array of p-values corresponding to the base statistics.
    predictor_names (list of str): 
        List of names for the predictors.
    outcome_names (list of str): 
        List of names for the outcomes.
    method (str): 
        Specifies the method used for testing. Options are:
        - "multivariate": For regression analysis with multiple predictors and outcomes.
        - Other: For other tests
    F_stats_list (numpy.ndarray): 
        Array of F-statistics across permutations (shape: n_T x Nnull_samples x n_q for time-dependent data, or Nnull_samples x n_q for time-independent data).
    t_stats_list (numpy.ndarray): 
        Array of t-statistics across permutations (shape: n_T x Nnull_samples x n_p x n_q for time-dependent data, or Nnull_samples x n_p x n_q for time-independent data).
    n_T (int): 
        Number of timepoints (set to 1 for time-independent data).
    n_N (int): 
        Number of observations.
    n_p (int): 
        Number of predictors.
    n_q (int): 
        Number of outcomes.
    Returns:
    ----------
    test_summary (dict): 
        A dictionary containing the summary report of the test
    """

    if method=="multivariate" and Nnull_samples >1 and combine_tests==False:
        # Default when we are doing f-regression
        if category_columns["f_anova_cols"] == []:
            df1 = n_p
            if np.any(np.isnan(Rin)):
                if Rin.shape[0]==n_T:
                    df2 = []  # To store df2 for each regression
                    for q_i in range(Rin.shape[-1]):
                        R_column = Rin[0,:, q_i][:,np.newaxis] if test_indices_list is None else Rin[0,np.concatenate(test_indices_list,axis=0), q_i][:,np.newaxis] 
                        n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                        df2_column = n_valid - n_p  # Compute df2 for the current regression
                        df2.append(df2_column)  # Store df2
                    df2 =np.array(df2)
                else:
                    df2 = []  # To store df2 for each regression
                    for q_i in range(Rin.shape[-1]):
                        R_column = Rin[:, q_i][:,np.newaxis] if test_indices_list is None else Rin[np.concatenate(test_indices_list,axis=0), q_i][:,np.newaxis]
                        n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                        df2_column = n_valid - n_p  # Compute df2 for the current regression
                        df2.append(df2_column)  # Store df2
                    df2 =np.array(df2)
            else:
                df2 = n_N - n_p
        else:
            # Now we need to do a columnwise opperation
            if category_columns["f_anova_cols"] == "all_columns":
                groups = len(np.unique(Rin[~np.isnan(Rin)])) 
                df1 = groups - 1
                if np.any(np.isnan(Rin)):
                    if Rin.shape[0]==n_T:
                        df2 = []  # To store df2 for each regression
                        for q_i in range(Rin.shape[-1]):
                            R_column = Rin[0,:, q_i][:,np.newaxis] if test_indices_list is None else Rin[0,np.concatenate(test_indices_list,axis=0), q_i][:,np.newaxis]
                            n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                            df2_column = n_valid - groups  # Compute df2 for the current regression
                            df2.append(df2_column)  # Store df2
                        df2 =np.array(df2)
                    else:
                        df2 = []  # To store df2 for each regression
                        for q_i in range(Rin.shape[-1]):
                            R_column = Rin[:, q_i][:,np.newaxis] if test_indices_list is None else Rin[np.concatenate(test_indices_list,axis=0), q_i][:,np.newaxis]
                            n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                            df2_column = n_valid - groups  # Compute df2 for the current regression
                            df2.append(df2_column)  # Store df2
                        df2 =np.array(df2)
                else:
                    df2 = n_N - groups
            else:
                # Get the unit for each column
                idx_names = {idx: key for key, indices in category_columns.items() for idx in indices}
                df1 = []  # To store df2 for each column
                df2 = []
                for col in range(len(idx_names)):
                    if idx_names[col]=='f_reg_cols':
                        df1.append(n_p)
                        if np.any(np.isnan(Rin[:, col])):
                            if Rin[:, col].shape[0]==n_T:
                                R_column = Rin[0,:, col][:,np.newaxis] if test_indices_list is None else Rin[0,np.concatenate(test_indices_list,axis=0), col][:,np.newaxis] 
                                n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                                df2_column = n_valid - n_p  # Compute df2 for the current regression
                                df2.append(df2_column)  # Store df2
                            else:
                                R_column = Rin[:, col][:,np.newaxis] if test_indices_list is None else Rin[np.concatenate(test_indices_list,axis=0), col][:,np.newaxis]
                                n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                                df2_column = n_valid - n_p  # Compute df2 for the current regression
                                df2.append(df2_column)  # Store df2
                        else:
                            df2.append(n_N - n_p)
                    elif idx_names[col]=='f_anova_cols':
                        groups = len(np.unique(Rin[~np.isnan(Rin[:, col]), col])) 
                        df1.append(groups - 1)
                        if np.any(np.isnan(Rin)):
                            if Rin.shape[0]==n_T:
                                R_column = Rin[0,:, col][:,np.newaxis] if test_indices_list is None else Rin[0,np.concatenate(test_indices_list,axis=0), col][:,np.newaxis]
                                n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                                df2_column = n_valid - groups  # Compute df2 for the current regression
                                df2.append(df2_column)  # Store df2
                            else:
                                R_column = Rin[:, col][:,np.newaxis] if test_indices_list is None else Rin[np.concatenate(test_indices_list,axis=0), col][:,np.newaxis]
                                n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                                df2_column = n_valid - groups  # Compute df2 for the current regression
                                df2.append(df2_column)  # Store df2
                        else:
                            df2.append(n_N - groups)
                df2 =np.array(df2)

        # Create Model Summary DataFrame
        test_summary = {
            "Predictor": np.repeat(predictor_names, n_q),
            "Outcome": outcome_names,
            "p-value (F-stat)": pval,
            "df1": df1,
            "df2": df2,
            "F-stat": f_t_stats['F_stats'],
            "T-stat": f_t_stats['t_stats'],
            "p-value (F-stat)": f_t_stats['perm_p_values_F'],
            "p-value (t-stat)": f_t_stats['perm_p_values_t'],
            "LLCI": f_t_stats['perm_ci_lower'],
            "ULCI": f_t_stats['perm_ci_upper'],
            "Timepoints": n_T
        }

    elif method=="multivariate" and Nnull_samples <=1 and combine_tests==False:
        # Default when we are doing f-regression
        if category_columns["f_anova_cols"] == []:
            df1 = n_p
            if np.any(np.isnan(Rin)):
                df2 = []  # To store df2 for each regression
                df2_t = []
                df2_t_pval = []
                if Rin.shape[0]==n_T:
                    for q_i in range(Rin.shape[-1]):
                        R_column = Rin[0,:, q_i][:,np.newaxis] if test_indices_list is None else Rin[0,np.concatenate(test_indices_list,axis=0), q_i][:,np.newaxis]
                        n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                        df2_column = n_valid - n_p  # Compute df2 for the current regression
                        df2.append(df2_column)  # Store df2
                        df2_column_t = n_valid - 2  # Compute df2 for the current regression
                        df2_t.append(df2_column_t)
                        df2_t_pval.append(2 * (1 - t.cdf(np.abs(f_t_stats['t_stats']), df2_column_t)))   
                    df2 =np.array(df2)
                    df2_t =np.array(df2_t)
                    df2_t_pval =np.array(df2_t_pval)
                else:
                    for q_i in range(Rin.shape[-1]):
                        R_column = Rin[:, q_i][:,np.newaxis] if test_indices_list is None else Rin[np.concatenate(test_indices_list,axis=0), q_i][:,np.newaxis]
                        n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                        df2_column = n_valid - n_p  # Compute df2 for the current regression
                        df2.append(df2_column)  # Store df2
                        df2_column_t = n_valid - 2  # Compute df2 for the current regression
                        df2_t.append(df2_column_t)
                        df2_t_pval.append(2 * (1 - t.cdf(np.abs(f_t_stats['t_stats']), df2_column_t)))   
                    df2 =np.array(df2)
                    df2_t =np.array(df2_t)
                    parametric_p_values_t =np.array(df2_t_pval)                                    
            else:
                df2 = n_N - n_p
                df2_t = n_N -2
                parametric_p_values_t = 2 * (1 - t.cdf(np.abs(f_t_stats['t_stats']), df2_t))
        else:
            # Now we need to do a columnwise opperation
            if category_columns["f_anova_cols"] == "all_columns":
                groups = len(np.unique(Rin[~np.isnan(Rin)])) 
                df1 = groups - 1
                df2 = []  # To store df2 for each regression
                df2_t = []
                df2_t_pval = []
                if np.any(np.isnan(Rin)):
                    if Rin.shape[0]==n_T:
                        for q_i in range(Rin.shape[-1]):
                            R_column = Rin[0,:, q_i][:,np.newaxis] if test_indices_list is None else Rin[0,np.concatenate(test_indices_list,axis=0), q_i][:,np.newaxis]
                            n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                            df2_column = n_valid - groups  # Compute df2 for the current regression
                            df2.append(df2_column)  # Store df2
                            df2_column_t = n_valid - 2  # Compute df2 for the current regression
                            df2_t.append(df2_column_t)
                            df2_t_pval.append(2 * (1 - t.cdf(np.abs(f_t_stats['t_stats']), df2_column_t)))   
                        df2 =np.array(df2)
                        df2_t =np.array(df2_t)
                        df2_t_pval =np.array(df2_t_pval)
                    else:
                        for q_i in range(Rin.shape[-1]):
                            R_column = Rin[:, q_i][:,np.newaxis] if test_indices_list is None else Rin[np.concatenate(test_indices_list,axis=0), q_i][:,np.newaxis]
                            n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                            df2_column = n_valid - groups  # Compute df2 for the current regression
                            df2.append(df2_column)  # Store df2
                            df2_column_t = n_valid - 2  # Compute df2 for the current regression
                            df2_t.append(df2_column_t)
                            df2_t_pval.append(2 * (1 - t.cdf(np.abs(f_t_stats['t_stats']), df2_column_t)))   
                        df2 =np.array(df2)
                        df2_t =np.array(df2_t)
                        df2_t_pval =np.array(df2_t_pval)
                    
                else:
                    df2 = n_N - groups
                    df2_t = n_N -2
                    parametric_p_values_t = 2 * (1 - t.cdf(np.abs(f_t_stats['t_stats']), df2_t))
            else:
                # Get the unit for each column
                idx_names = {idx: key for key, indices in category_columns.items() for idx in indices}
                df2 = []  # To store df2 for each regression
                df2_t = []
                df2_t_pval = []

                for col in range(len(idx_names)):
                    if idx_names[col]=='f_reg_cols':
                        df1 = n_p
                        if np.any(np.isnan(Rin[:, col])):
                            if Rin[:, col].shape[0]==n_T:
                                R_column = Rin[0,:, col][:,np.newaxis] if test_indices_list is None else Rin[0,np.concatenate(test_indices_list,axis=0), col][:,np.newaxis] 
                                n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                                df2_column = n_valid - n_p  # Compute df2 for the current regression
                                df2.append(df2_column)  # Store df2
                                df2_column_t = n_valid - 2  # Compute df2 for the current regression
                                df2_t.append(df2_column_t)
                                df2_t_pval.append(2 * (1 - t.cdf(np.abs(f_t_stats['t_stats'][:,:,col]), df2_column_t)))   
                            else:
                                R_column = Rin[:, col][:,np.newaxis] if test_indices_list is None else Rin[np.concatenate(test_indices_list,axis=0), col][:,np.newaxis]
                                n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                                df2_column = n_valid - n_p  # Compute df2 for the current regression
                                df2.append(df2_column)  # Store df2
                                df2_column_t = n_valid - 2  # Compute df2 for the current regression
                                df2_t.append(df2_column_t)
                                df2_t_pval.append(2 * (1 - t.cdf(np.abs(f_t_stats['t_stats'][:,:,col]), df2_column_t)))  
                        else:
                            df2.append(n_N - n_p)
                            df2_column_t = n_N - 2  # Compute df2 for the current regression
                            df2_t.append(df2_column_t)
                            df2_t_pval.append(2 * (1 - t.cdf(np.abs(f_t_stats['t_stats'][:,:,col]), df2_column_t)))   
                            
                    elif idx_names[col]=='f_anova_cols':
                        groups = len(np.unique(Rin[~np.isnan(Rin[:, col]), col])) 
                        df1 = groups - 1
                        if np.any(np.isnan(Rin)):
                            if Rin.shape[0]==n_T:
                                R_column = Rin[0,:, col][:,np.newaxis] if test_indices_list is None else Rin[0,np.concatenate(test_indices_list,axis=0), col][:,np.newaxis] 
                                n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                                df2_column = n_valid - groups  # Compute df2 for the current regression
                                df2.append(df2_column)  # Store df2
                                df2_column_t = n_valid - 2  # Compute df2 for the current regression
                                df2_t.append(df2_column_t)
                                df2_t_pval.append(2 * (1 - t.cdf(np.abs(f_t_stats['t_stats'][:,:,col]), df2_column_t)))  
                            else:
                                R_column = Rin[:, col][:,np.newaxis] if test_indices_list is None else Rin[np.concatenate(test_indices_list,axis=0), col][:,np.newaxis]
                                n_valid  = np.sum(np.all(~np.isnan(R_column), axis=1))
                                df2_column = n_valid - groups  # Compute df2 for the current regression
                                df2.append(df2_column)  # Store df2
                                df2_column_t = n_valid - 2  # Compute df2 for the current regression
                                df2_t.append(df2_column_t)
                                df2_t_pval.append(2 * (1 - t.cdf(np.abs(f_t_stats['t_stats'][:,:,col]), df2_column_t)))  
                        else:
                            df2.append(n_N - groups)
                            df2_column_t = n_N - 2  # Compute df2 for the current regression
                            df2_t.append(df2_column_t)
                            df2_t_pval.append(2 * (1 - t.cdf(np.abs(f_t_stats['t_stats'][:,:,col]), df2_column_t)))  
                parametric_p_values_t = np.array(df2_t_pval)
                df2 =np.array(df2)  
        # Create Model Summary DataFrame
        test_summary = {
            "Predictor": predictor_names,
            "Outcome": outcome_names,
            "p-value (F-stat)": pval,
            "df1": df1,
            "df2": df2,
            "F-stat": f_t_stats['F_stats'],
            "T-stat": f_t_stats['t_stats'],
            "p-value (t-stat)": parametric_p_values_t,
            
        }

    else:
        # Other tests
            test_summary = {
            "Predictor": predictor_names,
            "Outcome": outcome_names,
            "Base statistics": base_statistics,
            "P-value": pval,
            "Timepoints": n_T
        }

    return test_summary

def display_test_summary(result_dict, output="both", timepoint=0, return_tables=False):
    """
    Display or export test summary from result_dict.

    Parameters:
    --------------
    result_dict (dict): 
        A dictionary including:
        - 'base_statistics': Array of base statistics (e.g., correlation coefficients).
        - 'pval': Array of p-values from permutation testing.
        - 'test_summary': A dictionary containing the summary report of the test
        
    output (str, optional): 
        Specifies the output to display. Options are:
        - "both": Display both Model Summary and Coefficients Table (default).
        - "model": Display only the Model Summary.
        - "coef": Display only the Coefficients Table.
        
    timepoint (int, optional): 
        Specifies the timepoint index if T-statistics are time-dependent.
        
    return_tables (bool, optional):
        If True, returns the Model Summary and/or Coefficients Table as pandas DataFrames.
        If False, simply displays the tables (default).

    Returns:
    ----------
    None if `return_tables` is False.
    If `return_tables` is True:
        - Returns a tuple (model_summary, coef_table) if output="both".
        - Returns model_summary if output="model".
        - Returns coef_table if output="coef".
    """
    predictors = result_dict['test_summary']['Predictor']
    outcomes = result_dict['test_summary']['Outcome']
    if result_dict["test_type"]== 'test_across_state_visits':
        result_dict["combine_tests"] = False
        result_dict["Nnull_samples"] = result_dict["Nnull_samples"]
    if result_dict["method"] == "multivariate" and result_dict["combine_tests"] == False and result_dict["Nnull_samples"] > 1:
        t_stats = result_dict["test_summary"]["T-stat"]
        n_predictors = t_stats.shape[-2]
        model_summary = coef_table = None
        
        # Check if T-statistics are 2D or 3D (time-dependent)
        if t_stats.ndim == 2 or t_stats.ndim == 3 and t_stats.shape[0] == 1:
            # Time-independent case
            if output in ["both", "model"]:
                model_summary = pd.DataFrame({
                    "Outcome": result_dict["test_summary"]["Outcome"],
                    "F-stat": result_dict["test_summary"]["F-stat"].round(4),
                    "df1": result_dict["test_summary"]["df1"],
                    "df2": result_dict["test_summary"]["df2"],
                    "p-value (F-stat)": result_dict["test_summary"]["p-value (F-stat)"].round(4),
                })
                if not return_tables:
                    print("\nModel Summary:")
                    print(model_summary.to_string(index=False))
            
            if output in ["both", "coef"]:
                coef_table = pd.DataFrame({
                    "Predictor": result_dict["test_summary"]["Predictor"],
                    "Outcome": np.tile(result_dict["test_summary"]["Outcome"], n_predictors),
                    "T-stat": t_stats.flatten(),
                    "p-value": result_dict["test_summary"]["p-value (t-stat)"].flatten(),
                    "LLCI": result_dict["test_summary"]["LLCI"].flatten(),
                    "ULCI": result_dict["test_summary"]["ULCI"].flatten()
                })
                if not return_tables:
                    print("\nCoefficients Table:")
                    print(coef_table.to_string(index=False))
        
        else:
            # Time-dependent case
            if output in ["both", "model"]:
                F_stat = result_dict["test_summary"]["F-stat"].round(4)
                pval_stat =result_dict["test_summary"]["p-value (F-stat)"].round(4)
                if timepoint >= F_stat.shape[0]:
                    raise ValueError(f"Selected time point {timepoint} is out of range. "
                        f"Maximum available time point index is {base_stat.shape[0] - 1}.")
                model_summary = pd.DataFrame({
                    "Outcome": result_dict["test_summary"]["Outcome"],
                    "F-stat": F_stat[timepoint] if F_stat.ndim==1 else F_stat[timepoint,:],
                    "df1": result_dict["test_summary"]["df1"],
                    "df2": result_dict["test_summary"]["df2"],
                    "p-value (F-stat)": pval_stat[timepoint] if pval_stat.ndim==1 else pval_stat[timepoint,:],
                })
                if not return_tables:
                    print(f"\nModel Summary (timepoint {timepoint}):")
                    print(model_summary.to_string(index=False))
            
            expanded_predictors = predictors if isinstance(predictors, str) and len(base_statistics.flatten())==1 else np.repeat(predictors, len(outcomes))
            expanded_outcomes = outcomes if isinstance(outcomes, str) and len(base_statistics.flatten())==1 else np.tile(outcomes, n_predictors)

            if output in ["both", "coef"]:
                coef_table = pd.DataFrame({
                    "Predictor": expanded_predictors,
                    "Outcome": expanded_outcomes,
                    "T-stat": result_dict["test_summary"]["T-stat"][timepoint, :].flatten(),
                    "p-value": result_dict["test_summary"]["p-value (t-stat)"][timepoint, :].flatten(),
                    "LLCI": result_dict["test_summary"]["LLCI"][timepoint, :].flatten(),
                    "ULCI": result_dict["test_summary"]["ULCI"][timepoint, :].flatten()
                })
                if not return_tables:
                    print(f"\nCoefficients Table (timepoint {timepoint}):")
                    print(coef_table.to_string(index=False))

        # Return tables if requested
        if return_tables:
            if output == "both":
                return model_summary, coef_table
            elif output == "model":
                return model_summary
            elif output == "coef":
                return coef_table           
    elif result_dict["method"] == "multivariate" and result_dict["combine_tests"] == False and result_dict["Nnull_samples"] < 1:
        t_stats = result_dict["test_summary"]["T-stat"]
        n_predictors = t_stats.shape[-2]
        model_summary = coef_table = None

        # Ensure df1 and df2 have the correct shape
        df1_fixed = np.full(len(result_dict["test_summary"]["Outcome"]), result_dict["test_summary"]["df1"])
        df2_fixed = np.full(len(result_dict["test_summary"]["Outcome"]), result_dict["test_summary"]["df2"])
        
        # Check if T-statistics are 2D or 3D (time-dependent)
        if t_stats.ndim == 2 or t_stats.ndim == 3 and t_stats.shape[0] == 1:
            # Time-independent case
            if output in ["both", "model"]:
                model_summary = pd.DataFrame({
                    "Outcome": result_dict["test_summary"]["Outcome"],
                    "F-stat": result_dict["test_summary"]["F-stat"].round(4).flatten(),
                    "df1": df1_fixed,
                    "df2": df2_fixed,
                    "p-value (F-stat)": result_dict["test_summary"]["p-value (F-stat)"].round(4).flatten(),
                })
                if not return_tables:
                    print("\nModel Summary - Parametric-test:")
                    print(model_summary.to_string(index=False))

            # Expand "Predictor" to match the number of entries in t_stats
            expanded_predictors = predictors if isinstance(predictors, str) and len(base_statistics.flatten())==1 else np.repeat(predictors, len(outcomes))
            expanded_outcomes = outcomes if isinstance(outcomes, str) and len(base_statistics.flatten())==1 else np.tile(outcomes, n_predictors)

            if output in ["both", "coef"]:
                coef_table = pd.DataFrame({
                    "Predictor": expanded_predictors,
                    "Outcome": expanded_outcomes,
                    "T-stat": t_stats.flatten(),
                    "p-value": result_dict["test_summary"]["p-value (t-stat)"].flatten(),
                })
                if not return_tables:
                    print("\nCoefficients Table:")
                    print(coef_table.to_string(index=False))
        else:
            # Time-dependent case
            if output in ["both", "model"]:
                F_stat = result_dict["test_summary"]["F-stat"].round(4)
                pval_stat =result_dict["test_summary"]["p-value (F-stat)"].round(4)
                if timepoint >= F_stat.shape[0]:
                    raise ValueError(f"Selected time point {timepoint} is out of range. "
                        f"Maximum available time point index is {base_stat.shape[0] - 1}.")
                # Ensure df1 and df2 have the correct shape
                df1_fixed = np.full(len(result_dict["test_summary"]["Outcome"]), result_dict["test_summary"]["df1"])
                df2_fixed = np.full(len(result_dict["test_summary"]["Outcome"]), result_dict["test_summary"]["df2"])

                model_summary = pd.DataFrame({
                    "Outcome": result_dict["test_summary"]["Outcome"],
                    "F-stat": F_stat[timepoint] if F_stat.ndim==1 else F_stat[timepoint,:],
                    "df1": df1_fixed,
                    "df2": df2_fixed,
                    "p-value (F-stat)": pval_stat[timepoint] if pval_stat.ndim==1 else pval_stat[timepoint,:],
                })
                if not return_tables:
                    print(f"\nModel Summary - Parametric-test (timepoint {timepoint}):")
                    print(model_summary.to_string(index=False))
            
            if output in ["both", "coef"]:
                t_stat = result_dict["test_summary"]["T-stat"][timepoint, :].flatten()
                expanded_predictors = predictors if isinstance(predictors, list) and len(t_stat)==len(predictors) or isinstance(predictors, str) and len(t_stat)==1  else np.repeat(predictors, len(outcomes))
                expanded_outcomes = outcomes if isinstance(outcomes, str) and len(t_stat)==1 else np.tile(outcomes, n_predictors)
                t_stat_pval =result_dict["test_summary"]["p-value (t-stat)"]
                coef_table = pd.DataFrame({
                    "Predictor": expanded_predictors,
                    "Outcome": expanded_outcomes,
                    "T-stat": t_stat,
                    "p-value": t_stat_pval[timepoint, 0, :].flatten() if t_stat_pval.ndim==4 else t_stat_pval[timepoint, :].flatten() , # parametric vs using permuation
                })
                if not return_tables:
                    print(f"\nCoefficients Table (timepoint {timepoint}):")
                    print(coef_table.to_string(index=False))

        # Return tables if requested
        if return_tables:
            if output == "both":
                return model_summary, coef_table
            elif output == "model":
                return model_summary
            elif output == "coef":
                return coef_table   
                   
    elif result_dict["method"] == "multivariate" and result_dict["combine_tests"] !=False:
        combine_test_str = f" - {result_dict['combine_tests']}" if result_dict['combine_tests']!=False else ''
        if result_dict["test_summary"]['Timepoints']==1:
            # Time-independent case
            if output in ["both", "model"]:
                if len(np.unique(result_dict["test_summary"]["Outcome"]))==1 and (len(result_dict['pval'][0,:]))==1:
                    outcomes = result_dict["test_summary"]["Outcome"][0] if isinstance(result_dict["test_summary"]["Outcome"],list) else result_dict["test_summary"]["Outcome"]
                else:
                    outcomes = result_dict["test_summary"]["Outcome"]
                model_summary = pd.DataFrame({
                    "Outcome": outcomes,
                    "Base statistics": result_dict['base_statistics'][0,0,:].round(4),
                    "p-value": result_dict['pval'][0,:].round(4),
                    "Unit": "Z-score",
                    "Nnull_samples": result_dict["Nnull_samples"]
                })
                if not return_tables:
                    print(f"\nCoefficients Table{combine_test_str}:")
                    print(model_summary.to_string(index=False))

        else:

            # Time-dependent case
            if output in ["both", "model"]:
                model_summary = pd.DataFrame({
                    "Outcome": result_dict["test_summary"]["Outcome"],
                    "Base statistics": result_dict['base_statistics'][timepoint,0,:].round(4),
                    "p-value": result_dict['pval'][timepoint,:].round(4),
                    "Unit": "Z-score",
                    "Nnull_samples": result_dict["Nnull_samples"]
                })
                if not return_tables:
                    print(f"\nCoefficients Table{combine_test_str} (timepoint {timepoint}):")
                    print(model_summary.to_string(index=False))
                 
    elif result_dict["method"] == "osr":
        # Extract necessary data from result_dict
        base_statistics = result_dict['base_statistics'][0,:]
        pval = result_dict['pval']

        # Generate Model Summary
        max_stat = np.max(np.abs(base_statistics), axis=0)
        min_pval = np.min(pval, axis=0)
        n_predictors =len(np.unique(result_dict["test_summary"]["Predictor"]))

        # unit extraction
        if 'all_columns' in result_dict["statistical_measures"].values():
            key = list(result_dict["statistical_measures"].keys())[0]
            # Get the unit for each column
            unit_key = key.split('_cols')[0]
            
        # Check if all outcomes have the same prefix and end with a number
        prefix = outcomes[0].split(' ')[0]
        if all(re.match(rf"^{prefix} \d+$", outcome) for outcome in outcomes):
            # Sorting using the numerical part of each string
            outcomes = sorted(np.unique(outcomes), key=lambda x: int(x.split(' ')[1]))

        if output in ["both", "model"]:
            model_summary = pd.DataFrame({
                "Unit": [f"{unit_key}-diff"],  
                "Nnull_samples": [result_dict["Nnull_samples"]],  
                "Max Statistic": [max_stat], 
                "Min P-value": [min_pval],  

            })
        if not return_tables:
            print(f"\nModel Summary (OSR-{result_dict['test_summary']['state_comparison']}):")
            print(model_summary.to_string(index=False))

        if output in ["both", "coef"]:
            base_statistics = np.atleast_2d(base_statistics)
            if base_statistics.shape[0] == 1:
                base_statistics = base_statistics.T

            # Ensure pval is also at least 2D
            pval = np.atleast_2d(pval)
            if pval.shape[0] == 1:
                pval = pval.T

            # Construct the DataFrame
            coef_table = pd.DataFrame({"State": np.unique(predictors)})

            # Add columns for each dimension of base_statistics
            for i in range(base_statistics.shape[1]):
                coef_table[f"{unit_key} difference {i+1}"] = base_statistics[:, i]

            # Add columns for each dimension of pval
            for i in range(pval.shape[1]):
                coef_table[f"P-value {i+1}"] = pval[:, i]

            if not return_tables:
                print(f"\nCoefficients Table (OSR-{result_dict['test_summary']['state_comparison']}):")
                print(coef_table.to_string(index=False))
    elif result_dict["method"] == "osa":
        if result_dict["test_summary"]['Timepoints'] == 1:
            # Extract necessary data from result_dict
            base_statistics = result_dict['null_stat_distribution'][0,:]
            pval = result_dict['pval']
                
            # unit extraction
            if 'all_columns' in result_dict["statistical_measures"].values():
                key = list(result_dict["statistical_measures"].keys())[0]
                # Get the unit for each column
                unit_key = key.split('_cols')[0]
            if output in ["both", "model"]:     
                model_summary = pd.DataFrame({
                    "Unit": [f"{unit_key}-diff"],  
                    "Nnull_samples": [result_dict["Nnull_samples"]],  
                })
            if not return_tables:
                print(f"\nModel Summary (OSA)):")
                print(model_summary.to_string(index=False))

            if output in ["both", "coef"]:
                base_statistics = np.atleast_2d(base_statistics)
                if base_statistics.shape[0] == 1:
                    base_statistics = base_statistics.T

                # Ensure pval is also at least 2D
                pval = np.atleast_3d(pval)
                if pval.shape[0] == 1:
                    pval = pval.T

                coef_table = pd.DataFrame({
                    "State X": [x for x, y in result_dict['test_summary']['pairwise_comparisons']],
                    "State Y": [y for x, y in result_dict['test_summary']['pairwise_comparisons']]})
                # Add columns for each dimension of base_statistics
                for i in range(base_statistics.shape[1]):
                    coef_table[f"{unit_key} difference {i+1}"] = base_statistics[:, i]

                # Add columns for each dimension of pval
                for i in range(pval.shape[-1]):
                    coef_table[f"P-value {i+1}"] = [pval[x-1, y-1,i] for x, y in result_dict['test_summary']['pairwise_comparisons']]


                if not return_tables:
                    print(f"\nCoefficients Table (OSA):")
                    print(coef_table.to_string(index=False))

        else:
            # Time-dependent case
            # Extract necessary data from result_dict
            base_statistics = result_dict['base_statistics'][timepoint,:]
            pval = result_dict['pval'][timepoint,:]
                
            # Generate Model Summary
            max_stat = np.max(np.abs(base_statistics), axis=0)
            min_pval = np.min(pval, axis=0)
            n_predictors =len(np.unique(result_dict["test_summary"]["Predictor"]))

            # unit extraction
            if 'all_columns' in result_dict["statistical_measures"].values():
                key = list(result_dict["statistical_measures"].keys())[0]
                # Get the unit for each column
                unit_key = key.split('_cols')[0]

            if output in ["both", "model"]:

                if 'z_score' in result_dict["statistical_measures"]:
                    if output=="model":
                        print("Model summary export is not possible when combining tests with Nnull_samples set to 0.")  

                else:
                    model_summary = pd.DataFrame({
                        "Outcome": np.unique(outcomes),  # Make sure this is a list/array
                        "Max Statistic": [max_stat] if np.isscalar(max_stat) else max_stat,
                        "Min P-value": [min_pval] if np.isscalar(min_pval) else min_pval,
                        "Unit": [unit_key] if isinstance(unit_key, str) else unit_key,
                        "Nnull_samples": [result_dict["Nnull_samples"]] if np.isscalar(result_dict["Nnull_samples"]) else result_dict["Nnull_samples"]
                    })
                    
                    if not return_tables:
                        print("\nModel Summary:")
                        print(model_summary.to_string(index=False))


            if output in ["both", "coef"]:
                coef_table = pd.DataFrame({
                    "Predictor": result_dict["test_summary"]["Predictor"],
                    "Outcome": result_dict["test_summary"]["Outcome"],
                    "Base Statistic": base_statistics.flatten().round(5),
                    "P-value": pval.flatten().round(5)
                })
                if not return_tables:
                    print("\nCoefficients Table:")
                    print(coef_table.to_string(index=False))

                  # Return tables if requested
        if return_tables:
            if output == "both":
                return model_summary, coef_table
            elif output == "model":
                return model_summary
            elif output == "coef":
                return coef_table   
    else:
        if timepoint > result_dict['test_summary']['Timepoints']-1:
            raise ValueError(f"Selected time point {timepoint} is out of range. "
                    f"Maximum available time point index is {result_dict['test_summary']['Timepoints']-1}.")
        combine_test_str = f" - {result_dict['combine_tests']}" if result_dict['combine_tests']!=False else ''
        # Extract necessary data from result_dict
        # base_statistics = result_dict['base_statistics'][timepoint,:] if result_dict['base_statistics'].ndim==3 else result_dict['base_statistics']
        # pval = result_dict['pval'][timepoint,:] if result_dict['pval'].ndim==3 else result_dict['pval']
        base_statistics = result_dict['base_statistics'][timepoint,:]
        pval = result_dict['pval'][timepoint,:] if  base_statistics.ndim==3 and result_dict['pval'].ndim==3 else result_dict['pval']
        # Generate Model Summary
        max_stat = np.max(np.abs(base_statistics), axis=0)
        min_pval = np.min(pval, axis=0)
        n_predictors =len(np.unique(result_dict["test_summary"]["Predictor"]))

        # Check if all outcomes have the same prefix and end with a number
        prefix = outcomes[0].split(' ')[0]
        if all(re.match(rf"^{prefix} \d+$", outcome) for outcome in outcomes):
            # Sorting using the numerical part of each string
            outcomes = sorted(np.unique(outcomes), key=lambda x: int(x.split(' ')[1]))
        
        # unit extraction
        if 'all_columns' in result_dict["statistical_measures"].values():
            key = list(result_dict["statistical_measures"].keys())[0]
            # Get the unit for each column
            unit_key = key.split('_cols')[0]
        else:
            # Get the unit for each column
            idx_names = {idx: key for key, indices in result_dict["statistical_measures"].items() for idx in indices}
            unit_key = "cca" if idx_names=={} or result_dict["method"]=="cca" else [idx_names[i].split('_cols')[0] for i in range(len(outcomes))]

        if output in ["both", "model"]:
            if 'z_score' in result_dict["statistical_measures"]:
                # Apply the function to each value
                model_summary = pd.DataFrame({
                    "Unit": 'z_score',
                    "Nnull_samples": ensure_list(result_dict["Nnull_samples"])
                })
                if not return_tables:
                    if result_dict["test_summary"]['Timepoints'] == 1:
                        print(f"\nModel Summary {combine_test_str}:")
                        print(model_summary.to_string(index=False))
                    else: 
                        print(f"\nModel Summary {combine_test_str} (timepoint {timepoint}):")
                        print(model_summary.to_string(index=False))
            elif result_dict['method']=='cca':
                # Apply the function to each value
                model_summary = pd.DataFrame({
                    "Unit": 'cca',
                    "Nnull_samples": ensure_list(result_dict["Nnull_samples"])
                })
                if not return_tables:

                    if result_dict["test_summary"]['Timepoints'] == 1:
                        print(f"\nModel Summary - cca:")
                        print(model_summary.to_string(index=False))
                    else: 
                        print(f"\nModel Summary - caa (timepoint {timepoint}):")
                        print(model_summary.to_string(index=False))
                        
            else:
                expanded_unit_key =unit_key if len(ensure_list(unit_key))==len(ensure_list(max_stat)) else np.repeat(ensure_list(unit_key),len(min_pval))
                expanded_Nnull_samples = result_dict["Nnull_samples"] if isinstance(result_dict["Nnull_samples"], int) and len(ensure_list(max_stat))==1 else np.repeat(result_dict["Nnull_samples"],len(min_pval))

                # Apply the function to each value
                model_summary = pd.DataFrame({
                    # "Outcome": ensure_list(outcomes),
                    # "Max Statistic": ensure_list(np.squeeze(max_stat)),
                    # "Min P-value": ensure_list(min_pval),
                    "Unit": ensure_list(expanded_unit_key),
                    "Nnull_samples": ensure_list(expanded_Nnull_samples)
                })
                if not return_tables:
                    if result_dict["test_summary"]['Timepoints'] == 1:
                        print(f"\nModel Summary:")
                        print(model_summary.to_string(index=False))
                    else:
                        print(f"\nModel Summary (timepoint {timepoint}):")
                        print(model_summary.to_string(index=False))

        if output in ["both", "coef"]:
            if result_dict['method']=='cca':
                coef_table = pd.DataFrame({
                    "Predictor": 'States',
                    "Outcome": 'Regressors',
                    "Base Statistic": base_statistics.flatten(),
                    "P-value": pval.flatten()
                })
            else:
                expanded_predictors = predictors if isinstance(predictors, list) and len(base_statistics.flatten())==len(predictors) or isinstance(predictors, str) and len(base_statistics.flatten())==1  else np.repeat(predictors, len(outcomes))
                expanded_outcomes = outcomes if isinstance(outcomes, str) and len(base_statistics.flatten())==1 else np.tile(outcomes, n_predictors)
                
                coef_table = pd.DataFrame({
                    "Predictor": expanded_predictors,
                    "Outcome": expanded_outcomes,
                    "Base Statistic": base_statistics.flatten(),
                    "P-value": pval.flatten()
                })
            if not return_tables:
                if result_dict["test_summary"]['Timepoints'] == 1:
                    print("\nCoefficients Table:")
                    print(coef_table.to_string(index=False))
                else:
                    print(f"\nCoefficients Table (timepoint {timepoint}):")
                    print(coef_table.to_string(index=False))
                

                  # Return tables if requested
        if return_tables:
            if output == "both":
                return model_summary, coef_table
            elif output == "model":
                return model_summary
            elif output == "coef":
                return coef_table          

def ensure_list(value):
    """
    Ensure the input is returned as a list.

    This function converts scalar values and strings into a list while leaving 
    iterable inputs unchanged.

    Parameters:
    --------------
    value (any):  
        The input value, which can be a scalar, string, or an iterable.

    Returns:
    --------------  
    list:  
        - If `value` is a scalar (numeric or boolean) or a string, it is wrapped in a list.  
        - If `value` is already an iterable (e.g., list, tuple, numpy array), it is returned unchanged. 
    """
    return [value] if np.isscalar(value) or isinstance(value, str) else value
    
def update_permutation_matrix(permutation_matrix, nan_mask):
    """
    Updates a permutation matrix by removing NaN indices and adjusting remaining indices.

    Parameters:
    --------------
    permutation_matrix (numpy.ndarray): 
        A 2D array where each column represents a permutation.
    nan_mask (numpy.ndarray): 
        A boolean array indicating positions of NaN values in the original dataset.
    
    Returns:
    ----------
    updated_permutation_matrix (numpy.ndarray): 
        A 2D array with NaN indices removed and the remaining indices adjusted accordingly.
    """
    # Get the indices corresponding to NaN values
    indices = np.arange(len(permutation_matrix))
    nan_indices = indices[nan_mask]
    
    # Create a list of valid indices (excluding NaN indices)
    valid_indices = np.setdiff1d(indices, nan_indices)
    
    # Create an array to map old indices to new indices
    mapping_array = np.full(len(permutation_matrix), -1)  # Initialize with -1 for NaNs
    mapping_array[valid_indices] = np.arange(len(valid_indices))  # Map valid indices to new range
    
    # Apply the mapping to update the permutation matrix
    permutation_matrix_update = mapping_array[permutation_matrix]
    
    # Remove the -1 values column-wise
    permutation_matrix_update = np.array([
        col[col != -1] for col in permutation_matrix_update.T  # Transpose, filter, and transpose back
    ]).T
    
    return permutation_matrix_update

def assign_family_groups_from_matrix(data_twins):
    """
    Generate a 1D array of family group labels from a symmetric matrix indicating related subjects.

    Parameters:
    --------------
    data_twins (numpy.ndarray): 
        A square matrix (n_subjects x n_subjects) where a positive value indicates a relationship 
        between two subjects. The diagonal is ignored, and only the upper triangle is considered.

    Returns:
    ----------
    fam_array (numpy.ndarray): 
        A 1D array (n_subjects,) where related subjects are assigned the same group label. 
        Unrelated subjects are marked with 0.
    """
    n_subjects = data_twins.shape[0]
    upper_tri = np.triu(data_twins, k=1)

    fam_array = np.zeros(n_subjects, dtype=int)
    count = 1  # Start family labels from 1

    for subject in range(n_subjects):
        indices = np.where(upper_tri[subject] == 1)[0]
        
        if indices.size > 0:
            existing_labels = fam_array[indices]
            existing_labels = existing_labels[existing_labels > 0]

            if existing_labels.size > 0:
                label = existing_labels[0]
            else:
                label = count
                count += 1

            fam_array[subject] = label
            fam_array[indices] = label

    return fam_array

def __palm_quickperms(EB, M=None, nP=1000, CMC=False, EE=True):
    # Call palm_quickperms from palm_functions
    return palm_quickperms(EB, M, nP, CMC, EE)
