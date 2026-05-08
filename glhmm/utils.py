#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some public useful functions - Gaussian Linear Hidden Markov Model
@author: Diego Vidaurre 2023
"""

import numpy as np
import statistics
import math
import pickle
import re
import requests
from pathlib import Path as _Path
from glhmm import glhmm as _glhmm_mod


from scipy.optimize import linear_sum_assignment

def get_FO(Gamma,indices,summation=False):
    """Calculates the fractional occupancy of each state.
    
    Parameters:
    -----------
    Gamma : array-like, shape (n_samples, n_states)
        The state probability time series.
    indices : array-like, shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.
    summation : bool, optional, default=False
        If True, the sum of each row is not normalized, otherwise it is.

    Returns:
    --------
    FO : array-like, shape (n_sessions, n_states)
        The fractional occupancy of each state per session.

    """

    N = indices.shape[0]
    K = Gamma.shape[1]
    FO = np.zeros((N,K))
    for j in range(N):
        ind = np.arange(indices[j,0],indices[j,1])
        FO[j,:] = np.sum(Gamma[ind,:],axis=0)
        if not summation:
            FO[j,:] /= np.sum(FO[j,:])
    return FO
    

def get_maxFO(Gamma,indices):
    """Calculates the maximum fractional occupancy per session.

    The first argument can also be a viterbi path (vpath).

    Parameters:
    -----------
    Gamma : array-like of shape (n_samples, n_states); or a vpath, array of shape (n_samples,)
        The Gamma represents the state probability timeseries and the vpath represents the most likely state sequence.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.

    Returns:
    --------
    maxFO: array-like of shape (n_sessions,)
        The maximum fractional occupancy across states for each trial/session

    Notes:
    ------
    The maxFO is useful to assess the amount of `state mixing`. For more information, see [^1].

    References:
    -----------
    [^1]: Ahrends, R., et al. (2022). Data and model considerations for estimating time-varying functional connectivity in fMRI. NeuroImage 252, 119026.
           https://pubmed.ncbi.nlm.nih.gov/35217207/)

    """
    FO = get_FO(Gamma,indices)
    return np.max(FO,axis=1)


def get_state_evoked_response(Gamma,indices):
    """Calculates the state evoked response 

    The first argument can also be a viterbi path (vpath).

    Parameters:
    ---------------
    Gamma : array-like of shape (n_samples, n_states), or a vpath array of shape (n_samples,)
        The Gamma represents the state probability timeseries and the vpath represents the most likely state sequence.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.

    Returns:
    ------------
    ser : array-like of shape (n_samples, n_states)
        The state evoked response matrix.

    Raises:
    -------
    Exception
        If the input data violates any of the following conditions:
        - There is only one trial/session
        - Not all trials/sessions have the same length.
    """

    N = indices.shape[0]
    if N == 1: 
        raise Exception("There is only one segment / trial")
    T = indices[:,1] - indices[:,0]
    if not(np.all(T[0]==T)):
        raise Exception("All segments / trials must have the same length")
    K = Gamma.shape[1]
    T = T[0]

    ser = np.mean(np.reshape(Gamma,(T,N,K),order='F'),axis=1)
    return ser


def get_switching_rate(Gamma,indices):
    """Calculates the switching rate.

    The first argument can also be a viterbi path (vpath).

    Parameters:
    ---------------
    Gamma : array-like of shape (n_samples, n_states), or a vpath array of shape (n_samples,)
        The Gamma represents the state probability timeseries and the vpath represents the most likely state sequence.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.

    Returns:
    ------------
    SR : array-like of shape (n_sessions, n_states)
        The switching rate matrix.

    """
    N = indices.shape[0]
    K = Gamma.shape[1]
    SR = np.zeros((N,K))
    for j in range(N):
        ind = np.arange(indices[j,0],indices[j,1])
        SR[j,:] = np.mean(np.abs(Gamma[ind[1:],:]-Gamma[ind[0:-1],:]),axis=0)
    return SR


def get_visits(vpath,k,threshold=0):
    """Computes a list of visits for state k, given a viterbi path (vpath).

    Parameters:
    ---------------
    vpath : array-like of shape (n_samples,)
        The viterbi path represents the most likely state sequence.
    k : int
        The state for which to compute the visits.
    threshold : int, optional, default=0
        A threshold value used to exclude visits with a duration below this value.

    Returns:
    ------------
    lengths : list of floats
        A list of visit durations for state k, where each duration is greater than the threshold.
    onsets : list of ints
        A list of onset time points for each visit.

    Notes:
    ------
    A visit to state k is defined as a contiguous sequence of time points in which state k is active.

    """

    lengths = []
    onsets = []
    T = vpath.shape[0]
    vpath_k = vpath[:,k]
    t = 0 
    while t < T: 
        t += np.where(vpath_k[t:]==1)[0]
        if len(t)==0: 
            break
        t = t[0]
        onsets.append(t)
        tend = np.where(vpath_k[t:]==0)[0]
        if len(tend)==0: 
            length_visit = len(vpath_k)-t
            if length_visit > threshold: lengths.append(float(length_visit))
            break
        tend = tend[0]
        length_visit = tend
        if length_visit > threshold: lengths.append(float(length_visit))
        t += tend
    return lengths,onsets


def get_life_times(vpath,indices,threshold=0):
    """Calculates the average, median and maximum life times for each state.

    Parameters:
    -----------
    vpath : array-like of shape (n_samples,)
        The viterbi path represents the most likely state sequence.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.
    threshold : int, optional, default=0
        A threshold value used to exclude visits with a duration below this value.

    Returns:
    --------
    meanLF : array-like of shape (n_sessions, n_states)
        The average visit duration for each state in each trial/session.
    medianLF : array-like of shape (n_sessions, n_states)
        The median visit duration for each state in each trial/session.
    maxLF : array-like of shape (n_sessions, n_states)
        The maximum visit duration for each state in each trial/session.

    Notes:
    ------
    A visit to a state is defined as a contiguous sequence of time points in which the state is active.
    The duration of a visit is the number of time points in the sequence.
    This function uses the `get_visits` function to compute the visits and exclude those below the threshold.

    """
    N = indices.shape[0]
    K = vpath.shape[1]    
    meanLF = np.zeros((N,K)) 
    medianLF = np.zeros((N,K)) 
    maxLF = np.zeros((N,K)) 
    for j in range(N):
        ind = np.arange(indices[j,0],indices[j,1]).astype(int)
        for k in range(K):
            visits,_ = get_visits(vpath[ind,:],k,threshold=threshold)
            if len(visits) > 0:
                meanLF[j,k] = statistics.mean(visits)
                medianLF[j,k] = statistics.median(visits)
                maxLF[j,k] = max(visits)
    return meanLF, medianLF, maxLF


def get_state_onsets(vpath,indices,threshold=0):
    """Calculates the state onsets, i.e., the time points when each state activates.

    Parameters:
    ---------------
    vpath : array-like of shape (n_samples, n_states)
        The viterbi path represents the most likely state sequence.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.
    threshold : int, optional, default=0
        A threshold value used to exclude visits with a duration below this value.

    Returns:
    --------
    onsets : list of lists of ints
        A list of the time points when each state activates for each trial/session.

    Notes:
    ------
    A visit to a state is defined as a contiguous sequence of time points in which the state is active.
    This function uses the `get_visits` function to compute the visits and exclude those below the threshold.

    """

    N = indices.shape[0]
    K = vpath.shape[1]    
    onsets = []
    for j in range(N):
        onsets_j = []
        ind = np.arange(indices[j,0],indices[j,1]).astype(int)
        for k in range(K):
            _,onsets_k = get_visits(vpath[ind,:],k,threshold=threshold)
            onsets_j.append(onsets_k)
        onsets.append(onsets_j)
    return onsets


def get_FO_entropy(Gamma,indices):
    """Calculates the entropy of each session, if we understand fractional occupancies as probabilities.

    Parameters:
    --------------
    Gamma : array-like of shape (n_samples, n_states)
        The Gamma represents the state probability timeseries.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.

    Returns:
    --------
    entropy : array-like of shape (n_sessions,)
        The entropy of each session.

    """  
    fo = get_FO(Gamma,indices)
    N,K = fo.shape
    entropy = np.zeros(N)
    for j in range(N):
        for k in range(K):
            if fo[j,k] == 0: continue
            entropy[j] -= math.log(fo[j,k]) * fo[j,k]
    return entropy
    

def get_state_evoked_response_entropy(Gamma,indices):
    """Calculates the entropy of each time point, if we understand state evoked responses as probabilities.

    Parameters:
    ---------------
    Gamma: array-like of shape (n_samples, n_states)
        The Gamma represents the state probability timeseries.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.

    Returns:
    ------------
    entropy: array-like of shape (n_samples,)
        The entropy of each time point.

    """  
    ser = get_state_evoked_response(Gamma,indices)
    T,K = ser.shape
    entropy = np.zeros(T)
    for t in range(T):
        for k in range(K):
            if ser[t,k] == 0: continue
            entropy[t] -= math.log(ser[t,k]) * ser[t,k]
    return entropy


def get_gamma_similarity(gamma1, gamma2):
    """Computes a measure of similarity between two sets of state time courses.

    These can have different numbers of states, but they must have the same
    number of time points.

    Parameters:
    -----------
    gamma1 : numpy.ndarray
        First set of state time courses with shape (T, K).
    gamma2 : numpy.ndarray
        Second set of state time courses with shape (T, K2), where K2 may be different from K.

    Returns:
    --------
    S : float
        Similarity, measured as the sum of joint probabilities under the optimal state alignment.
    assig : numpy.ndarray
        Optimal state alignment for gamma2 (uses Munkres' algorithm).
    gamma2 : numpy.ndarray
        The second set of state time courses reordered to match gamma1.
    """
    
    T, K = gamma1.shape
    gamma1_0 = gamma1.copy()   
    g = gamma2
    K2 = g.shape[1]
    
    if K < K2:
        gamma1 = np.hstack((gamma1_0, np.zeros((T, K2 - K))))
        K = K2
    elif K > K2:
        g = np.hstack((g, np.zeros((T, K - K2))))
    
    M = np.zeros((K, K))  # cost
    
    for k1 in range(K):
        for k2 in range(K):
            M[k1, k2] += (T - np.sum(np.minimum(gamma1[:, k1], g[:, k2]))) / T
    
    row_ind, col_ind = linear_sum_assignment(M)
    S = K - M[row_ind, col_ind].sum()
    
    gamma2 = g[:, col_ind]
    
    return S, col_ind, gamma2


# ---------------------------------------------------------------------------
# Stability-training helpers
# ---------------------------------------------------------------------------

def load_stability_results(save_dir):
    """
    Load HMM stability training results from disk.

    Handles two cases automatically:
    - `summary_results.pkl` present -> loads directly (fast path).
    - Only individual `hmm_K*_rep*.pkl` files -> rebuilds the summary from them
      by recomputing Gamma similarity between repetition 0 and every subsequent repetition.

    Parameters:
    --------------
    save_dir (str or Path):
        Directory where ``run_stability_training()`` saved its outputs.

    Returns:
    ----------
    results (dict):
        Dictionary with K values as keys, each containing:
        - `'FE'`: list of free energy arrays, one per repetition.
        - `'similarity_scores'`: list of Gamma similarity floats (rep 0 vs rep i).
    state_range (list of int):
        Sorted list of K values found in the directory.
    """
    import pickle
    import re
    from pathlib import Path as _Path

    save_dir = _Path(save_dir)
    summary_path = save_dir / 'summary_results.pkl'

    if summary_path.exists():
        with open(summary_path, 'rb') as f:
            results = pickle.load(f)
        state_range = sorted(results.keys())
        print(f'Loaded summary_results.pkl  (K = {list(state_range)})')
    else:
        print('summary_results.pkl not found. Rebuilding from individual model files...')
        pattern = re.compile(r'hmm_K(\d+)_rep(\d+)\.pkl')
        saved = {}
        for p in sorted(save_dir.glob('hmm_K*_rep*.pkl')):
            m = pattern.match(p.name)
            if m:
                K, rep = int(m.group(1)), int(m.group(2))
                saved.setdefault(K, []).append((rep, p))

        if not saved:
            raise FileNotFoundError(
                f'No hmm_K*_rep*.pkl files found in {save_dir}. '
                'Check that save_dir is correct and training has been run.'
            )

        state_range = sorted(saved.keys())
        results = {K: {'similarity_scores': [], 'FE': []} for K in state_range}

        for K in state_range:
            reps_sorted = sorted(saved[K], key=lambda x: x[0])
            print(f'  K={K}: loading {len(reps_sorted)} repetitions...', end=' ')
            Gamma_ref = None
            for rep, fpath in reps_sorted:
                with open(fpath, 'rb') as f:
                    d = pickle.load(f)
                results[K]['FE'].append(d['FE'])
                if Gamma_ref is None:
                    Gamma_ref = d['Gamma']
                else:
                    sim, _, _ = get_gamma_similarity(Gamma_ref, d['Gamma'])
                    results[K]['similarity_scores'].append(sim)
            del Gamma_ref
            print('done')

        with open(summary_path, 'wb') as f:
            pickle.dump(results, f)
        print(f'Saved reconstructed summary to {summary_path}')

    print(f"\n {'K':>4} | {'N reps':>6} | {'N sim scores':>12} | {'Min final FE':>14}")
    print('-' * 44)
    for K in state_range:
        n_reps = len(results[K]['FE'])
        n_sim  = len(results[K]['similarity_scores'])
        min_fe_val = min(fe[-1] for fe in results[K]['FE'])
        print(f'{K:>4} | {n_reps:>6} | {n_sim:>12} | {min_fe_val:>14.2f}')

    return results, state_range


def run_stability_training(Y, indices, state_range, n_repeats, save_dir,
                           log_preproc=None, covtype='full', model_mean='no',
                           options=None):
    """
    Train HMMs across a range of K values to assess solution stability.

    For each K and random repetition: initialises an HMM; trains with full-batch EM
    until convergence; saves the model to disk; and computes Gamma similarity between
    repetition 0 (reference) and all subsequent repetitions to measure how
    reproducible the state solution is across random initialisations.

    Parameters:
    --------------
    Y (numpy.ndarray):
        Preprocessed data array of shape `(n_total_timepoints, n_features)`,
        with all subjects concatenated along the time axis.
    indices (numpy.ndarray):
        Start and end indices for each subject, shape `(n_subjects, 2)`.
    state_range (iterable of int):
        K values to test, e.g. ``range(5, 13)``.
    n_repeats (int):
        Number of independent random initialisations per K value.
    save_dir (str or Path):
        Directory to write per-model pickle files and the ``summary_results.pkl`` summary.
    log_preproc (preprocessing log or None, optional), default=None:
        Log returned by ``preproc.preprocess_data()``. Passed as ``preproclogY`` to
        the HMM so that state parameters can be back-transformed to the original space.
    covtype (str, optional), default='full':
        Covariance type passed to ``glhmm()``. Options: ``'full'`` (state-specific FC
        matrices), ``'diag'`` (diagonal, faster), ``'sharedfull'`` (one shared FC matrix),
        ``'shareddiag'``.
    model_mean (str, optional), default='no':
        Whether to model per-state activation means. Use ``'no'`` for standardised data;
        ``'state'`` if activation levels carry information.
    options (dict or None, optional), default=None:
        Training options passed to ``hmm.train()``. Defaults to
        ``{'cyc': 500, 'min_cyc': 25, 'tol': 1e-5, 'verbose': False}``.

    Returns:
    ----------
    results (dict):
        Dictionary with K values as keys, each containing:
        - `'FE'`: list of free energy arrays, one per repetition.
        - `'similarity_scores'`: list of Gamma similarity floats (rep 0 vs rep i).
    """


    if options is None:
        options = {'cyc': 500, 'min_cyc': 25, 'tol': 1e-5, 'verbose': False}

    save_dir = _Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results = {K: {'similarity_scores': [], 'FE': []} for K in state_range}

    for K in state_range:
        print(f'Training HMM with {K} states ({n_repeats} repetitions)...')
        Gamma_ref = None

        for repeat in range(n_repeats):
            np.random.seed(repeat)

            hmm = _glhmm_mod.glhmm(
                K=K,
                covtype=covtype,
                model_mean=model_mean,
                model_beta='no',
                preproclogY=log_preproc,
            )

            Gamma1, _, FE1 = hmm.train(Y=Y, indices=indices, options=options)

            with open(save_dir / f'hmm_K{K}_rep{repeat + 1}.pkl', 'wb') as f:
                pickle.dump({'hmm': hmm, 'Gamma': Gamma1, 'FE': FE1}, f)

            results[K]['FE'].append(FE1)

            if repeat == 0:
                Gamma_ref = Gamma1
            else:
                sim, _, _ = get_gamma_similarity(Gamma_ref, Gamma1)
                results[K]['similarity_scores'].append(sim)
                del Gamma1

            del hmm
            print(f'  K={K}  rep={repeat + 1}/{n_repeats}  FE={FE1[-1]:.2f}', flush=True)

        del Gamma_ref

    with open(save_dir / 'summary_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print('Training complete.')
    return results


def run_stability_training_stochastic(files, state_range, n_repeats, save_dir,
                                       log_preproc=None, covtype='full', model_mean='no',
                                       options=None):
    """
    Train HMMs stochastically across a range of K values to assess solution stability.

    For each K and random repetition: initialises an HMM; trains with stochastic
    mini-batch EM; calls ``hmm.decode()`` to obtain the Gamma time series (stochastic
    training returns empty Gamma by design); saves the model to disk; and computes
    Gamma similarity between repetition 0 (reference) and all subsequent repetitions.

    Use this function when your dataset is too large to hold in RAM. Data must be
    split into one ``.npy`` or ``.npz`` file per subject on disk (see
    ``io.save_subjects_file()``). For in-memory data, use ``run_stability_training()``.

    Parameters:
    --------------
    files (list of str or Path):
        Paths to per-subject preprocessed data files (one file per subject).
    state_range (iterable of int):
        K values to test, e.g. ``range(5, 13)``.
    n_repeats (int):
        Number of independent random initialisations per K value.
    save_dir (str or Path):
        Directory to write per-model pickle files and the ``summary_results.pkl`` summary.
    log_preproc (preprocessing log or None, optional), default=None:
        Log returned by ``preproc.preprocess_data()``. Passed as ``preproclogY`` to
        the HMM so that state parameters can be back-transformed to the original space.
    covtype (str, optional), default='full':
        Covariance type passed to ``glhmm()``. Options: ``'full'`` (state-specific FC
        matrices), ``'diag'`` (diagonal, faster), ``'sharedfull'`` (one shared FC matrix),
        ``'shareddiag'``.
    model_mean (str, optional), default='no':
        Whether to model per-state activation means. Use ``'no'`` for standardised data;
        ``'state'`` if activation levels carry information.
    options (dict or None, optional), default=None:
        Training options passed to ``hmm.train()``. ``stochastic`` is always set to
        ``True``. Defaults to ``{'Nbatch': 20, 'initNbatch': 20, 'initcyc': 50,
        'cyc': 500, 'min_cyc': 100, 'forget_rate': 0.5, 'base_weights': 0.75,
        'cyc_to_go_under_th': 10, 'deactivate_states': False, 'verbose': False}``.

    Returns:
    ----------
    results (dict):
        Dictionary with K values as keys, each containing:
        - `'FE'`: list of free energy arrays, one per repetition.
        - `'similarity_scores'`: list of Gamma similarity floats (rep 0 vs rep i).
    """

    _default_options = {
        'stochastic': True,
        'Nbatch': 20,
        'initNbatch': 20,
        'initcyc': 50,
        'cyc': 500,
        'min_cyc': 100,
        'forget_rate': 0.5,
        'base_weights': 0.75,
        'cyc_to_go_under_th': 10,
        'deactivate_states': False,
        'verbose': False,
    }
    if options is None:
        options = _default_options
    else:
        options = dict(options)
        options.setdefault('stochastic', True)

    save_dir = _Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results = {K: {'similarity_scores': [], 'FE': []} for K in state_range}

    for K in state_range:
        print(f'Training HMM with {K} states ({n_repeats} repetitions)...')
        Gamma_ref = None

        for repeat in range(n_repeats):
            np.random.seed(repeat)

            hmm = _glhmm_mod.glhmm(
                K=K,
                covtype=covtype,
                model_mean=model_mean,
                model_beta='no',
                preproclogY=log_preproc,
            )

            _, _, FE1 = hmm.train(files=files, options=options)
            Gamma1, _, _ = hmm.decode(X=None, Y=None, files=files)

            with open(save_dir / f'hmm_K{K}_rep{repeat + 1}.pkl', 'wb') as f:
                pickle.dump({'hmm': hmm, 'Gamma': Gamma1, 'FE': FE1}, f)

            results[K]['FE'].append(FE1)

            if repeat == 0:
                Gamma_ref = Gamma1
            else:
                sim, _, _ = get_gamma_similarity(Gamma_ref, Gamma1)
                results[K]['similarity_scores'].append(sim)
                del Gamma1

            del hmm
            print(f'  K={K}  rep={repeat + 1}/{n_repeats}  FE={FE1[-1]:.2f}', flush=True)

        del Gamma_ref

    with open(save_dir / 'summary_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print('Stochastic training complete.')
    return results


def osf_download_data(osf_url, data_dir='data', folder=None):
    """
    Download files from an OSF project to a local directory.

    Queries the OSF storage API for the given project, optionally navigates
    into a named sub-folder, and downloads every file that does not yet exist
    locally. Files already present are silently skipped, so the function is
    safe to re-run.

    Parameters:
    --------------
    osf_url (str):
        OSF project URL (e.g. ``'https://osf.io/8qcyj/'``) or bare project
        identifier (e.g. ``'8qcyj'``). The project ID is extracted automatically
        so any standard OSF URL format works.
    data_dir (str or Path, optional), default=``'data'``:
        Local directory to download files into. Created automatically if it does
        not exist.
    folder (str or None, optional), default=None:
        Name of a sub-folder inside the project's OSF Storage to download from.
        ``None`` downloads all files from the storage root level.

    Returns:
    ----------
    None

    Examples:
    ----------
    Download all files from the root of a project::

        utils.osf_download_data('https://osf.io/8qcyj/')

    Download files from a specific sub-folder::

        utils.osf_download_data('https://osf.io/8qcyj/', folder='Simulation_data_numpy')
    """
    data_dir = _Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    _match = re.search(r'osf\.io/([A-Za-z0-9]+)', osf_url)
    project_id = _match.group(1) if _match else osf_url

    _root = requests.get(
        f'https://api.osf.io/v2/nodes/{project_id}/files/osfstorage/'
    ).json()['data']

    if folder is None:
        _files = [x for x in _root if x['attributes']['kind'] == 'file']
    else:
        _folder_node = next(
            x for x in _root if x['attributes']['name'] == folder
        )
        _files = [
            x for x in requests.get(
                _folder_node['relationships']['files']['links']['related']['href']
            ).json()['data']
            if x['attributes']['kind'] == 'file'
        ]

    downloaded = 0
    for _f in _files:
        _name = _f['attributes']['name']
        _dest = data_dir / _name
        if not _dest.exists():
            print(f'  Downloading {_name} ...', end=' ', flush=True)
            _dest.write_bytes(requests.get(_f['links']['download']).content)
            print('done')
            downloaded += 1

    if downloaded == 0:
        print(f'Files already present in {data_dir}/.')
    else:
        print(f'Downloaded {downloaded} file(s) to {data_dir}/.')
