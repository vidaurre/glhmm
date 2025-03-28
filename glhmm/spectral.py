"""
Spectral analysis from Gaussian Linear Hidden Markov Model
@author: Laura Masaracchia, 2025
"""

import numpy as np
from scipy.signal import windows
from glhmm.auxiliary import padGamma, get_T
from sklearn.decomposition import non_negative_factorization


def getfgrid(Fs, nfft, fpass):
    """
    Generate frequency grid for FFT computation.
    """
    df = Fs / nfft
    f = np.linspace(0, Fs - df, nfft)
    findx = (f >= fpass[0]) & (f <= fpass[1])
    return f[findx], findx


def multitaper_spectra(data, tapers, window_length, nfft, findx):
    """
    Perform multitaper spectral analysis on input data.

    Parameters:
    ----------
    data : array-like of shape (n_samples, n_channels)
        (subject- or session-level) timeseries data
    tapers : array-like of shape (n_tapers, window_length)
        the multitapers
    window_length : int, default=
        the length of each multitaper window
    nfft : int, default=
        number of FFT points to use
    findx : array-like or list of length n_freq
        indices of frequency points to keep (frequency range for power spectrum estimation)

    Returns:
    -------
    PW : array-like of shape (n_tapers, n_channels, n_freq)
        power spectrum for each taper and each channel within the indicated frequency range
    CPW: array-like of shape (n_tapers, n_channels, n_channels, n_freq)
        cross-channels power spectrum, for each taper and each channel within the indicated frequency range

    Raises:
    ------
    Error : if data and tapers dimensions are not compatible

    """

    if len(data.shape) == 1:
        data = data[:, np.newaxis]

    n_samples, n_channels = data.shape
    n_freq = np.sum(findx * 1)

    # pad data to have enough timepoints for the windows
    # pad only along time axis, keep channels unchanged
    data = np.pad(data, window_length // 2)[:, window_length // 2: n_channels + window_length // 2]

    # get number of windows
    n_windows = n_samples // window_length

    # initialize p
    PW = np.empty(shape=(n_windows, n_channels, n_freq))
    # inizialize cross p
    CPW = np.empty([n_windows, n_channels, n_channels, n_freq], dtype=np.complex64)

    for w in range(n_windows):
        # select data within window,
        # transpose to make data compatible with tapers multiplicaton
        x_window = data[w * window_length:(w + 1) * window_length, :].transpose(1, 0)

        # check dimensions for debugging purposes
        NC, WLD = x_window.shape  # channels, window_length
        NT, WLT = tapers.shape  # n_tapers, window_length
        if WLD != WLT:
            raise AssertionError('data and tapers dimensions are not compatible')

        # multiply data with tapers
        # add dimension for tapers to the data,
        # add dimension for channels to the tapers
        tapered_x = x_window[np.newaxis, :, :] * tapers[:, np.newaxis, :]
        # shape n_tapers,n_channels,window_length

        # compute fourier transform (along time axis)
        taper_fft = np.fft.fft(tapered_x, nfft)
        taper_range = taper_fft[:, :, findx]
        # should be of shape [n_tapers, n_channels,n_freq]

        # compute power
        power_window = np.real(np.conj(taper_range) * taper_range)
        # compute cross-power
        cross_power_window = np.conj(taper_range)[:, :, np.newaxis, :] * taper_range[:, np.newaxis, :, :]

        # average spectra across tapers
        PW[w, :, :] = np.mean(power_window, axis=0)
        CPW[w, :, :, :] = np.mean(cross_power_window, axis=0)

    return PW, CPW


def multitaper_coherence(cpsd):
    """
    Compute the coherence across frequency between channels

    Parameters:
    ----------
    cpsd : array-like with shape (n_channels, n_channels, n_freq)
        input is the cross power spectral density

    Returns:
    -------
    coherence : array-like, with shape (n_channels, n_channels, n_freq)
        coherence values across channels at each frequency bin
    """
    Nx, Ny, Nf = cpsd.shape
    # Ensure the input signals have the same length
    if Nx != Ny:
        raise ValueError("Wrong shape of cpsd. Row and columns must have the same shape (n_channels)")

    coh = np.empty(shape=(Nx, Ny, Nf))

    for x in range(Nx):
        Sxx = cpsd[x, x, :].real
        for y in range(Ny):
            Syy = cpsd[y, y, :].real

            Sxy = cpsd[x, y, :]

            coh[x, y, :] = np.abs(Sxy) ** 2 / (Sxx * Syy)

    return coh


def multitaper_spectral_analysis(data, indices, Fs, Gamma=None, options=None):
    """
    Compute spectral measures using the multitaper non parametric method, with options specified in "options".
    If the state time courses of a fitted hidden Markov models (Gamma) are given as input,
    the function computes the spectral measures for each state.
    These spectral measures are power, coherence, cross-power spectral density

    Parameters:
    ----------
    data : array-like of shape (total_samples, n_channels)
        the data, all subjects or sessions concatenated
    indices : array-like of shape (n_subjects or n_sessions, 2)
        the indices to start and end of each subject/session
    Fs : int, sampling frequency of the data
    Gamma : array-like of shape (total_samples, n_states)
        State time courses. If specified, the spectral measures are computed per state
        If HMM-MAR or HMM-TDE are used, and Gamma is used unpadded,
        the order or embedded lags have to be specified in the options (see below)
    options : dict
        Possible keys of the options dictionary
        - 'standardize' : bool, whether to standardize the data
        - 'fpass' : tuple or list of length 2, the frequency range for the power spectrum estimation.
            If not specified, the whole range from 0 to Fs/2 is used
        - 'win' : int, window length of each multitaper.
            If not specified, default as Fs * 2 is used
        - 'tapers_res' : int, half time bandwidth, resolution of the tapers
            If not specified, default as 3 is uded
        - 'n_tapers' : int, number of tapers to use.
            If not specified, default as 5 is used
        If Gamma is given as an input, you might need to specify
        - 'order' or 'embeddedlags' : compulsory if the HMM was trained using those (respectively, MAR or TDE)

    Returns:
    -------
    fit : dict with spectral estimates per state
        The dict would contain a dict for each state, with measures:
        - 'f' : frequency bins, with shape (n_freq)
        - 'p' : power spectrum for each channel, with shape (n_subjects or n_sessions, n_freq, n_channels, n_states)
        - 'psdc' : cross-channel power spectral density, with shape (n_subjects or n_sessions, n_freq, n_channels, n_channels, n_states)
        - 'coh' : channels coherence, with shape (n_subjects or n_sessions, n_freq, n_channels, n_channels, n_states)

    Author: Laura Masaracchia
    Email: laurama@cfin.au.dk
    Date: 3/12/2024

    """

    # --------- basic checks ------------

    # check data dimensions
    if len(data.shape) > 2:
        raise AssertionError(
            'Data dimensions incompatible with analysis. Data should be of shape (n_samples, n_channels)')
    if len(data.shape) == 1:  # one channel
        data = data[:, np.newaxis]
        indices = indices[np.newaxis,:]

    n_samples, n_channels = data.shape
    n_subj = indices.shape[0]

    # check indices
    if indices[-1,1] != n_samples:
        raise AssertionError('Indices must specify the start and end of each session or subject data')

    # check Gamma
    if Gamma is None:
        Gamma = np.ones([n_samples, 1], dtype=np.float32)

    K = Gamma.shape[1]

    # if length of Gamma is different than n_samples, pad gamma
    if Gamma.shape[0] != n_samples:
        if ('order' in options.keys()) or ('embeddedlags' in options.keys()):
            T = get_T(indices)
            Gamma = padGamma(Gamma, T, options)
        else:
            raise AssertionError('Wrong shape of Gamma, order or embeddedlags need to be specified in the options')

    # -------------- retrieve parameters from options -------------
    if isinstance(options,dict):
        # Frequency range to be considered
        if 'fpass' in options.keys():
            fpass = options['fpass']
            if len(fpass) != 2:
                print('WARNING: fpass field in option mispecified. Continuing with default range, [0, Fs/2]')
                fpass = [0, Fs / 2]
        else:  # default, from 0 to nyquist frequency
            fpass = [0, Fs / 2]

        if 'win_len' in options.keys():
            window_length = int(options['win_len'])
        else:  # default: Fs * 2
            window_length = int(Fs * 2)

        if 'n_tapers' in options.keys():
            n_tapers = options['n_tapers']
        else:  # set default value to 5 (OSL dynamics default is 7). this is typically NW*2-1
            n_tapers = 5

        if 'tapers_res' in options.keys():
            time_half_bandwidth = options['tapers_res']
        else:  # set default to 3 (OSL dynamics has it to 4)
            time_half_bandwidth = 3

        # standardise data for the computation of the power spectra
        if 'standardize' in options.keys():
            if options['standardize']:
                data = data.copy()
                for j in range(n_subj):
                    t = np.arange(indices[j, 0], indices[j, 1])
                    data[t, :] -= np.mean(data[t, :], axis=0)
                    data[t, :] /= np.std(data[t, :], axis=0)
    else:
        # there is no options specified: use all default
        fpass = [0, Fs / 2]
        window_length = int(Fs * 2)
        n_tapers = 5
        time_half_bandwidth = 3

    # ---------------- compute frequency grid and tapers ----------------

    # number of fft should be power of 2 to speed up computations
    nfft = 2 ** int(np.ceil(np.log2(window_length)))

    # Frequency grid
    f, findx = getfgrid(Fs, nfft, fpass)
    n_freq = len(f)

    # compute tapers
    dpss_tapers = windows.dpss(window_length, time_half_bandwidth, n_tapers)
    # dpss_tapers has shape [n_tapers, window_length]
    # check that it has the right shape
    # for debugging purposes
    # print(dpss_tapers.shape)

    # get scaling coefficient to account for states activation period
    scaling_coefs = n_samples / (np.sum(Gamma ** 2, axis=0))

    # Initialize the measures to return

    p = np.empty(shape=(n_subj,n_freq, n_channels, K))
    psdc = np.empty(shape=(n_subj,n_freq, n_channels, n_channels, K))
    coh = np.empty(shape=(n_subj,n_freq, n_channels, n_channels, K))

    for j, (start, end) in enumerate(indices):
        # get data and Gamma per subject
        data_s = data[start:end, :]
        Gamma_s = Gamma[start:end, :]

        for k in range(K):
            # multiply data by the states probability
            X = data_s * Gamma_s[:, k][:, np.newaxis]

            # compute state-spectra
            p_sk, psdc_sk = multitaper_spectra(X, dpss_tapers, window_length, nfft, findx)

            # Scaling for the multitapers
            p_sk *= 2 / Fs
            psdc_sk *= 2 / Fs

            # average over time windows
            p_sk = np.mean(p_sk, axis=0)
            psdc_sk = np.mean(psdc_sk, axis=0)

            # scaling for the Gamma
            p_sk *= scaling_coefs[k]
            psdc_sk *= scaling_coefs[k]

            # calculate coherence
            coh_sk = multitaper_coherence(psdc_sk)

            psdc[j,:,:,:,k] = psdc_sk.transpose(2,0,1)
            coh[j,:,:,:,k] = coh_sk.transpose(2,0,1)
            p[j,:,:,k] = p_sk.transpose(1,0)

    # if only one channel and one subject, data can be squeezed
    if n_channels == 1 or K==1 or n_subj==1:
        # squeeze eventual extra dimensions
        p = np.squeeze(p)
        psdc = np.squeeze(psdc)
        coh = np.squeeze(coh)

    fit = {'f': f,  # frequency
            'p': p,  # power
            'psdc': psdc,  # cross-power spectral density
            'coh': coh} # coherence

    return fit


def nnmf_decompose(coh, indices, n_components, max_iter=1000):
    """
    Perform a non negative matrix factorization spectral decomposition of the data

    Parameters:
    ----------
    coh : array-like of shape (n_subjects, n_freq, n_channels, n_channels, n_states)
    or (n_subjects, n_freq, n_channels, n_channels), or (n_freq, n_channels, n_channels, n_states)
        coherence just as in the fit dictionary 'coh' outputted by the multitaper spectral estimation
    indices : array-like of shape (n_subjects or n_sessions, 2)
        the indices of timepoints indicating start and end of each subject/session data
    n_components : int
        number of components into which the signal will be decomposed
    max_iter : int, optional. default=1000
        maximum number of iteration to allow the method to do.

    Returns:
    -------
    dec_spectrum: array-like with shape (n_subjects, n_components, n_freq)
        decomposed spectrum
    """
    # coh should have dimensions [n_subjects, n_freq, n_channels, n_channels, n_states]
    # for debugging purposes
    # print(coh.shape)
    if len(coh.shape) == 5:
        n_subj, n_freq, n_channels, n_channels, n_states = coh.shape
    elif len(coh.shape) == 3:
        # assume it's only one subject and no states
        n_freq, n_channels, n_channels = coh.shape
        n_subj=1
        n_states=1

    elif len(coh.shape) == 4:
        # check which dimensions are present
        if len(indices)==coh.shape[0]:
            # assume it's subjects, no states
            n_subj, n_freq, n_channels, n_channels = coh.shape
            n_states=1
        elif coh.shape[1]==coh.shape[2]:
            # assume it's states, no subject
            n_freq, n_channels, n_channels, n_states = coh.shape
            n_subj=1
        else:
            raise AssertionError('Wrong shape of coh. Make sure indices match data')
    else:
        raise AssertionError('Wrong shape of coherence. Make sure it has at least (n_freq,n_channels,n_channels)')

    dec_spectrum = np.empty(shape=(n_subj,n_components,n_freq))
    coh = coh.copy().reshape(n_subj, n_freq, n_channels, n_channels, n_states)

    for n in range(n_subj):
        # put the data in the right shape for NNMF, shape [n_samples, n_features]
        # here is [n_states*n_channels*(n_channels-1)/2, n_freq]
        coh_s = coh[n].transpose(1,2,3,0)
        i, j = np.triu_indices(n_channels, 1)
        coh_sreal = coh_s[i, j, :,:]
        coh_sreshaped = coh_sreal.reshape(-1, n_freq)

        # out of non_negative_factorization is
        # W, n_samples by components,
        # H, n_components by n_features
        # n_iter, number of actual iterations performed until convergence
        W, H, n_iter = non_negative_factorization(coh_sreshaped, n_components=n_components, max_iter=max_iter)

        # Order the weights and components in ascending frequency - just as done in OSL
        order = np.argsort(H.argmax(axis=1))
        dec_spectrum[n] = H[order]

    return dec_spectrum