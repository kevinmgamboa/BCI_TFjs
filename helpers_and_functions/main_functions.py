"""
Meine Projekt Funcktioniert
---------------------------
This file contains most of the functions to be use during this project
"""
import os
import datetime as dt
from tqdm import tqdm

import numpy as np
# Libraries with important functions
import antropy as ant
from scipy import signal as sg
from scipy.signal import hilbert
from scipy.signal import lfilter, butter, filtfilt

# Library for EEG signal management
import mne
# Personal functions
from helpers_and_functions import config

# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
def vec_nor(x, a: int = -1, b: int = 1):
    """
    Normalize the amplitude of a vector from a to b
    [-1, 1] by default
    """
    # normalize [0,1]
    n_vec = np.divide(x - min(x), max(x) - min(x))
    # normalize [-1,1]
    n_vec = (b - a) * n_vec + a

    return n_vec


def nor_dataset(dataset, a: int = -1, b: int = 1):
    """
     Normalize an entire dataset between a to b

     [-1, 1] by default
     """
    for n, epoch in enumerate(tqdm(dataset)):
        # Gets normalization per channel
        k = [vec_nor(channel, a=a, b=b) for channel in epoch]
        # Added to the dataset
        dataset[n] = k

    return np.array(dataset)


def binary_matrix(signal):
    # applying hilbert transform
    ht = hilbert(signal)
    # extracting real part
    e = ht.real
    # converting to binary
    b = 1 * (e > np.mean(e))
    return b


def binary_matrix_dataset(dataset):
    for n, epoch in enumerate(tqdm(dataset)):
        # finds the binary matrix
        k = np.array([binary_matrix(channel) for channel in epoch])
        # Added to the dataset
        dataset[n] = np.array(k)

    return dataset


def zero_mean(x):
    """
    Zero mean
    """
    n_vec = np.divide(x, max(x))
    n_vec = n_vec - np.mean(n_vec)

    return n_vec


def eeg_time(ts: int) -> str:
    """
    ts: time in seconds
    """
    # get time in datetime format
    td = dt.timedelta(seconds=ts)
    # get days
    days = td.days
    # calculates hours
    hours, remainder = divmod(td.seconds, 3600)
    # calculates minutes and seconds
    minutes, seconds = divmod(remainder, 60)

    return f'{days}d-{hours}h:{minutes}m:{seconds}s'


# -----------------------------------------------------------------------------
#                                Envelopes
# -----------------------------------------------------------------------------
def sim_envelope(x, fs, seconds=0.3, smooth=[False, 10]):
    """
    Develope by: Kevin Machado G
    sim_envelope states for "simple envelope"
        Reference
    C. Jarne. Simple empirical algorithm to obtain signal envelope in three steps
    March 21, 2017. University of Quilmes (UNQ) e-mail: cecilia.jarne@unq.edu.ar

    @param x:
    @param fs:
    @param seconds:
    @param smooth:
    @return:
    """
    Y = np.zeros(len(x))
    # 1. Take the absolute value of the signal
    SE_x = abs(x)
    # 2. Divide the signal into k bunches of N samples (corresponding to 0.1 s) (0.1*fs)/1
    sec = seconds
    N = int(sec * fs)
    k, m = divmod(len(x), N)

    for i in range(N):
        n = max(SE_x[i * k:i * k + (k - 1)])
        Y[i * k:i * k + k] = n
    if smooth[0] is True:
        # 4. Smoothing the signal
        Y = butter_lowpass_filter(Y, smooth[1], fs, order=1)
    return Y


# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------
def butter_lowpass(cutoff: int,
                   fs: int,
                   order: int = 3) -> [np.ndarray, np.ndarray]:
    """

    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=3):
    b, a = butter_lowpass(cutoff, fs, order)
    y = filtfilt(b, a, data)
    return y


def butter_highpass(cutoff: int,
                    fs: int,
                    order: int = 3) -> [np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=3):
    b, a = butter_highpass(cutoff, fs, order)
    y = filtfilt(b, a, data)
    return y


def butter_bp_coe(lowcut, highcut, fs, order=1):
    """
    Butterworth passband filter coefficients b and a
    Ref:
    [1] https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
    [2] https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bp_fil(data, lowcut, highcut, fs, order=1):
    """
    Butterworth passband filter
    Ref:
    [1] https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
    [2] https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
    """
    b, a = butter_bp_coe(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return vec_nor(y)


# Notch Filter
def notch_filter(val: float,
                 data: np.ndarray,
                 fs: int = 250) -> np.ndarray:
    """
        Notch Filter
    """
    notch_freq_Hz = np.array([float(val)])
    for freq_Hz in np.nditer(notch_freq_Hz):
        bp_stop_Hz = freq_Hz + 3.0 * np.array([-1, 1])
        b, a = sg.butter(3, bp_stop_Hz / (fs / 2.0), 'bandstop')
        fin = data = sg.lfilter(b, a, data)
    return fin


# -----------------------------------------------------------------------------
#                               Pre-Processing
# -----------------------------------------------------------------------------
def filter_block(in_signal: np.ndarray,
                 sf: int,
                 lowcut: int = 100,
                 highcut: int = 0.5,
                 notch: int = 50,
                 ) -> np.ndarray:
    """
    Applies filtering process to an input signal
    """
    # Applies low-pass filter
    out_signal = butter_lowpass_filter(in_signal, lowcut, sf)
    # Applies high-pass filter
    out_signal = butter_highpass_filter(out_signal, highcut, sf)
    # Applies notch-pass filter
    out_signal = notch_filter(notch, out_signal, sf)

    return out_signal


def band_pass_bl(in_signal: np.ndarray,
                 sf: int,
                 band: tuple,
                 ) -> np.ndarray:
    """
    Applies filtering process to an input signal
    @param in_signal: input signal
    @param sf: input signal sampling frequency
    @param lowcut: low-pass filter frequency
    @param highcut: high-pass filter frequency
    @return: Filtered signal
    """
    low, high = band
    # Applies low-pass filter
    out_signal = butter_lowpass_filter(in_signal, high, sf)
    # Applies high-pass filter
    out_signal = butter_highpass_filter(out_signal, low, sf)

    return out_signal


# -----------------------------------------------------------------------------
#                               Feature Extraction
# -----------------------------------------------------------------------------
def band_binary_matrix(signal: np.array, sf: int = 100) -> np.array:
    """

    @param signal: input signal
    @param sf: sampling frequency of the signal
    @return: array with 5 rows x samples equal to input signal
    """
    # defines eeg bands
    eeg_bands = config.eeg_bands
    # applying filter bank and binary matrix
    band_eeg = [binary_matrix(band_pass_bl(signal, sf, eeg_bands[band])) for band in eeg_bands]

    return np.array(band_eeg)


def band_binary_matrix_dataset(dataset):
    # initializing transformed dataset
    t_dataset = []
    for n, epoch in enumerate(tqdm(dataset)):
        ch1 = band_binary_matrix(epoch[0], sf=100)
        ch2 = band_binary_matrix(epoch[1], sf=100)
        k = np.dstack((ch1, ch2))
        # # finds the binary matrix
        # k = [band_binary_matrix(channel, sf=100) for channel in epoch]
        # # Added to the dataset
        t_dataset.append(k)

    return np.array(t_dataset)


def sleep_montage(raw):
    # Creates Fpz-Cz difference
    new_raw = mne.set_bipolar_reference(raw.load_data(), anode=['Fp2'], cathode=['Cz'])
    # Creates Pz-Oz difference
    new_raw = mne.set_bipolar_reference(new_raw, anode=['Pz'], cathode=['O2'])

    return new_raw


def lz_algorithm(signal):
    # applying hilbert transform
    ht = hilbert(signal)
    # finding envelope
    e = np.abs(ht)
    # converting to binary
    b = 1 * (e > np.mean(e))
    # LZ complexity
    lz = ant.lziv_complexity(b, normalize=True)

    return lz


def get_Spectrogram(audio, Fs, fft_size=200, step_size=1, spec_thresh=2.5, window_type=0):
    """
    Returns a spectrogram with default fft_size, step_size and spec_thresh
    Parameters
    ----------
    audio : audio data.
    Fs : sampling frequency of the audio data.

    Returns
    Spectrogram: Matrix of frequencies in time and their power
    -------
    TYPE
        DESCRIPTION.

    """
    return abs(move_spec(
        aid_spectrogram(audio.astype('float64'), log=True, thresh=spec_thresh, fft_size=fft_size, step_size=step_size,
                        window_type=0))) / 2.5


def aid_spectrogram(in_signal: np.ndarray, log: bool = True, thresh: int = 5,
                    fft_size: int = 512, step_size: int = 64, window_type=0) -> np.ndarray:
    """
    Generates the spectrogram of an 1-D input signal
    @param in_signal: 1D input signal.
    @param log: applies the log of the spectrogram.
    @param thresh: threshold minimum power for log spectrogram.
    @param fft_size: length of FFT used. Normally this is the sampling frequency of the input signal.
    @param step_size:
    @param window_type: desired window to multiply the signal.
    @return: Matrix of frequencies in time and their power.
    """
    # Applies the short time Fourier transform and gets the abs value
    specgram = np.abs(
        stft(in_signal, fft_size=fft_size, step=step_size, real=False, compute_onesided=True, window_type=window_type))
    # Applies the log to the spectrogram
    if log:
        specgram /= specgram.max()  # volume normalize to max 1
        specgram = np.log10(specgram)  # take log
        specgram[specgram < -thresh] = -thresh  # set anything less than the threshold as the threshold
    else:
        specgram[specgram < thresh] = thresh  # set anything less than the threshold as the threshold

    return np.transpose(specgram)


def stft(in_signal: np.ndarray, fft_size: int = 40, step: int = 66, mean_normalize: bool = True,
         real: bool = False, compute_onesided: bool = True, window_type: int = 0) -> np.ndarray:
    """
    Implements the Short-time Fourier transform (STFT) of a 1D real input signal.
    @param in_signal: input signal
    @param fft_size: length of FFT used. Normally this is the sampling frequency of the input signal.
    @param step:
    @param mean_normalize:
    @param real:
    @param compute_onesided:
    @param window_type: desired window to multiply the signal.
    @return:
    """
    if real:
        local_fft = np.fft.rfft
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    if compute_onesided:
        cut = fft_size // 2
    if mean_normalize:
        in_signal -= np.mean(in_signal)
    # Applies signal overlapping
    in_signal = overlapping(in_signal, stepsize=step, number_time_samples=fft_size)

    size = fft_size
    if window_type == 0:
        win = sg.general_gaussian(size, p=0.5, sig=200)
    if window_type == 1:
        win = 0.8 - .9 * np.sin(2 * np.pi * np.arange(size) / (size - 1))  # Modify Hamming Window
    if window_type == 2:
        win = 100 - 1 * np.tanh(2 * np.pi * np.arange(size) / (size - 1))  # Hann window

    in_signal = in_signal * win[None]
    in_signal = local_fft(in_signal)[:, :cut]
    return in_signal


def xrange(x):
    return iter(range(x))


def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))

    ws = window_size
    ss = window_step
    a = X

    valid = len(a) - ws
    nw = int((valid) // ss)
    out = np.ndarray((nw, ws), dtype=a.dtype)
    i = 0

    for i in xrange(nw):
        # "slide" the window along the samples
        start = i * ss
        stop = start + ws
        out[i] = a[start: stop]

    return out


def overlapping(signal, stepsize=1, fs=1000, time_window=0.2, number_time_samples=1):
    """
    Create an overlapped version of the input signal
    Parameters
    ----------
    signal : input vector corresponding to audio data with shape (n_audio_poins,)
    fs : sampling rate of the input signal
    time_window : Duration of the window in seconds
    stepsize : number of samples between the beginning of a window and the beginning of the next one

    Returns
    -------
    chunked_and_overlapped_signal : Matrix with shape (number_examples, number_time_samples)
    """
    if number_time_samples == 1:
        number_time_samples = int(fs * time_window)
    else:
        pass
    append = np.zeros((number_time_samples - len(
        signal) % number_time_samples))  # this calculates how many samples to add to the input vector for the windows to fit along it, and creates a zeros vector of that size.
    signal = np.hstack((signal,
                        append))  # completes the input vector with the zeros vector created in order to have an even number of windows fit in the data
    result = np.vstack(signal[i:i + number_time_samples] for i in range(0, len(signal) - number_time_samples, stepsize))
    return result


def move_spec(spectrogram, shift=100):
    '''This function move the spectogram 'shift' numbers of points to the right.
    This is done to fix the displacement given when applying the spectrogram with
    our personal function'''
    cut = spectrogram[:, 0:shift]
    for i in range(spectrogram.shape[1] - 1, spectrogram.shape[1] - shift, -1):
        spectrogram = np.roll(spectrogram, 1, axis=1)
        spectrogram[:, -1] = 0
    spectrogram[:, 0:shift] = cut
    return spectrogram


# -----------------------------------------------------------------------------
# Project Dataset
# -----------------------------------------------------------------------------
def load_sleep_physionet_raw(raw_fname, annot_fname, load_eeg_only=True,
                             crop_wake_mins=30):
    """Load a recording from the Sleep Physionet dataset.

    Parameters
    ----------
    raw_fname : str
        Path to the .edf file containing the raw data.
    annot_fname : str
        Path to the annotation file.
    load_eeg_only : bool
        If True, only keep EEG channels and discard other modalities
        (speeds up loading).
    crop_wake_mins : float
        Number of minutes of wake events before and after sleep events.

    Returns
    -------
    mne.io.Raw :
        Raw object containing the EEG and annotations.
    """
    mapping = {'EOG horizontal': 'eog',
               'Resp oro-nasal': 'misc',
               'EMG submental': 'misc',
               'Temp rectal': 'misc',
               'Event marker': 'misc'}
    exclude = mapping.keys() if load_eeg_only else ()

    raw = mne.io.read_raw_edf(raw_fname, exclude=exclude)
    annots = mne.read_annotations(annot_fname)
    raw.set_annotations(annots, emit_warning=False)
    if not load_eeg_only:
        raw.set_channel_types(mapping)

    if crop_wake_mins > 0:  # Cut start and end Wake periods
        # Find first and last sleep stages
        mask = [x[-1] in ['1', '2', '3', '4', 'R']
                for x in annots.description]
        sleep_event_inds = np.where(mask)[0]

        # Crop raw
        tmin = annots[int(sleep_event_inds[0])]['onset'] - \
               crop_wake_mins * 60
        tmax = annots[int(sleep_event_inds[-1])]['onset'] + \
               crop_wake_mins * 60
        raw.crop(tmin=tmin, tmax=tmax)

    # Rename EEG channels
    ch_names = {i: i.replace('EEG ', '')
                for i in raw.ch_names if 'EEG' in i}
    mne.rename_channels(raw.info, ch_names)

    # Save subject and recording information in raw.info
    basename = os.path.basename(raw_fname)
    subj_nb, rec_nb = int(basename[3:5]), int(basename[5])
    raw.info['subject_info'] = {'id': subj_nb, 'rec_id': rec_nb}

    return raw

def raw_chunks_to_spectrograms(data, sf=100):
    """Converts raw chunks into spectrogram chunks.

    Parameters
    ----------
    data : Tuple
        Raw data object to be windowed.

    Returns
    -------
    np.ndarray
        .
    np.array
        Labels.
    """
    print("applying spectrogram transformation")
    # Spectrogram parameters
    # sf = 100  # This is the eeg signal sampling frequency
    fft_size = int(sf / 2)  # 1000 samples represent 500ms in time  # window size for the FFT
    step_size = 1  # distance to slide along the window (in time) if devided by 40 is good
    spec_thresh = 5  # threshold for spectrograms (lower filters out more noise)

    # Initialising chunked-spectrograms variable
    X = []
    for sample_idx in tqdm(range(len(data))):
        ch1 = aid_spectrogram(data[sample_idx][0].astype('float64'), log=True, thresh=spec_thresh, fft_size=fft_size,
                              step_size=step_size)
        ch2 = aid_spectrogram(data[sample_idx][1].astype('float64'), log=True, thresh=spec_thresh, fft_size=fft_size,
                              step_size=step_size)
        chs = np.dstack((ch1, ch2)).astype('float32')

        X.append(chs)

    print("Finished spectrogram transformation")

    return X

def load_anaesthesia_data(ane_data_path):
    """Loads the Anaesthesia dataset

        Parameters
        ----------
        ane_data_path : string
            path to the Anaesthesia dataset

        Returns
        -------
        data : dict
            dictionary with eeg recordings
        """
    # Initializing dict to store eeg's
    ane_data = {"eye_open": [],
                "eye_closed": [],
                "sedation_1": []}

    # Using the first 5 minutes of sedated EEG data
    tmin, tmax = 0, 5 * 60
    # loading process
    for subject in tqdm(os.listdir(ane_data_path)):
        for eeg_file in os.listdir(os.path.join(ane_data_path, subject)):
            # Loads eyes closed
            if eeg_file.endswith("ec.vhdr"):
                ane_data["eye_closed"].append(mne.io.read_raw(os.path.join(ane_data_path,
                                                                           os.path.join(subject, eeg_file))))
            # Loads eyes open
            if eeg_file.endswith("eo.vhdr"):
                ane_data["eye_open"].append(mne.io.read_raw(os.path.join(ane_data_path,
                                                                         os.path.join(subject, eeg_file))))
            # Sedated EEG recording 1
            if eeg_file.endswith("SED_1.vhdr") or eeg_file.endswith("SED_1_rest.vhdr") or \
                    eeg_file.endswith("sed_1.vhdr") or eeg_file.endswith("1045_SED_1_rest_2.vhdr"):
                # loads the raw signal and uses the first 5 minutes
                ane_data["sedation_1"].append(mne.io.read_raw(os.path.join(ane_data_path,
                                                                           os.path.join(subject, eeg_file))).crop(tmin,
                                                                                                                  tmax))
    return ane_data

def extract_epochs(raw: None,
                   dataset: str,
                   chunk_duration: int = 3.,
                   binary: bool = False):
    """Extract non-overlapping epochs from raw data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object to be windowed.
    chunk_duration : float
        Length of a window.

    Returns
    -------
    np.ndarray
        Epoched data, of shape (n_epochs, n_channels, n_times).
    np.ndarray
        Event identifiers for each epoch, shape (n_epochs,).
    """
    if dataset == 'sleep':

        if binary is True:
            annotation_desc_2_event_id = {
                'Sleep stage W': 1,
                'Sleep stage 2': 3,
                'Sleep stage 3': 4,
                'Sleep stage 4': 4}

            event_id = {
                'Sleep stage W': 1,
                'Sleep stage 2': 3,
                'Sleep stage 3': 4,
                'Sleep stage 4': 4}
        else:
            annotation_desc_2_event_id = {
                'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3': 4,
                'Sleep stage 4': 4,
                'Sleep stage R': 5}

            event_id = {
                'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3': 4,
                'Sleep stage 4': 4,
                'Sleep stage R': 5}

        events, _ = mne.events_from_annotations(
            raw, event_id=annotation_desc_2_event_id,
            chunk_duration=chunk_duration)

        # create a new event_id that unifies stages 3 and 4

        tmax = chunk_duration - 1. / raw.info['sfreq']  # tmax in included
        picks = mne.pick_types(raw.info, eeg=True, eog=True)  # pick channel numbers
        epochs = mne.Epochs(raw=raw, events=events, picks=picks, preload=False,
                            event_id=event_id, tmin=0., tmax=tmax, baseline=None)

        return epochs.get_data(), epochs.events[:, 2] - 1

    if dataset == 'eeg':
        # Creates fixed length epochs, given that there are none
        epochs = mne.make_fixed_length_epochs(raw, duration=chunk_duration, preload=False)
        return epochs.get_data(), epochs.events[:, 2]

# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
def eeg_total_time(dataset, which='eeg'):
    """
    counts the total amount og EEG data in time
    """

    if which == 'sleep':
        tc = sum([eeg.times[-1] for eeg in dataset])
        print(f'{len(dataset)} files found')
        print('Total EEG time lenght: {} '.format(dt.timedelta(seconds=tc)))

    if which == 'eeg':
        # dataset categories
        dc = list(dataset.keys())
        # initializing time per category
        tpc = np.zeros(len(dc))
        # Initialize time counter
        tc = 0
        # Count the number of eeg files
        eegf = 0

        for n, clase in enumerate(dataset):
            eeg_class = dataset[str(clase)]
            # get the num of files
            eegf += len(eeg_class)
            # sums the time in each file
            tc = sum([eeg.times[-1] for eeg in eeg_class])

            tpc[n] = tc
            # counter reset
            tc = 0
        # print results
        print(f'{len(dataset)} categories found, {eegf} eeg files')
        print(f'Total EEG time lenght: {dt.timedelta(seconds=sum(tpc))}')
        print('{} category: {}\n{} category: {}\n{} category: {}'.format(dc[0],
                                                                         dt.timedelta(seconds=tpc[0]),
                                                                         dc[1],
                                                                         dt.timedelta(seconds=tpc[1]),
                                                                         dc[2],
                                                                         dt.timedelta(seconds=tpc[2])))
