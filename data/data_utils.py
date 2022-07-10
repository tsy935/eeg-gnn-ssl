import numpy as np
import random
import os
import sys

sys.path.append("../")
import pyedflib
from constants import INCLUDED_CHANNELS, FREQUENCY, ALL_LABEL_DICT
from scipy.fftpack import fft
from scipy.signal import resample, correlate


def computeFFT(signals, n):
    """
    Args:
        signals: EEG signals, (number of channels, number of data points)
        n: length of positive frequency terms of fourier transform
    Returns:
        FT: log amplitude of FFT of signals, (number of channels, number of data points)
        P: phase spectrum of FFT of signals, (number of channels, number of data points)
    """
    # fourier transform
    fourier_signal = fft(signals, n=n, axis=-1)  # FFT on the last dimension

    # only take the positive freq part
    idx_pos = int(np.floor(n / 2))
    fourier_signal = fourier_signal[:, :idx_pos]
    amp = np.abs(fourier_signal)
    amp[amp == 0.0] = 1e-8  # avoid log of 0

    FT = np.log(amp)
    P = np.angle(fourier_signal)

    return FT, P


def get_swap_pairs(channels):
    """
    Swap select adjacenet channels
    Args:
        channels: list of channel names
    Returns:
        list of tuples, each a pair of channel indices being swapped
    """
    swap_pairs = []
    if ("EEG FP1" in channels) and ("EEG FP2" in channels):
        swap_pairs.append((channels.index("EEG FP1"), channels.index("EEG FP2")))
    if ("EEG Fp1" in channels) and ("EEG Fp2" in channels):
        swap_pairs.append((channels.index("EEG Fp1"), channels.index("EEG Fp2")))
    if ("EEG F3" in channels) and ("EEG F4" in channels):
        swap_pairs.append((channels.index("EEG F3"), channels.index("EEG F4")))
    if ("EEG F7" in channels) and ("EEG F8" in channels):
        swap_pairs.append((channels.index("EEG F7"), channels.index("EEG F8")))
    if ("EEG C3" in channels) and ("EEG C4" in channels):
        swap_pairs.append((channels.index("EEG C3"), channels.index("EEG C4")))
    if ("EEG T3" in channels) and ("EEG T4" in channels):
        swap_pairs.append((channels.index("EEG T3"), channels.index("EEG T4")))
    if ("EEG T5" in channels) and ("EEG T6" in channels):
        swap_pairs.append((channels.index("EEG T5"), channels.index("EEG T6")))
    if ("EEG O1" in channels) and ("EEG O2" in channels):
        swap_pairs.append((channels.index("EEG O1"), channels.index("EEG O2")))

    return swap_pairs


def getOrderedChannels(file_name, verbose, labels_object, channel_names):
    labels = list(labels_object)
    for i in range(len(labels)):
        labels[i] = labels[i].split("-")[0]

    ordered_channels = []
    for ch in channel_names:
        try:
            ordered_channels.append(labels.index(ch))
        except:
            if verbose:
                print(file_name + " failed to get channel " + ch)
            raise Exception("channel not match")
    return ordered_channels


def getSeizureTimes(file_name):
    """
    Args:
        file_name: edf file name
    Returns:
        seizure_times: list of times of seizure onset in seconds
    """
    tse_file = file_name.split(".edf")[0] + ".tse_bi"

    seizure_times = []
    with open(tse_file) as f:
        for line in f.readlines():
            if "seiz" in line:  # if seizure
                # seizure start and end time
                seizure_times.append(
                    [
                        float(line.strip().split(" ")[0]),
                        float(line.strip().split(" ")[1]),
                    ]
                )
    return seizure_times


def getSeizureClass(file_name, target_labels_dict=None, file_type="edf"):
    """
    Args:
        file_name: file name of .edf file etc.
        target_labels_dict: dict, key is seizure class str, value is seizure class number,
                        e.g. {'fnsz': 0, 'gnsz': 1}
        file_type: "edf" or "tse"
    Returns:
        seizure_class: list of seizure class in the .edf file
    """
    label_dict = (
        target_labels_dict if target_labels_dict is not None else ALL_LABEL_DICT
    )
    target_labels = list(label_dict.keys())

    tse_file = ""
    if file_type == "edf":
        tse_file = file_name[:-4] + ".tse"
    elif file_type == "tse":
        tse_file = file_name
    else:
        raise valueError("Unrecognized file type.")

    seizure_class = []
    with open(tse_file) as f:
        for line in f.readlines():
            if any(
                s in line for s in target_labels
            ):  # if this is one of the seizure types of interest
                seizure_str = [s for s in target_labels if s in line]
                seizure_class.append(label_dict[seizure_str[0]])
    return seizure_class


def getEDFsignals(edf):
    """
    Get EEG signal in edf file
    Args:
        edf: edf object
    Returns:
        signals: shape (num_channels, num_data_points)
    """
    n = edf.signals_in_file
    samples = edf.getNSamples()[0]
    signals = np.zeros((n, samples))
    for i in range(n):
        try:
            signals[i, :] = edf.readSignal(i)
        except:
            pass
    return signals


def resampleData(signals, to_freq=200, window_size=4):
    """
    Resample signals from its original sampling freq to another freq
    Args:
        signals: EEG signal slice, (num_channels, num_data_points)
        to_freq: Re-sampled frequency in Hz
        window_size: time window in seconds
    Returns:
        resampled: (num_channels, resampled_data_points)
    """
    num = int(to_freq * window_size)
    resampled = resample(signals, num=num, axis=1)
    return resampled


######## Graph related data utils ########
def keep_topk(adj_mat, top_k=3, directed=True):
    """ "
    Helper function to sparsen the adjacency matrix by keeping top-k neighbors
    for each node.
    Args:
        adj_mat: adjacency matrix, shape (num_nodes, num_nodes)
        top_k: int
        directed: whether or not a directed graph
    Returns:
        adj_mat: sparse adjacency matrix, directed graph
    """
    # Set values that are not of top-k neighbors to 0:
    adj_mat_noSelfEdge = adj_mat.copy()
    for i in range(adj_mat_noSelfEdge.shape[0]):
        adj_mat_noSelfEdge[i, i] = 0

    top_k_idx = (-adj_mat_noSelfEdge).argsort(axis=-1)[:, :top_k]

    mask = np.eye(adj_mat.shape[0], dtype=bool)
    for i in range(0, top_k_idx.shape[0]):
        for j in range(0, top_k_idx.shape[1]):
            mask[i, top_k_idx[i, j]] = 1
            if not directed:
                mask[top_k_idx[i, j], i] = 1  # symmetric

    adj_mat = mask * adj_mat
    return adj_mat


def comp_xcorr(x, y, mode="valid", normalize=True):
    """
    Compute cross-correlation between 2 1D signals x, y
    Args:
        x: 1D array
        y: 1D array
        mode: 'valid', 'full' or 'same',
            refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
        normalize: If True, will normalize cross-correlation
    Returns:
        xcorr: cross-correlation of x and y
    """
    xcorr = correlate(x, y, mode=mode)
    # the below normalization code refers to matlab xcorr function
    cxx0 = np.sum(np.absolute(x) ** 2)
    cyy0 = np.sum(np.absolute(y) ** 2)
    if normalize and (cxx0 != 0) and (cyy0 != 0):
        scale = (cxx0 * cyy0) ** 0.5
        xcorr /= scale
    return xcorr


######## Graph related data utils ########