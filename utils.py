import numpy as np
from constants import PERSONAL_INFO_KEYS, STUDY_PREREQUISITE_KEYS, YES_NO_MAPPING, GENDER_AND_DOMINANT_HAND_MAPPING
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage.filters import gaussian_filter


def convert_readme_info_to_numerical(info, info_key):
    # If info is numerical, then return the int, otherwise return 0 or 1 for male/female and right/left
    if info_key in PERSONAL_INFO_KEYS[:3]:
        return int(info)
    elif info_key in STUDY_PREREQUISITE_KEYS:
        return YES_NO_MAPPING[info]
    else:
        return GENDER_AND_DOMINANT_HAND_MAPPING[info]


def downsample(signal, downsample_factor=4):
    downsampled_signal = signal[::downsample_factor, :]
    return downsampled_signal


def butter_lowpass_filter(data, cutoff=2, fs=700//4, order=2):
    normal_cutoff = cutoff / (0.5 * fs)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def fft_denoiser(x, n_components=0.0001, to_real=True):
    n = len(x)

    # compute the fft
    fft = np.fft.fft(x, n)

    # compute power spectrum density
    # squared magnitud of each fft coefficient
    PSD = fft * np.conj(fft) / n

    # keep high frequencies
    _mask = PSD > n_components
    fft = _mask * fft

    # inverse fourier transform
    clean_data = np.fft.ifft(fft)

    if to_real:
        clean_data = clean_data.real

    return clean_data


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def apply_gaussian_filter(x, sigma=2.5):
    return gaussian_filter(x, sigma)

