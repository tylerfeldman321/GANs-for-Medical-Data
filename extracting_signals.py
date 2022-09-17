import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd  # version >= 0.24.1
import os
import pickle
# scipy version >= 1.2.0
from scipy import stats, integrate
# from scipy.signal import find_peaks
# import matplotlib.pyplot as plt
# from pyEDA.main import *
import biosppy
# import neurokit2 as nk
# import pyhrv
# import cv2


class OneSubjectPrepare:
    def __init__(self, subject, path):
        self.keys = ['label', 'subject', 'signal']
        self.signal_keys = ['wrist', 'chest']
        self.wrist_keys = ['ACC', 'BVP', 'EDA', 'TEMP']
        self.chest_keys = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        os.chdir(path)
        os.chdir(subject)
        with open(subject + '.pkl', 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        self.data = data

    def get_labels(self):
        return self.data[self.keys[0]]

    def get_wrist_data(self):
        # assert subject == self.data[self.keys[1]]
        signal = self.data[self.keys[2]]
        wrist_data = signal[self.signal_keys[0]]
        return wrist_data

    def get_chest_data(self):
        # assert subject == self.data[self.keys[1]]
        signal = self.data[self.keys[2]]
        chest_data = signal[self.signal_keys[1]]
        return chest_data


def trans_lb(labels):
    new_lb = []
    for i in range(0, len(labels) - 1, 2):
        lb = int(stats.mode(labels[i:(i + 2)])[0])
        new_lb.append(lb)
    return np.array(new_lb).reshape(-1, 1)


def heartbeat(ecg, labels, samples_in_window, rate=700/4, start_at=10, end_at=-10):
    # naive segmentation based on R-peaks
    samples = []
    sample_labels = []
    # ecg = resample_data(ecg, 700, rate)

    pad_left = samples_in_window // 2
    pad_right = samples_in_window - pad_left

    ecg = ecg[:, 0]
    peaks = biosppy.signals.ecg.christov_segmenter(ecg, sampling_rate=rate)[0]
    for i in peaks[start_at:end_at]:
        sample = ecg[i - pad_left: i + pad_right]
        if len(sample) != samples_in_window:
            continue
        samples.append(sample)  # fixed-length segments
        sample_labels.append(labels[i])
    return samples, sample_labels
