from constants import *
import pickle
from extracting_signals import heartbeat
from utils import downsample, convert_readme_info_to_numerical, moving_average
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt


class SubjectData:
    def __init__(self, subject_id, sample_length=200, sample_type='start_at_r_peak'):
        self.metadata = {}
        self.features = {}
        self.signals = {}
        self.id = subject_id
        self.sample_length = sample_length
        self.sample_labels = None
        assert (subject_id in SUBJECT_IDS), 'Subject ID is not in SUBJECT IDS'

        # Collect sensor measurement data from the pickle file
        pkl_filepath = os.path.join(DATA_DIR, f'S{subject_id}', f'S{subject_id}.pkl')
        with open(pkl_filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        self.subject = data['subject']

        self.chest = data['signal']['chest']
        self.labels = data['label']
        self.labels = np.expand_dims(self.labels, axis=1)

        # TODO: IGNORING WRIST DATA FOR NOW
        # self.wrist = data['signal']['wrist'] # IGNORING WRIST DATA FOR NOW SINCE IT HAS DIFFERENT SAMPLING RATE
        # for signal_name in self.wrist:
        #     self.signals['wrist_' + signal_name] = downsample(self.wrist[signal_name]) # IGNORING WRIST DATA FOR NOW SINCE IT HAS DIFFERENT SAMPLING RATE

        # TODO: ONLY USING ECG DATA FOR NOW
        for signal_name in self.chest:
            self.signals['chest_' + signal_name] = downsample(self.chest[signal_name])
        # self.signals['chest_ECG'] = downsample(self.chest['ECG'])
        self.labels = downsample(self.labels)

        # Take the signals and convert them into smaller time series features
        self.num_features = self._extract_time_windows(signal_name='chest_ECG', samples_in_window=self.sample_length, get_labels=True)

        # Collect data from the subject readme
        readme_filepath = os.path.join(DATA_DIR, f'S{subject_id}', f'S{subject_id}_readme.txt')
        self.personal_info = {}
        self.prerequisite_questions = {}
        with open(readme_filepath, 'r') as f:
            lines = f.read()
            for personal_info_key in PERSONAL_INFO_KEYS:
                info = lines.split(personal_info_key)[1][2:].split('\n')[0]
                info = convert_readme_info_to_numerical(info, personal_info_key)
                self.metadata[personal_info_key] = np.ones((self.num_features, 1)) * info
            for study_prerequisite_question in STUDY_PREREQUISITE_KEYS:
                info = lines.split(study_prerequisite_question)[1][1:].split('\n')[0]
                info = convert_readme_info_to_numerical(info, study_prerequisite_question)
                self.metadata[study_prerequisite_question] = np.ones((self.num_features, 1)) * info

    def _extract_time_windows(self, signal_name, normalize=True, samples_in_window=200, sample_type='full_heartbeat',
                              get_labels=True, lpf=True, plot_time_windows=False):
        num_windows_list = []
        signal = self.signals[signal_name]
        samples = []
        sample_metadata = []

        # For only QRS, find each peak, and extract it. Then pad the left and right to fill up the space
        if sample_type == 'only_qrs':
            signal_max = np.max(signal)
            signal_min = np.min(signal)
            normalized_signal = (signal - signal_min) / (signal_max - signal_min)

            peaks = find_peaks(normalized_signal[:, 0], height=0.6, distance=100)
            peaks_indices = peaks[0]

            qrs_width = 50
            for peak_index in peaks_indices:
                if peak_index <= 300:
                    continue

                qrs_complex = normalized_signal[int(peak_index - qrs_width / 2):int(peak_index + qrs_width / 2)]
                qrs_complex = qrs_complex.flatten()

                pad_left = (samples_in_window - qrs_width) // 2
                pad_right = samples_in_window - pad_left - qrs_width
                sample = np.pad(qrs_complex, pad_width=(pad_left, pad_right), mode='constant')

                if plot_time_windows:
                    plt.figure()
                    x = np.arange(int(peak_index - qrs_width / 2), int(peak_index + qrs_width / 2))
                    plt.plot(x, qrs_complex, 'b-', label='QRS Complex')
                    plt.plot(np.arange(peak_index - 300, peak_index + 300),
                             normalized_signal[peak_index - 300:peak_index + 300], 'r:', label='Full sample')
                    plt.legend(loc='best')
                    plt.show()
                    plt.figure()
                    plt.plot(np.arange(samples_in_window), sample, linewidth=3)

                # If unable to extract enough points (e.g. sampling at the very end of the signal), then disregard
                if sample.shape[0] != samples_in_window:
                    continue
                samples.append(sample)

        # If start at r peak or random start, then just loop through time slices of the signal
        elif sample_type == 'start_at_r_peak' or sample_type == 'random_start':
            num_windows = len(signal) // samples_in_window
            num_windows_list.append(num_windows)
            for window_index in range(num_windows):
                start_index = window_index * samples_in_window
                sample = signal[start_index:start_index + samples_in_window]

                signal_max = np.max(sample)
                signal_min = np.min(sample)

                if normalize:
                    sample = (sample - signal_min) / (signal_max - signal_min)

                # If start at R peak, then find the peak in the extracted sample and shift to start there
                # Must normalize again
                if sample_type == 'start_at_r_peak':
                    peaks = find_peaks(sample[:, 0], height=0.95)

                    if len(peaks[0]) == 0:
                        continue

                    peak_index = peaks[0][0]

                    if plot_time_windows:
                        plt.plot(np.arange(len(sample)), sample)
                        plt.plot(peak_index, sample[peak_index], 'ro')
                        plt.show()

                    sample = signal[start_index + peak_index:start_index + peak_index + samples_in_window]
                    signal_max = np.max(sample)
                    signal_min = np.min(sample)

                    if normalize:
                        sample = (sample - signal_min) / (signal_max - signal_min)

                    if len(sample) != samples_in_window:
                        continue

                    if plot_time_windows:
                        plt.plot(np.arange(len(sample)), sample)
                        plt.show()

                sample_metadata.append([signal_min, signal_max])
                samples.append(sample[:, 0])
        elif sample_type == 'full_heartbeat':
            samples, sample_labels = heartbeat(signal, self.labels, samples_in_window)

        if lpf:
            samples = [moving_average(sample, 5) for sample in samples]

        if plot_time_windows:
            plt.figure()
            plt.title('ECG Heartbeat Sample')
            plt.xlabel('Time Index')
            plt.ylabel('ECG Signal Value')
            plt.plot(np.arange(len(samples[0])), samples[0])
            plt.show()

        samples = np.stack(samples, axis=0)
        self.features[signal_name] = np.asarray(samples)
        self.sample_labels = np.asarray(sample_labels)
        self.metadata[signal_name] = np.asarray(sample_metadata)
        print(self.features[signal_name].shape, self.sample_labels.shape)

        num_features = len(self.features['chest_ECG'])

        assert (all(x == num_windows_list[0] for x in num_windows_list))
        return num_features
