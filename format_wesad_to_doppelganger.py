from constants import *
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from output import *
from tempfile import TemporaryFile
from tqdm import tqdm


def _convert_readme_info_to_numerical(info, info_key):
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





class SubjectData:
    def __init__(self, subject_id, sample_length=300):
        self.metadata = {}
        self.features = {}
        self.signals = {}
        self.id = subject_id
        self.sample_length = sample_length
        assert (subject_id in SUBJECT_IDS), 'Subject ID is not in SUBJECT IDS'

        # Collect sensor measurement data from the pickle file
        pkl_filepath = os.path.join(DATA_DIR, f'S{subject_id}', f'S{subject_id}.pkl')
        with open(pkl_filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        self.subject = data['subject']
        # self.wrist = data['signal']['wrist'] # IGNORING WRIST DATA FOR NOW SINCE IT HAS DIFFERENT SAMPLING RATE
        self.chest = data['signal']['chest']
        self.labels = data['label']
        # for signal_name in self.wrist:
        #     self.signals['wrist_' + signal_name] = downsample(self.wrist[signal_name]) # IGNORING WRIST DATA FOR NOW SINCE IT HAS DIFFERENT SAMPLING RATE
        for signal_name in self.chest:
            self.signals['chest_' + signal_name] = downsample(self.chest[signal_name])

        # Take the signals and convert them into smaller time series features
        self.num_features = self._extract_time_windows(samples_in_window=self.sample_length)

        # Collect data from the subject readme
        readme_filepath = os.path.join(DATA_DIR, f'S{subject_id}', f'S{subject_id}_readme.txt')
        self.personal_info = {}
        self.prerequisite_questions = {}
        with open(readme_filepath, 'r') as f:
            lines = f.read()
            for personal_info_key in PERSONAL_INFO_KEYS:
                info = lines.split(personal_info_key)[1][2:].split('\n')[0]
                info = _convert_readme_info_to_numerical(info, personal_info_key)
                self.metadata[personal_info_key] = np.ones((self.num_features, 1)) * info
            for study_prerequisite_question in STUDY_PREREQUISITE_KEYS:
                info = lines.split(study_prerequisite_question)[1][1:].split('\n')[0]
                info = _convert_readme_info_to_numerical(info, study_prerequisite_question)
                self.metadata[study_prerequisite_question] = np.ones((self.num_features, 1)) * info

    def _extract_time_windows(self, normalize=True, samples_in_window=300):
        num_windows_list = []
        # For each signal in self.chest
        for signal_name in self.signals:
            signal = self.signals[signal_name]
            signal_time_windows = []
            signal_time_window_metadata = []

            num_windows = len(signal) // samples_in_window
            num_windows_list.append(num_windows)

            # Extract small time windows from the signal and normalize. Add the min and max to the metadata if we are normalizing
            for window_index in range(num_windows):
                start_index = window_index * samples_in_window
                signal_time_window = signal[start_index:start_index + samples_in_window]

                signal_max = np.max(signal_time_window)
                signal_min = np.min(signal_time_window)

                if normalize:
                    signal_time_window = (signal_time_window - signal_min) / (signal_max - signal_min)

                signal_time_window_metadata.append([signal_min, signal_max])
                signal_time_windows.append(signal_time_window)

            self.features[signal_name] = np.asarray(signal_time_windows)
            self.metadata[signal_name] = np.asarray(signal_time_window_metadata)

        assert(all(x == num_windows_list[0] for x in num_windows_list))
        return num_windows_list[0]


class SubjectDataset:
    def __init__(self):
        self.subjects = {}
        self.data_feature_output = []
        self.data_attribute_output = []
        self.data_train = None
        for subject_id in tqdm(SUBJECT_IDS, desc='Loading Data From Each Subject'):
            self.subjects[subject_id] = SubjectData(subject_id)
        self.num_features = self.subjects[subject_id].num_features

    def create_doppelganger_data(self, verbose=False):
        data_feature = None
        data_attribute = None
        data_gen_flag = None
        first = True
        for subject in tqdm(self.subjects, desc='Creating Data for Each Subject'):
            subject_data = self.subjects[subject]
            subject_id = subject_data.id
            subject_features = self._select_features(subject_id)
            subject_attributes = self._select_attributes(subject_id)
            if first:
                data_feature = subject_features
                data_attribute = subject_attributes
                first = False
            else:
                data_feature = np.concatenate((data_feature, subject_features), axis=0)
                data_attribute = np.concatenate((data_attribute, subject_attributes), axis=0)

        data_gen_flag = np.ones((data_feature.shape[0], data_feature.shape[1]))  # For now just setting gen to 1 for all signals

        with open('data_train.npz', 'wb') as f:
            np.savez(f, data_feature=data_feature, data_attribute=data_attribute, data_gen_flag=data_gen_flag)

        self._get_data_outputs()
        with open('data_feature_output.pkl', 'wb') as f:
            pickle.dump(self.data_feature_output, f)
        with open('data_attribute_output.pkl', 'wb') as f:
            pickle.dump(self.data_attribute_output, f)

        if verbose:
            print(f'Data Feature: {data_feature}, Data Feature Shape: {data_feature.shape}')
            print(f'Data Attribute: {data_attribute}, Data Attribute Shape: {data_attribute.shape}')
            print(f'Data Gen Flag: {data_gen_flag}, Data Gen Flag: {data_gen_flag.shape}')
            print('data_feature_output:')
            for output in self.data_feature_output:
                output.print()
                print('data_attribute_output:')
            for output in self.data_attribute_output:
                output.print()

    def _select_features(self, subject_id, keys=FEATURE_KEYS):
        subject_data = self.subjects[subject_id]
        subject_features = None
        first = True
        for feature_key in keys:
            feature_data = subject_data.features[feature_key]
            if first:
                subject_features = feature_data
                first = False
            else:
                subject_features = np.concatenate((subject_features, feature_data), axis=2)
        return subject_features

    def _select_attributes(self, subject_id, keys=ATTRIBUTE_KEYS):
        subject_data = self.subjects[subject_id]
        subject_attributes = None
        first = True
        for attribute_key in keys:
            attribute_data = subject_data.metadata[attribute_key]
            if first:
                subject_attributes = attribute_data
                first = False
            else:
                subject_attributes = np.concatenate((subject_attributes, attribute_data), axis=1)
        return subject_attributes

    def _get_data_outputs(self, feature_keys=FEATURE_KEYS, attribute_keys=ATTRIBUTE_KEYS):
        subject_data = self.subjects[SUBJECT_IDS[0]]

        for feature_key in feature_keys:
            feature_data = subject_data.features[feature_key]
            feature_output = Output(type_=feature_keys[feature_key], dim=feature_data.shape[2],
                                    normalization=Normalization.ZERO_ONE, is_gen_flag=False)
            self.data_feature_output.append(feature_output)

        for attribute_key in attribute_keys:
            attribute_data = subject_data.metadata[attribute_key]
            attribute_output = Output(type_=attribute_keys[attribute_key], dim=attribute_data.shape[1],
                                      normalization=Normalization.ZERO_ONE, is_gen_flag=False)
            self.data_attribute_output.append(attribute_output)


def main():
    """
    Loads the data from each subject listed in
    :return: Saves three files
    """
    s = SubjectDataset()
    s.create_doppelganger_data()


if __name__ == '__main__':
    main()
