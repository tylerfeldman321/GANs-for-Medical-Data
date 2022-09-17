from subject_data import *
from tqdm import tqdm
import numpy as np


class DoppelGANgerDataset:
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

        with open(os.path.join(DOPPELGANGER_OUTPUT_DIR, 'data_train.npz'), 'wb') as f:
            np.savez(f, data_feature=data_feature, data_attribute=data_attribute, data_gen_flag=data_gen_flag)

        self._get_data_outputs()
        with open(os.path.join(DOPPELGANGER_OUTPUT_DIR, 'data_feature_output.pkl'), 'wb') as f:
            pickle.dump(self.data_feature_output, f)
        with open(os.path.join(DOPPELGANGER_OUTPUT_DIR, 'data_attribute_output.pkl'), 'wb') as f:
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