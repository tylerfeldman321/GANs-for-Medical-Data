from subject_data import *
from tqdm import tqdm
import numpy as np
import pandas as pd


class ECGSynthesisDataset:
    def __init__(self):
        self.subjects = {}
        self.ecg_samples = None
        self.sample_labels = None

        # Collect data for each subject
        for subject_id in tqdm(SUBJECT_IDS, desc='Loading Data From Each Subject'):
            self.subjects[subject_id] = SubjectData(subject_id, sample_length=187)
        self.num_features = self.subjects[subject_id].num_features

        # # Get data separately for each subject
        # for subject_id in SUBJECT_IDS:
        #     subject_ecg_samples = self.subjects[subject_id].features['chest_ECG']
        #
        #     # If 3D array (the third dimension in this case would be 1), then convert to 2D array
        #     if len(subject_ecg_samples.shape) == 3:
        #         subject_ecg_samples = subject_ecg_samples[:, :, 0]
        #
        #     df = pd.DataFrame(subject_ecg_samples)
        #     df[subject_ecg_samples.shape[1]] = pd.Series([0]*subject_ecg_samples.shape[0]) # Set class to zero
        #
        #     # Create train and test sets
        #     indices = np.arange(0, subject_ecg_samples.shape[0])
        #     np.random.shuffle(indices)
        #
        #     train_indicies = indices[:int(subject_ecg_samples.shape[0]*.75)]
        #     test_indicies = indices[int(subject_ecg_samples.shape[0] * .75):]
        #
        #     train_df = df.loc[train_indicies]
        #     test_df = df.loc[test_indicies]
        #
        #     train_df.to_csv(os.path.join(ECG_SYN_OUTPUT_DIR, f'wesad_train_full_heartbeat_subject_{subject_id}.csv'), index=False, header=False)
        #     test_df.to_csv(os.path.join(ECG_SYN_OUTPUT_DIR, f'wesad_test_full_heartbeat_subject_{subject_id}.csv'), index=False, header=False)

        # Get data for all subjects
        subject_classes = []
        for subject_id in SUBJECT_IDS:
            new_ecg_samples = self.subjects[subject_id].features['chest_ECG']
            new_sample_labels = self.subjects[subject_id].sample_labels
            if subject_id == SUBJECT_IDS[0]:
                self.ecg_samples = new_ecg_samples
                self.sample_labels = new_sample_labels
            else:
                self.ecg_samples = np.concatenate((self.ecg_samples, new_ecg_samples), axis=0)
                self.sample_labels = np.concatenate((self.sample_labels, new_sample_labels), axis=0)
            subject_classes += [subject_id] * new_ecg_samples.shape[0]

        print(len(subject_classes))
        print(self.ecg_samples.shape[0])

        # If 3D array (the third dimension in this case would be 1), then convert to 2D array
        if len(self.ecg_samples.shape) == 3:
            self.ecg_samples = self.ecg_samples[:, :, 0]

        df = pd.DataFrame(self.ecg_samples)
        include_signal_class = True
        include_subject_labels = False
        if include_signal_class:
            # TODO: make it able to save the actual class of the heartbeats
            # df[df.shape[1]] = pd.Series([0] * self.ecg_samples.shape[0])  # Set class to zero
            df[df.shape[1]] = pd.Series(self.sample_labels[:, 0])
        if include_subject_labels:
            df[df.shape[1]] = pd.Series(subject_classes)

        # Create train and test sets
        indices = np.arange(0, self.ecg_samples.shape[0])
        np.random.shuffle(indices)

        train_indicies = indices[:int(self.ecg_samples.shape[0]*.75)]
        test_indicies = indices[int(self.ecg_samples.shape[0] * .75):]

        train_df = df.loc[train_indicies]
        test_df = df.loc[test_indicies]

        print(train_df.info(), train_df.head(), test_df.info(), test_df.head())

        train_df.to_csv(os.path.join(ECG_SYN_OUTPUT_DIR, 'wesad_train_full_heartbeat_affected_state_labels.csv'), index=False, header=False)
        test_df.to_csv(os.path.join(ECG_SYN_OUTPUT_DIR, 'wesad_test_full_heartbeat_affected_state_labels.csv'), index=False, header=False)
