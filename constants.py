import os
from output import *

# Configurations
FEATURE_KEYS = {'chest_ECG': OutputType.CONTINUOUS}  # Which features to select
ATTRIBUTE_KEYS = {'Age': OutputType.DISCRETE, 'Weight (kg)': OutputType.DISCRETE}  # Which attributes/metadata to select

# Path configurations
DATA_DIR = os.path.join('data', 'WESAD')

# Dataset constants
SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
E4_DATA_FILENAMES = ['ACC.csv', 'BVP.csv', 'EDA.csv', 'HR.csv', 'IBI.csv', 'TEMP.csv', 'tags.csv', 'info.txt']

WRIST_KEYS = ['ACC', 'BVP', 'EDA', 'TEMP']
CHEST_KEYS = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']

# Readme Data
SECTION_TITLES = ['### Personal information ###', '### Study pre-requisites ###', '### Additional notes ###']
PERSONAL_INFO_KEYS = ['Age', 'Height (cm)', 'Weight (kg)', 'Gender', 'Dominant hand']
GENDER_AND_DOMINANT_HAND_MAPPING = {'male': 1, 'female': 0, 'right': 1, 'left': 0}
YES_NO_MAPPING = {'NO': 0, 'YES': 1}
STUDY_PREREQUISITE_KEYS = ['Did you drink coffee today?', 'Did you drink coffee within the last hour?',
                           'Did you do any sports today?', 'Are you a smoker?', 'Did you smoke within the last hour?',
                           'Do you feel ill today?']

# Sampling rates
SAMPLING_RATES = {'wrist_ACC': 32, 'wrist_BVP': 64, 'wrist_EDA': 4, 'wrist_TEMP': 4, 'chest_ACC': 700,
                  'chest_ECG': 700, 'chest_EDA': 700, 'chest_EMG': 700, 'chest_Temp': 700, 'chest_Resp': 700}
