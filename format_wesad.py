import pandas as pd
from ecg_synthesis_dataset import ECGSynthesisDataset
from doppelganger_dataset import DoppelGANgerDataset


def main():
    """
    Loads the data from each subject listed in
    :return: Saves three files
    """
    # s = DoppelGANgerDataset()
    # s.create_doppelganger_data()
    s = ECGSynthesisDataset()


def check_csvs():
    df_train = pd.read_csv('wesad_train.csv')
    print(df_train.info())
    print(df_train.head())
    df_test = pd.read_csv('wesad_test.csv')
    print(df_test.info())
    print(df_test.head())


if __name__ == '__main__':
    # main()
    main()
