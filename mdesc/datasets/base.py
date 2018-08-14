import pandas as pd
import os


def load_data(module_path, data_file_name):
    """
    read data as pandas dataframe

    :param module_path: string
        the module path
    :param data_file_name: string
        Name of csv file to be loaded from
    :return: pd.DataFrame
        loaded file in dataframe format
    """

    data_file_path = os.path.join(module_path, 'data', data_file_name)

    return pd.read_csv(data_file_path)

def load_wine():
    module_path = os.path.dirname(__file__)
    data = load_data(module_path, 'wine.csv')
    return data
