import os

from model.data import load_dataset


def test_load_dataset():
    data_folder = 'formatted_data'
    if os.path.exists(data_folder):
        load_dataset(data_folder)
