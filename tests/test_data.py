import os

from model.data import load_dataset, split

data_folder = 'formatted_data'


def test_load_dataset():
    if os.path.exists(data_folder):
        dataset = load_dataset(data_folder)


def test_split():
    if os.path.exists(data_folder):
        train, valid, test = split(load_dataset(data_folder))
