import os

from classifier.data import load_dataset

DATA_FOLDER = 'balanced_data'


def test_load_dataset():
    if os.path.exists(DATA_FOLDER):
        dataset = load_dataset(DATA_FOLDER)
