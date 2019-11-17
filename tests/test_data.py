import os

from classifier.data import load_dataset, load_image

DATA_FOLDER = 'balanced_data'


def test_load_dataset():
    if os.path.exists(DATA_FOLDER):
        dataset = load_dataset(DATA_FOLDER)


def test_load_image():
    if os.path.exists(DATA_FOLDER):
        split = os.listdir(DATA_FOLDER)[0]
        folder = os.path.join(DATA_FOLDER, split)
        cls = os.listdir(folder)[0]
        folder = os.path.join(folder, cls)
        file = os.listdir(folder)[0]
        path = os.path.join(folder, file)
        image = load_image(path)
        assert image.size(0) == 3 and image.size(1) == 224 and image.size(2) == 224
