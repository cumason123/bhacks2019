import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset


def load_dataset(data_folder):
    image_dataset = ImageFolder(data_folder)
    return image_dataset


def split(dataset, seed=0, valid_split=0.1, test_split=0.1):
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    i = int(len(dataset) * (1 - valid_split - test_split))
    j = int(len(dataset) * (1 - test_split))
    train = Subset(dataset, indices[:i])
    valid = Subset(dataset, indices[i:j])
    test = Subset(dataset, indices[j:])
    return train, valid, test
