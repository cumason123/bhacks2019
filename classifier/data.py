import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset






def load_dataset(data_folder):
    train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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
