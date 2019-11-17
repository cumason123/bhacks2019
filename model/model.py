import torch
import torch.nn as nn


def build_model(num_classes, model_function):
    model = model_function(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


class Model(nn.Module):

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class TransferModel(Model):

    def __init__(self, pretrained_model_function, num_classes):
        super().__init__()
        self.model = build_model(num_classes, pretrained_model_function)

    def forward(self, x):
        return self.model(x)
