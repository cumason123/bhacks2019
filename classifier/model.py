import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


class Model(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class TransferModel(Model):
    def __init__(self, pretrained_model_function, num_classes):
        super().__init__()
        model = pretrained_model_function(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)


class Resnet(TransferModel):
    def __init__(self, num_classes):
        super().__init__(resnet18, num_classes)
