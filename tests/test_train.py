from classifier.train import *
from classifier.model import Resnet


def test_build_model():
    num_classes = 10
    model = Resnet(num_classes)
    output = model(torch.zeros(1, 3, 100, 100))
    assert num_classes == output.size(1)
