import os

import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from classifier.data import load_dataset
from classifier.model import TransferModel


def evaluate(args):
    test_dataset = load_dataset(os.path.join(args.data_folder, 'test'), 'test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                             pin_memory=True)
    if hasattr(resnet, args.pretrained_model):
        model_cls = getattr(resnet, args.pretrained_model)
    else:
        raise ModuleNotFoundError
    model = TransferModel(model_cls, len(test_dataset.classes))
    loss_function = nn.CrossEntropyLoss()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.load(os.path.join(args.save_dir, args.model_file))
    model.to(device)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    pbar = tqdm(test_loader)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(test_dataset)
    epoch_acc = running_corrects.double() / len(test_dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('Test', epoch_loss, epoch_acc))
