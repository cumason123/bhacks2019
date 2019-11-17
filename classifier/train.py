import argparse
import copy
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.dataloader import DataLoader

from classifier.data import load_dataset
from classifier.model import Resnet


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', default='balanced_data')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--freeze-weights', action='store_true')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--learning-rate', type=int, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    return parser.parse_args()


def train_model(model, loss_function, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set classifier to training mode
            else:
                model.eval()  # Set classifier to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_function(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the classifier
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best classifier weights
    model.load_state_dict(best_model_wts)
    return model


def train(args):
    train_dataset = load_dataset(os.path.join(args.data_folder, 'train'), 'train')
    valid_dataset = load_dataset(os.path.join(args.data_folder, 'validation'), 'validation')

    train_loader = DataLoader(train_dataset, shuffle=True)
    valid_loader = DataLoader(train_dataset, shuffle=True)

    model = Resnet(args.num_classes)
    dataloaders = {
        'train': train_loader,
        'val': valid_loader
    }
    dataset_size = {
        'train': len(train_dataset),
        'val': len(valid_dataset)
    }

    loss_function = nn.CrossEntropyLoss()

    if args.optimizer == 'adam':
        optimizer_class = optim.Adam
    else:
        optimizer_class = optim.SGD

    optimizer = optimizer_class(model.parameters(), lr=args.learning_rate)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    device = torch.device('cuda:0')

    train_model(model, loss_function, optimizer, scheduler, dataloaders, device, dataset_size, args.epochs)
