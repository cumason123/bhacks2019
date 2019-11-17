import argparse
import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models.resnet as resnet
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from classifier.data import load_dataset
from classifier.model import TransferModel


def train_model(model, loss_function, optimizer, scheduler, dataloaders, device, dataset_sizes, args, num_epochs=25):
    since = time.time()

    eval_interval = 10
    model.to(device)

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
                if not (epoch - eval_interval + 1) % eval_interval == 0:
                    continue
                model.eval()  # Set classifier to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            pbar = tqdm(dataloaders[phase])
            for inputs, labels in pbar:
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

                model.save(os.path.join(args.save_dir, args.model_file))

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)

    if hasattr(resnet, args.pretrained_model):
        model_cls = getattr(resnet, args.pretrained_model)
    else:
        raise ModuleNotFoundError

    model = TransferModel(model_cls, len(train_dataset.classes))
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

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if torch.cuda.device_count() > 1:
        print('using data parallel')
        model = nn.DataParallel(model)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = train_model(model, loss_function, optimizer, scheduler, dataloaders, device, dataset_size, args,
                        args.epochs)

    model.save(os.path.join(args.save_dir, args.pretrained_model + '_' + args.model_file))
