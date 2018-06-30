"""Using resnet152 provided by pytorch"""

from torchvision import models
from torchvision import transforms
from torch import nn
from torch import optim
import os, sys
import torch

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import copy

import statistics

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((300,400)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((300, 400)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = '../data/task3'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            hps = None
            statistics_ = statistics.statistics(hps, mode='evaluate')
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # TODO
                statistics_.add_labels_predictions(labels.data, preds)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            im_acc = statistics_.get_acc_imbalanced()

            print('{} Loss: {:.4f} Acc: {:.4f} im_Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, im_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model
    resnet_model = models.resnet152(pretrained=True)
    for param in resnet_model.parameters():
        param.requires_grad = False
    # add layers
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc= nn.Linear(in_features=num_ftrs, out_features=7)
    #resnet_model = resnet_model.to(device)
    #resnet_model = torch.nn.parallel.DistributedDataParallel(resnet_model)
    resnet_model = resnet_model.to(device)


    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(params=resnet_model.fc.parameters(), lr=1e-3, momentum=0.9)

    optimizer_conv = optim.SGD(resnet_model.fc.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(resnet_model, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=1000)

main()
