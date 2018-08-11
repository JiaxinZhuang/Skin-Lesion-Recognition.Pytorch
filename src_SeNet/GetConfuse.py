from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import pandas
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
import torch
import torchvision
import FineTuneModel
import torchvision.transforms as transform
from torchvision import models
from ReadCSV import DatasetFolder
import pretrainedmodels
import pandas as pd

from collections import defaultdict
lr = 0.01/100
batch_size = 32
n_epochs = 250
GPU_ids = 0
nclass = 7

CUDA_VISIBLE_DEVICES = "0, 1"
parser = argparse.ArgumentParser(description='Resnet')
parser.add_argument('--lr', default=lr)
parser.add_argument('--batch_size', default=batch_size)
parser.add_argument('--n_epochs', default=n_epochs)
parser.add_argument('--GPU_ids', default=GPU_ids)
parser.add_argument('--nclass', default=nclass)
parser.add_argument('--desc', default='Resnet')
args = parser.parse_args()

resnet_model = torch.load('./model/senet154150.pkl')
resnet_model = nn.DataParallel(resnet_model, device_ids=[0, 1])

resnet_model = resnet_model.cuda()
cudnn.benchmark = True

def get_loaders():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.Resize((300, 400)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation([-180, 180]),
        transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
        transforms.RandomCrop(224),
        #            transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.Resize((300, 400)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])


    # Dataset
    print('==> Preparing data..')
    testset = DatasetFolder(train=False, transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=1,
                                         shuffle=False, num_workers=1)
    return testloader, len(testset)

testloader, ntest = get_loaders()

weights = [0.036, 0.002, 0.084, 0.134, 0.037, 0.391, 0.316]
class_weights = torch.FloatTensor(weights).cuda()
# if not isinstance(self.args.GPU_ids, list) == 1:
criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(args.GPU_ids)
optimizer = optim.Adam(resnet_model.parameters(),
                           lr=args.lr,
                           betas=(0.9, 0.99), eps=1e-8,
                           amsgrad=True)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
 #                                                     milestones=[150, 250, 350, 450],
  #                                                    gamma=0.1)


def getMCA(correct, predicted):
    mca = 0
    for lbl, w in enumerate(class_weights):
        count = 0.0
        tot = 0.0
        for i, x in enumerate(correct):
            if x == lbl:
                tot = tot + 1
                if x == predicted[i]:
                    count = count + 1

        acc_t = count / tot * 100.0
        mca = mca + acc_t
    mca = mca / len(class_weights)

    acc = 0
    for i, x in enumerate(correct):
        if x == predicted[i]:
            acc = acc + 1

    acc = acc / len(predicted) * 100
    return acc, mca

def test(epoch):
    global best_acc
    resnet_model.eval()
    test_loss = 0
    correct = []
    predicted = []
    print('Testing==>')
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # if not isinstance(self.args.GPU_ids, list) == 1:
        inputs, targets = inputs.cuda(args.GPU_ids), targets.cuda(args.GPU_ids)
        # else: inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = resnet_model(inputs)
            loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, pred = torch.max(outputs.data, 1)
        correct.extend(targets.cpu().numpy())
        predicted.extend(pred.cpu().numpy())

        del inputs, targets

        if (batch_idx + 1) % 100 == 0:
            print('Completed: [%d/%d]' % (batch_idx + 1, ntest // args.batch_size))

    acc, mca = getMCA(correct, predicted)
    return test_loss, acc, mca

def predict():
    resnet_model.eval()
    print('Predict==>')
    predicted = []
    correct = []
    for index, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(args.GPU_ids), targets.cuda(args.GPU_ids)

        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = resnet_model(inputs)

        _, pred = torch.max(outputs.data, 1)
        # pred = torch.max(outputs.data, 1)
        correct.extend(targets.cpu().numpy())
        predicted.extend(pred.cpu().numpy())

        del inputs, targets
    npy = []
    npy.append(predicted)
    npy.append(correct)
    np.save(npy, './confuse.npy')
    return predicted

pre = predict()

