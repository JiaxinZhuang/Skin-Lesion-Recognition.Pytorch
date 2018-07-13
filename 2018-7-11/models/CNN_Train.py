from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
from DatasetFolder import DatasetFolder
import os

from tensorboardX import SummaryWriter

class CNN_Train(nn.Module):
    def __init__(self, net, args):
        super(CNN_Train, self).__init__()
        self.args = args
        if torch.cuda.is_available():
            #if not isinstance(self.args.GPU_ids, list) == 1:
            self.net = net.cuda(self.args.GPU_ids)
            #else: net = torch.nn.DataParallel(net, device_ids = self.args.GPU_ids)

        self.net = net.cuda()
        cudnn.benchmark = True
        self.trainloader, self.testloader, self.ntrain, self.ntest = self.get_loaders()


        # Loss and Optimizer
        weights = [0.036,0.002,0.084,0.134,0.037,0.391,0.316]
        self.class_weights = torch.FloatTensor(weights).cuda()
        #if not isinstance(self.args.GPU_ids, list) == 1:
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights).cuda(self.args.GPU_ids)
        #else:
        #	self.criterion = nn.CrossEntropyLoss(weight=self.class_weights).cuda()

        # cudnn.benchmark = True
#        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad,net.parameters()),
        self.optimizer = optim.SGD([{'params': net.features.parameters(), 'lr': args.lr},
                                   {'params': net.classifier.parameters(), 'lr': args.lr*100}],
                                   lr=args.lr,
                                    momentum=0.9,
                                    nesterov=True,
                                    weight_decay=0.0005)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                milestones=[150, 250, 350, 450],
                                                                gamma=0.1)
        self.print_net()
        self.iterate_CNN()

    # Training
    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = []
        predicted = []
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            #if not isinstance(self.args.GPU_ids, list) == 1:
            inputs, targets = inputs.cuda(self.args.GPU_ids), targets.cuda(self.args.GPU_ids)
            #else: inputs, targets = inputs.cuda(), targets.cuda()


            self.optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            correct.extend(targets.cpu().numpy())
            predicted.extend(pred.cpu().numpy())

            if (batch_idx+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                    %(epoch+1, self.args.n_epochs, batch_idx+1, self.ntrain//self.args.batch_size, loss.item()))

        acc, mca = self.getMCA(correct, predicted)
        return train_loss, acc, mca


    def test(self, epoch):
        global best_acc
        self.net.eval()
        test_loss = 0
        correct = []
        predicted = []
        print ('Testing==>')
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            #if not isinstance(self.args.GPU_ids, list) == 1:
            inputs, targets = inputs.cuda(self.args.GPU_ids), targets.cuda(self.args.GPU_ids)
            #else: inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

            test_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            correct.extend(targets.cpu().numpy())
            predicted.extend(pred.cpu().numpy())

            del inputs, targets

            if (batch_idx+1) % 100 == 0:
                print('Completed: [%d/%d]' %(batch_idx+1, self.ntest//self.args.batch_size))

        acc, mca = self.getMCA(correct, predicted)
        return test_loss, acc, mca


    def print_net(self):
        print('----------------------------')
        print(self.net)
        params = list(self.net.parameters())
        # for p in params:
        #    print(p.size())  # conv1's .weight
        pytorch_total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(len(params))
        print('total parameters %d'%(pytorch_total_params))
        pytorch_total_params = float(pytorch_total_params)/10**6
        print('total parameters requires_grad %.3f M'%(pytorch_total_params))

        pytorch_total_params = sum([param.nelement() for param in self.net.parameters()])
        print('total parameters %d'%(pytorch_total_params))
        pytorch_total_params = float(pytorch_total_params)/10**6
        print('total parameters %.3f M'%(pytorch_total_params))
        print('----------------------------')
        #return pytorch_total_params

    def get_loaders(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#        transform_train = transforms.Compose([
#            transforms.Resize(350),
#            transforms.RandomHorizontalFlip(),
#            transforms.RandomVerticalFlip(),
#            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
#            transforms.RandomRotation([-180, 180]),
#            transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.6, 1.4]),
#            transforms.RandomCrop(224),
#            transforms.ToTensor(),
#            normalize
#        ])

        imsize = 300

        transform_train = transforms.Compose([
            transforms.Resize(imsize),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomRotation([-180, 180]),
            transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
            transforms.RandomCrop(224),
#            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            transforms.Resize(imsize),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

        # Dataset
        print('==> Preparing data..')
        trainset = DatasetFolder(train=True, transform=transform_train, iterNo=self.args.iterNo)

        testset = DatasetFolder(train=False, transform=transform_test, iterNo=self.args.iterNo)


        # Data Loader (Input Pipeline)
        #prob = trainset.get_weights_for_balanced_classes()
        #prob = torch.DoubleTensor(prob)
        #sampler = torch.utils.data.sampler.WeightedRandomSampler(prob, len(prob))

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=self.args.batch_size,
                                                  num_workers=50,
                                                  shuffle=True) #sampler = sampler)		#
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=self.args.batch_size,
                                                 num_workers=2,
                                                 shuffle=False)
        return trainloader, testloader, len(trainset), len(testset)


    def iterate_CNN(self):
        tr_loss_arr = []

        output_writer_path = os.path.join('./run', 'baseline')
        writer = SummaryWriter(output_writer_path)

        for epoch in range(self.args.n_epochs):
            self.scheduler.step()
            train_loss, accTr, mcaTr = self.train(epoch)
            if epoch%10==0:
                train_dir = '../train_dir/baseline'
                if os.path.exists(train_dir) == False:
                    os.mkdir(train_dir)
                test_loss, accTe, mcaTe = self.test(epoch)
                path = os.path.join(train_dir, str(epoch))
                torch.save(self.net, path)
                writer.add_scalar('test/acc', accTe, epoch)
                writer.add_scalar('test/mca', mcaTe, epoch)
            else: test_loss, accTe, mcaTe = 0,0,0
            tr_loss_arr.append([train_loss, accTr, mcaTr, test_loss, accTe, mcaTe])
            print (self.args.desc);

            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/mca', mcaTr, epoch)
            writer.add_scalar('train/acc', accTr, epoch)

            for index, lr in enumerate(self.scheduler.get_lr()):
                writer.add_scalar('train/lr_' + str(index+1), lr, epoch)

            print('----------------------', torch.__version__)
            print ('Epoch	TrLoss	TrAcc	TrMCA  TeLoss	TeMCA');
            for i in range(len(tr_loss_arr)):
                print ('%d %.4f  %.3f%%  %.3f%% %.4f  %.3f%%  %.3f%%'
                    %(i, tr_loss_arr[i][0], tr_loss_arr[i][1], tr_loss_arr[i][2],
                      tr_loss_arr[i][3], tr_loss_arr[i][4], tr_loss_arr[i][5]))


    def getMCA(self,correct, predicted):
        mca = 0
        for lbl,w in enumerate(self.class_weights):
            count = 0.0
            tot = 0.0
            for i,x in enumerate(correct):
                if x==lbl:
                    tot = tot + 1
                    if x==predicted[i]:
                        count = count+1

            acc_t = count/tot*100.0
            mca = mca + acc_t
        mca = mca/len(self.class_weights)

        acc = 0
        for i,x in enumerate(correct):
            if x==predicted[i]:
                    acc = acc + 1

            acc = acc/len(predicted)*100
            return acc, mca





