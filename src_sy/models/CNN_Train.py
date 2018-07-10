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
import os

from tensorboardX import SummaryWriter

#from DatasetFolder import DatasetFolder

from ReadCSV import DatasetFolder

import FocalLoss
import focalloss2d


class CNN_Train(nn.Module):
    def __init__(self, net, args):
        super().__init__()
        self.args = args
        if torch.cuda.is_available():
            self.net = net.cuda(self.args.GPU_ids)
            cudnn.benchmark = True
            #self.net = torch.nn.DataParallel(net, device_ids = self.args.GPU_ids)
        else:
            print('No availble GPU')
            sys.exit(-1)

        if self.args.train == True:
            self.trainloader, self.testloader, self.ntrain, self.ntest = self.get_loaders()
        else:
            self.data = self.get_data()

        # Loss and Optimizer
        weights = [0.036,0.002,0.084,0.134,0.037,0.391,0.316]
        self.class_weights = torch.FloatTensor(weights).cuda()
        #if not isinstance(self.args.GPU_ids, list) == 1:
        #self.criterion = nn.CrossEntropyLoss(weight=self.class_weights).cuda(self.args.GPU_ids)

        # triplet loss
        #self.criterion = nn.TripletMarginLoss(margin=1.0, p=2).cuda(self.args.GPU_ids)

        # Focal Loss
        self.criterion = focalloss2d.FocalLoss2d(gamma=2.0).cuda(self.args.GPU_ids)
        #self.criterion = FocalLoss.FocalLoss().cuda(self.args.GPU_ids)

        #else:
        #	self.criterion = nn.CrossEntropyLoss(weight=self.class_weights).cuda()

        # cudnn.benchmark = True
        #self.optimizer = optim.SGD([{'params': net.features.parameters(), 'lr': args.lr},
        #                           {'params': net.classifier.parameters(), 'lr': args.lr*100}],
        #                           lr=args.lr,
        #                            momentum=0.9,
        #                            nesterov=True,
        #                            weight_decay=0.0005)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
        #                                                        milestones=[150, 250, 350, 450],
        #                                                        gamma=0.1)

        self.optimizer = optim.Adam(net.features.parameters(),
                                    lr=args.lr,
                                    betas=(0.9, 0.99),
                                    eps=1e-8,
                                    amsgrad=True)
        self.print_net()

        if self.args.train ==  True:
            self.iterate_CNN()
        else:
            predicted = self.predict()
            np.save(self.args.prediction, predicted)

    def get_data(self):
        if self.args.model == 'inception_v3' or self.args.model == 'inceptionresnetv2':
            img_size = 299
        else:
            img_size = 224

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        data_transform = transforms.Compose([
                            transforms.Resize(400),
                            transforms.ToTensor(),
                            transforms.CenterCrop(img_size),
                            normalize])

        test_data_dir = '../data/ISIC2018/test/'
        dataset = datasets.ImageFolder(root=test_data_dir, transform=data_transform)
        dataset_loader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=30,
                                                     shuffle=False,
                                                     num_workers=6)
        return dataset_loader, len(dataset_loader)

    def getMCA(self,correct, predicted):
        mca = 0
        class_precision = []
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
            class_precision.append(acc_t)
        mca = mca/len(self.class_weights)

        acc = 0
        for i,x in enumerate(correct):
            if x==predicted[i]:
                acc = acc + 1

        acc = acc/len(predicted)*100
        return acc, mca, class_precision

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

        acc, mca, class_precision = self.getMCA(correct, predicted)
        return train_loss, acc, mca, class_precision

    def predict(self):
        self.net.eval()
        print('Predict==>')
        predicted = []
        for index, inputs in enumerate(self.data):
            inputs = inputs.cuda(self.args.GPU_ids)

            with torch.no_grad():
                inputs = Variable(inputs)
                outputs = self.net(inputs)

            pred = torch.max(outputs.data, 1)
            predicted.extend(pred.cpu().numpy())

            del inputs

        return predicted

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

        acc, mca, class_precision = self.getMCA(correct, predicted)
        return test_loss, acc, mca, class_precision, correct, predicted


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

        if self.args.model == 'inception_v3' or self.args.model == 'inceptionresnetv2':
            img_size = 299
        else:
            img_size = 224

        transform_train = transforms.Compose([
            #transforms.Resize((255,300)),
            #transforms.Resize(300),
            transforms.Resize((300,400)),
            #transforms.Resize((450,600)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            #transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomRotation([-180, 180]),
            #transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
            transforms.RandomCrop(img_size),
#            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            #transforms.Resize((255,300)),
            #transforms.Resize(300),
            #transforms.Resize((450,600)),
            transforms.Resize((300,400)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize])

        # Dataset
        print('==> Preparing data..')
        trainset = DatasetFolder(train=True, transform=transform_train, iterNo=int(self.args.iterNo), data_dir=self.args.data_dir)

        testset = DatasetFolder(train=False, transform=transform_test, iterNo=int(self.args.iterNo), data_dir=self.args.data_dir)


        # Data Loader (Input Pipeline)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=self.args.batch_size,
                                                  num_workers=30,
                                                  shuffle=True)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=self.args.batch_size,
                                                 num_workers=2,
                                                 shuffle=False)
        return trainloader, testloader, len(trainset), len(testset)


    def iterate_CNN(self):
        tr_loss_arr = []
        train_mca = []
        test_mca = []

        writer = SummaryWriter()

        for epoch in range(self.args.n_epochs):

            for index, params in enumerate(self.optimizer.state_dict()['param_groups']):
                writer.add_scalar('train/lr_' + str(index+1), params['lr'], epoch)

            #self.scheduler.step()
            train_loss, accTr, mcaTr, class_precision_train = self.train(epoch)
            if epoch %10 ==0:
                if os.path.exists(self.args.train_dir) == False:
                    os.mkdir(self.args.train_dir)
                test_loss, accTe, mcaTe, class_precision_test, correct, predicted = self.test(epoch)
                path = os.path.join(self.args.train_dir, str(epoch))
                torch.save(self.net, path)
                writer.add_scalar('test/mca', accTe, epoch)
            else:
                test_loss, accTe, mcaTe = 0,0,0

            tr_loss_arr.append([train_loss, accTr, mcaTr, test_loss, accTe, mcaTe])

            train_mca.append((mcaTr, class_precision_train))
            test_mca.append((mcaTe, class_precision_test))
            print('Epoch %d %.4f %.4f' % (epoch, mcaTr, mcaTe))

            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/mca', accTr, epoch)

            #for index, lr in enumerate(self.scheduler.get_lr()):
            #    writer.add_scalar('train/lr_' + str(index+1), lr, epoch)
                #writer.add_scalar('train/lr', param_group['lr'], epoch)

            #print (self.args.desc);

            #print('----------------------', torch.__version__)
            #print ('Epoch	TrLoss	TrAcc	TrMCA  TeLoss	TeMCA');
            #for i in range(len(tr_loss_arr)):
            #    print ('%d %.4f  %.3f%%  %.3f%% %.4f  %.3f%%  %.3f%%'
            #        %(i, tr_loss_arr[i][0], tr_loss_arr[i][1], tr_loss_arr[i][2],
            #          tr_loss_arr[i][3], tr_loss_arr[i][4], tr_loss_arr[i][5]))
        np.save(self.args.logfile, [train_mca, test_mca, class_precision_train, class_precision_test, correct, predicted])


    def getMCA(self,correct, predicted):
        mca = 0
        class_precision = []
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
            class_precision.append(acc_t)
        mca = mca/len(self.class_weights)

        acc = 0
        for i,x in enumerate(correct):
            if x==predicted[i]:
                acc = acc + 1

        acc = acc/len(predicted)*100
        return acc, mca, class_precision
