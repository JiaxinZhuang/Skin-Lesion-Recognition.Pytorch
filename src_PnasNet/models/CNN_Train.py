import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision.transforms as transforms
import numpy as np
import os, sys

from tensorboardX import SummaryWriter

from DatasetFolder import DatasetFolder
from DatasetFolder import testsetFolder

import focalloss2d
import pandas as pd
from collections import defaultdict
import shutil


class CNN_Train(nn.Module):
    """Main class to train and generate result."""

    def __init__(self, net, args, checkpoint=None):
        super(CNN_Train, self).__init__()
        self.args = args
        self.best_pred = None

        # GPU
        if torch.cuda.is_available():
            cuda = self.args.c.split(',')
            device_ids = list(range(len(cuda)))
            print('=> Using device_ids ', *device_ids)
            self.net = nn.DataParallel(net, device_ids=device_ids).cuda()
            cudnn.benchmark = True
        else:
            print('No availble GPU')
            sys.exit(-1)

        # Loss
        #   class weigth
        weights = [0.036,0.002,0.084,0.134,0.037,0.391,0.316]
        self.class_weights = torch.FloatTensor(weights).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights).cuda()

        # focal loss
        self.criterion = focalloss2d.FocalLoss2d(gamma=2.0).cuda()

        # optimizer
        #self.optimizer = optim.SGD([{'params': net.features.parameters(), 'lr': lr},
        #                           {'params': net.classifier.parameters(), 'lr': lr*100}],
        #                           lr=lr,
        #                            momentum=0.9,
        #                            nesterov=True,
        #                            weight_decay=0.0005)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
        #                                                        milestones=[150, 250, 350, 450],
        #                                                        gamma=0.1)
        #print('using optim {} with init_lr: {}'.format('SGD', lr))

        lr = float(self.args.lr)
        self.optimizer = optim.Adam(
                                   #[{'params': net.features.parameters(), 'lr': lr},
                                    # {'params': net.classifier.parameters(), 'lr': lr*100}],
                                    net.parameters(),
                                    lr=lr,
                                    betas=(0.9, 0.99),
                                    eps=1e-8,
                                    amsgrad=True)

        if self.args.lm == True and checkpoint != None:
            print("=> Using model {}".format(self.args.model))

            self.args.start_epoch = checkpoint['epoch']
            self.best_pred = checkpoint['best_pred']
            self.net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.args.start_epoch, checkpoint['epoch']))

        #print('using optim {} with init_lr: {}'.format('Adam', lr))

        #self.print_net()

        # train & test
        if self.args.train == True:
            self.trainloader, self.testloader, self.ntrain, self.ntest = self.get_loaders()
            print(self.ntrain, self.ntest)
            self.iterate_CNN()
        else:
            #train_features, train_labels, test_features, test_labels = self.extract_feature()
            #np.save(self.args.prediction, [train_features, train_labels, test_features, test_labels])
            #self.predict_valid()
            self.predict_test()

    def predict_test(self):
        """predict on test set
        """
        self.data, self.ndata = self.get_data()
        #def sigmod(x):
        #    return (1 / (1 + np.exp(-x)))
        predicted, filenames= self.predict()

        predicted = np.array(predicted)
        np.save(self.args.prediction+'_val_predict.npy', predicted)
        print('predict test finished')

        d = defaultdict(list)
        for x in predicted:
            for i, v in enumerate(x):
                d[i].append(v)
                #d[i].append(sigmod(v))

        raw_data = {
                'image': filenames,
                'MEL': d[0],
                'NV':  d[1],
                'BCC': d[2],
                'AKIEC': d[3],
                'BKL': d[4],
                'DF': d[5],
                'VASC': d[6]
                }
        df = pd.DataFrame(raw_data, columns = ['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'])
        df.to_csv(self.args.prediction, index=False)

    def predict_valid(self):
        """predict on validation set
           for fusion
        """
        epoch = -1
        _, self.testloader, _, self.ntest = self.get_loaders()
        self.data, self.ndata = self.get_data()
        test_loss, accTe, mcaTe, class_precision_test, correct, predicted = self.test(epoch)
        predicted = np.array(predicted)
        np.save(self.args.prediction+'_val_correct.npy', correct)
        np.save(self.args.prediction+'_val_predict.npy', predicted)
        print('Epoch %d %.4f' % (epoch, mcaTe))


    def extract_feature(self):
        """extract feature after conv layers
        """
        self.trainloader, self.testloader, self.ntrain, self.ntest = self.get_loaders()
        self.net.eval()

        print('Extract Train Feature==>')
        train_features = []
        train_labels = []
        for index, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = self.net(inputs)
                train_features.extend(outputs.cpu().numpy())
                train_labels.extend(targets.cpu().numpy())

            del inputs, targets

        print('Extract Test Feature==>')
        test_features = []
        test_labels = []
        for index, (inputs, targets) in enumerate(self.testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = self.net(inputs)
                test_features.extend(outputs.cpu().numpy())
                test_labels.extend(targets.cpu().numpy())

        return train_features, train_labels, test_features, test_labels


    def get_data(self):
        if self.args.model == 'inception_v3' or self.args.model == 'inceptionresnetv2':
            resize_img = 399
            img_size = 299
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif self.args.model == 'nasnetalarge' or self.args.model == 'pnasnet5large':
            resize_img = 441
            img_size = 331
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            #resize_img = 366
            #img_size = 275
        else:
            #resize_img = 250
            #resize_img = 275
            #resize_img = 300
            #resize_img = 350
            resize_img = 400
            img_size = 224
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.args.resize_img != 300:
            resize_img = self.args.resize_img

        #data_transform = transforms.Compose([
        #                    transforms.Resize(resize_img),
        #                    transforms.CenterCrop(img_size),
        #                    transforms.ToTensor(),
        #                    normalize])

        data_transform = transforms.Compose([
                            transforms.Resize(resize_img),
                            #transforms.FiveCrop(img_size),
                            transforms.TenCrop(img_size),
                            transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),
                            #transforms.CenterCrop(img_size),
                            #transforms.ToTensor(),
                            #normalize
                            ])

        test_data = 'ISIC2018_Task3_Test_Input/'
        #test_data = 'ISIC2018_Task3_Validation_Input/'
        #test_data_dir = 'valid_wpr1/'
        test_data_dir = os.path.join(self.args.data_dir, test_data)
        dataset = testsetFolder(transform=data_transform, data_dir=test_data_dir)
        dataset_loader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     num_workers=1)
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
            inputs, targets = inputs.cuda(), targets.cuda()

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
        filenames = []
        for index, (filename, inputs) in enumerate(self.data):

            bs, ncrops, c, h, w = inputs.size()
            inputs_ = inputs.view(-1, c, h, w)
            inputs_ = inputs_.cuda()

            #print(inputs_.size())
            #inputs = inputs.cuda()


            with torch.no_grad():
                inputs = Variable(inputs_)
                outputs = self.net(inputs)
                # Five Crop
                outputs = outputs.view(bs, ncrops, -1).mean(1)

                softmax_function = torch.nn.Softmax(dim=1)
                soft_outputs = softmax_function(outputs)

            #pred = torch.max(outputs.data, 1)
            #predicted.extend(outputs.cpu().numpy())
            predicted.extend(soft_outputs.cpu().numpy())
            filenames.extend(filename)

            del inputs
            del filename

        return predicted, filenames

    def test_vote(self, epoch):
        global best_acc
        self.net.eval()
        test_loss = 0
        correct = []
        predicted = []
        print ('Testing==>')
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

            test_loss += loss.item()

            pred = torch.mean(outputs.data, 0)
            mean_pred = torch.argmax(pred)
            #mean_correct = torch.mean(targets.float())

            #if epoch == -1:
            #    softmax_function = torch.nn.Softmax(dim=1)
            #    soft_outputs = softmax_function(outputs)
            #    predicted_.extend(soft_outputs.cpu().numpy())

            correct.extend(targets.cpu().numpy()[:1])
            predicted.extend([mean_pred.cpu().numpy()])

            del inputs, targets

            if (batch_idx+1) % 100 == 0:
                print('Completed: [%d/%d]' %(batch_idx+1, self.ntest//30))

        acc, mca, class_precision = self.getMCA(correct, predicted)

        #if epoch == -1:
        #    predicted = predicted_
        return test_loss, acc, mca, class_precision, correct, predicted




    def test(self, epoch):
        global best_acc
        self.net.eval()
        test_loss = 0
        correct = []
        predicted = []
        predicted_ = []
        print ('Testing==>')
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

            test_loss += loss.item()

            _, pred = torch.max(outputs.data, 1)

            if epoch == -1:
                softmax_function = torch.nn.Softmax(dim=1)
                soft_outputs = softmax_function(outputs)
                predicted_.extend(soft_outputs.cpu().numpy())

            correct.extend(targets.cpu().numpy())
            predicted.extend(pred.cpu().numpy())

            del inputs, targets

            if (batch_idx+1) % 100 == 0:
                print('Completed: [%d/%d]' %(batch_idx+1, self.ntest//self.args.batch_size))
            #print('Test batch size is {}'.format(batch_idx+1))

        acc, mca, class_precision = self.getMCA(correct, predicted)

        if epoch == -1:
            predicted = predicted_
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
        if self.args.model == 'inception_v3' or self.args.model == 'inceptionresnetv2':
            resize_img = 399
            img_size = 299
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif self.args.model == 'nasnetalarge' or self.args.model == 'pnasnet5large':
            resize_img = 441
            img_size = 331
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            resize_img = 300
            img_size = 224
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.args.resize_img != 300:
            resize_img = self.args.resize_img

        transform_train = transforms.Compose([
            transforms.Resize(resize_img),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
            transforms.RandomRotation([-180, 180]),
            transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
            transforms.RandomCrop(img_size),
            transforms.ToTensor(),
            normalize
        ])

        # Dataset
        print('==> Preparing data..')
        trainset = DatasetFolder(train=True, transform=transform_train, iterNo=int(self.args.iterNo), data_dir=self.args.data_dir, use_all_data=self.args.use_all_data)

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=self.args.batch_size,
                                                  num_workers=50,
                                                  shuffle=True)



        if self.args.use_all_data == False:
            #transform_test = transforms.Compose([
            #    transforms.Resize(resize_img),
            #    transforms.CenterCrop(img_size),
            #    transforms.ToTensor(),
            #    normalize])

            # for vote
            transform_test = transforms.Compose([
                transforms.Resize(resize_img),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
                transforms.RandomRotation([-180, 180]),
                transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
                transforms.RandomCrop(img_size),
                transforms.ToTensor(),
                normalize])

            testset = DatasetFolder(train=False, transform=transform_test, iterNo=int(self.args.iterNo), data_dir=self.args.data_dir, for_vote=self.args.for_vote)

            testloader = torch.utils.data.DataLoader(testset,
                                                 #batch_size=self.args.batch_size,
                                                 batch_size=30,
                                                 num_workers=50,
                                                 shuffle=False)
        else:
            testloader = []
            testset = []

        return trainloader, testloader, len(trainset), len(testset)


    def iterate_CNN(self):
        best_test_mca = 0.0 if self.best_pred == None else self.best_pred
        correct_best = None
        predicted_best = None
        tr_loss_arr = []
        train_mca = []
        test_mca = []

        output_writer_path = os.path.join('./run', self.args.logfile)
        writer = SummaryWriter(output_writer_path)

        for epoch in range(self.args.start_epoch, self.args.n_epochs):
            for index, params in enumerate(self.optimizer.state_dict()['param_groups']):
                writer.add_scalar('train/lr_' + str(index+1), params['lr'], epoch)

            #self.scheduler.step()
            train_loss, accTr, mcaTr, class_precision_train = self.train(epoch)
            if epoch %2 ==0:
                if self.args.use_all_data == False:
                    if os.path.exists(self.args.train_dir) == False:
                        os.mkdir(self.args.train_dir)
                    test_loss, accTe, mcaTe, class_precision_test, correct, predicted = self.test(epoch)
                    #test_loss, accTe, mcaTe, class_precision_test, correct, predicted = self.test_vote(epoch)

                    if mcaTe > best_test_mca:
                        best_test_mca = mcaTe
                        is_best = True
                        correct_best = correct
                        predicted_best = predicted
                    else:
                        is_best = False

                    path = os.path.join(self.args.train_dir, str(epoch))
                    self.save_checkpoint({
                        'epoch': epoch+1,
                        'arch': self.args,
                        'state_dict':self.net.state_dict(),
                        'best_pred': best_test_mca,
                        'optimizer': self.optimizer.state_dict()
                        }, is_best, path)
                    #torch.save(self.net, path)

                    writer.add_scalar('test/acc', accTe, epoch)
                    writer.add_scalar('test/mca', mcaTe, epoch)
                    test_mca.append((mcaTe, class_precision_test))
                else:
                    if os.path.exists(self.args.train_dir) == False:
                        os.mkdir(self.args.train_dir)

                    path = os.path.join(self.args.train_dir, str(epoch))
                    is_best = False
                    self.save_checkpoint({
                        'epoch': epoch+1,
                        'arch': self.args,
                        'state_dict':self.net.state_dict(),
                        'best_pred': best_test_mca,
                        'optimizer': self.optimizer.state_dict()
                        }, is_best, path)
                    test_loss, accTe, mcaTe = 0,0,0
            else:
                test_loss, accTe, mcaTe = 0,0,0

            tr_loss_arr.append([train_loss, accTr, mcaTr, test_loss, accTe, mcaTe])

            train_mca.append((mcaTr, class_precision_train))
            print('Epoch %d %.4f %.4f' % (epoch, mcaTr, mcaTe))

            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/mca', mcaTr, epoch)
            writer.add_scalar('train/acc', accTr, epoch)

        if self.args.use_all_data == False:
            np.save(self.args.logfile, [train_mca, test_mca, class_precision_train, class_precision_test, correct_best, predicted_best])

    def save_checkpoint(self, state, is_best, filename):
        torch.save(state, filename)
        if is_best:
            dirname = os.path.dirname(filename)
            shutil.copyfile(filename, os.path.join(dirname, 'model_best.pth.tar'))
