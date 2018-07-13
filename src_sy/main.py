import argparse
import torch.nn as nn
import sys
sys.path.append('./models/')
from models import *
import torchvision
import pretrainedmodels

import sys

# parameters
lr = 0.01/100
batch_size = 30
n_epochs = 501
GPU_ids = 0
nclass = 7
model = 'resnet50'
iterNo = '0'

# data dir including input and split data
data_dir = '../data/ISIC2018/'
train_dir = '../train_dir'

parser = argparse.ArgumentParser(description='Densenet')

# !! Must provide when running main
parser.add_argument('--train_dir', default=train_dir)
parser.add_argument('--logfile', default='result')
parser.add_argument('--c', default='0')

parser.add_argument('--model',default=model)
parser.add_argument('--lr',default=lr)
parser.add_argument('--batch_size', default=batch_size)
parser.add_argument('--n_epochs', default=n_epochs)
parser.add_argument('--GPU_ids', default=GPU_ids)
parser.add_argument('--nclass', default=nclass)
parser.add_argument('--desc', default='Densenet')
parser.add_argument('--iterNo', default='1')

# load from trained_model
parser.add_argument('--lm', default=False)
parser.add_argument('--train', default=True)

# all data directory including train, test, validation and split_data
parser.add_argument('--data_dir', default=data_dir)
parser.add_argument('--prediction', default='../predictions/')

args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"]=args.c

def main():
    model = args.model

    if model == 'resnet152' or model == 'resnet152_3c':
        original_model = torchvision.models.resnet152(pretrained=True)
    elif model == 'resnet50' or model == 'resnet50_3c':
        original_model = torchvision.models.resnet50(pretrained=True)
    elif model == 'inceptionresnetv2':
        original_model = pretrainedmodels.inceptionresnetv2(pretrained='imagenet')
    elif model == 'densenet161':
        original_model = torchvision.models.densenet161(pretrained=True)
    elif model == 'vgg16':
        original_model = torchvision.models.vgg16(pretrained=True)
    elif model == 'squeezenet':
        original_model = torchvision.models.squeezenet1_0(pretrained=True)
    else:
        sys.exit(-1)

    #for name, param in original_model.named_children():
    #   print(name)
    net = FineTuneModel(original_model, model)
    #for name, param in net.named_children():
    #   print(name)

    if args.lm == 'True' and os.path.isfile(args.train_dir) == True:
        the_model = torch.load(args.train_dir)
        #the_model = net(*args, **kwargs)
        print('Restore train model from {}'.format(args.train_dir))
        #the_model.load_state_dict(torch.load(args.train_dir))
        net = the_model

    CNN_Train(net, args)


if __name__ == '__main__':
    main()


