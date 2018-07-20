import argparse
import torch.nn as nn
import sys
sys.path.append('./models/')
from models import *
import torchvision
import pretrainedmodels

import sys

# auxiliary function
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# parameters
lr = 0.01/100
batch_size = 12
#batch_size = 8
#batch_size = 16
n_epochs = 250
GPU_ids = 0
nclass = 7
model = 'resnet50'
iterNo = '0'

# data dir including input and split data
data_dir = '../data/ISIC2018/'
train_dir = '../train_dir'

parser = argparse.ArgumentParser(description='Densenet')

# !! Must provide when running main
parser.add_argument('--model_path', default='')
parser.add_argument('--train_dir', default=train_dir)
parser.add_argument('--logfile', default='result')
parser.add_argument('--c', default='0')

parser.add_argument('--model',default=model)
parser.add_argument('--lr',default=lr)
parser.add_argument('--batch_size', type=int, default=batch_size)
parser.add_argument('--start_epoch', default=0)
parser.add_argument('--n_epochs', type=int, default=n_epochs)
parser.add_argument('--GPU_ids', default=GPU_ids)
parser.add_argument('--nclass', default=nclass)
parser.add_argument('--desc', default='Densenet')
parser.add_argument('--iterNo', default='4')

# load from trained_model
parser.add_argument('--lm', default=False, type=str2bool)
parser.add_argument('--train', default=True, type=str2bool)
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
    elif model == 'densenet161_m':
        original_model = torchvision.models.densenet161(pretrained=True)
    elif model == 'vgg16':
        original_model = torchvision.models.vgg16(pretrained=True)
    elif model == 'squeezenet':
        original_model = torchvision.models.squeezenet1_0(pretrained=True)
    elif model == 'pnasnet5large':
        original_model = pretrainedmodels.pnasnet5large(num_classes=1000, pretrained='imagenet')
    elif model == 'nasnetalarge':
        original_model = pretrainedmodels.nasnetalarge(num_classes=1000, pretrained='imagenet')
    else:
        sys.exit(-1)

    net = FineTuneModel(original_model, model)


    #for name, param in original_model.named_children():
    #   print(name)
    #for name, param in net.named_children():
    #   print(name)

    if args.lm == False:
        CNN_Train(net, args)
    elif args.lm == True:
        if os.path.isfile(args.model_path) == True:
            print("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            CNN_Train(net, args, checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.model_path))

    else:
        print('Error --lm parameter')
        sys.exit(-1)


if __name__ == '__main__':
    main()


