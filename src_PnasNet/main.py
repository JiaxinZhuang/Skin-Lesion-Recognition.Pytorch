# Copyright 2018 The SYSU-ISEE. All Rights Reserved.
"""Main file to train and test.
"""

import argparse
import torch

import sys
sys.path.append('./models/')
from models import CNN_Train
from models import FineTuneModel

# Models
import torchvision
import pretrainedmodels


# auxiliary function
def str2bool(v):
    """Convert string to Boolean

    Args:
        v: True or False but in string

    Returns:
        True or False in Boolean

    Raises:
        TyreError
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser(description='Net')
# !! Must provide when running main
parser.add_argument('--model_path', default='',
                    help='select model to retrain or generate test result')
parser.add_argument('--train_dir', default='../train_dir',
                    help='save model dir')
parser.add_argument('--logfile', default='result',
                    help='save log dir')
parser.add_argument('--c', default='0',
                    help='cuda device')

parser.add_argument('--data_dir', default='../data/ISIC2018/')

# FOR TEST
parser.add_argument('--use_all_data', type=str2bool, default=False,
                    help='use all train data so no validation')
parser.add_argument('--for_vote', default=False, type=str2bool)
parser.add_argument('--lm', default=False, type=str2bool,
                    help='load from trained_model')


# FOR TRAIN
parser.add_argument('--train', default=True, type=str2bool)
parser.add_argument('--resize_img', default=300, type=int)
parser.add_argument('--model',default='resnet50')
parser.add_argument('--lr',default=0.01/100)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=250)
parser.add_argument('--nclass', type=int, default=7)
parser.add_argument('--desc', default='Densenet')
parser.add_argument('--iterNo', default='0',
                     help='split data into 5-fold and select the iterNo fold to evaluate')
parser.add_argument('--GPU_ids', default=0)

# mode: extract features
parser.add_argument('--extract_feature', default=False, type=str2bool)
# all data directory including train, test, validation and split_data
parser.add_argument('--prediction', default='../predictions/')

args = parser.parse_args()


import os
os.environ["CUDA_VISIBLE_DEVICES"]=args.c


def main():
    """main Module which select models and load.
    """

    model = args.model
    # Use args.model as pretrain model
    if model == 'resnet152':
        original_model = torchvision.models.resnet152(pretrained=True)
    elif model == 'resnet50' or model == 'resnet50_3c':
        original_model = torchvision.models.resnet50(pretrained=True)
    elif model == 'densenet161':
        original_model = torchvision.models.densenet161(pretrained=True)
    elif model == 'pnasnet5large':
        original_model = pretrainedmodels.pnasnet5large(num_classes=1000, pretrained='imagenet')
    elif model == 'nasnetalarge':
        original_model = pretrainedmodels.nasnetalarge(num_classes=1000, pretrained='imagenet')
    else:
        sys.exit(-1)

    net = FineTuneModel(original_model, model, args)

    #for name, param in original_model.named_children():
    #   print(name)
    #for name, param in net.named_children():
    #   print(name)

    if args.lm == False:
        CNN_Train(net, args)
    elif args.lm == True:
        # Load from saved model
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
