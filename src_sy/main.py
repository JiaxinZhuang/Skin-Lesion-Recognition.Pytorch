import argparse
import torch.nn as nn
import sys
sys.path.append('./models/')
from models import *
import torchvision

import sys

# parameters
lr = 0.01/100
batch_size = 30
n_epochs = 500
GPU_ids = 0
nclass = 7


parser = argparse.ArgumentParser(description='Densenet')
parser.add_argument('--lr', default=lr)
parser.add_argument('--batch_size', default=batch_size)
parser.add_argument('--n_epochs', default=n_epochs)
parser.add_argument('--GPU_ids', default=GPU_ids)
parser.add_argument('--nclass', default=nclass)
parser.add_argument('--desc', default='Densenet')
# arguement model
parser.add_argument('--model', default='resnet50')
# color constancy
parser.add_argument('--data_dir', default='/home/jiaxin/myGithub/Reverse_CISI_Classification/data/ISIC2018/ISIC2018_Task3_Training_Input/')
# cuda
parser.add_argument('--cuda_visible', default='0')
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_visible

#model = 'resnet50'
#model = 'resnet152'
#model = 'inception_v3'
model = args.model
if model == 'resnet152':
    original_model = torchvision.models.resnet152(pretrained=True)
elif model == 'resnet152_c':
    original_model = torchvision.models.resnet152(pretrained=True)
    # use conv instead of avg
    original_model.avgpool = nn.Conv2d(2048, 2048, kernel_size=7, stride=1, bias=False)
elif model == 'resnet50':
    original_model = torchvision.models.resnet50(pretrained=True)
elif model == 'inception_v3':
    original_model = torchvision.models.inception_v3(pretrained=True)
elif model == 'densenet121':
    original_model = torchvision.models.densenet161(pretrained=True)
elif model == 'vgg16':
    original_model = torchvision.models.vgg16(pretrained=True)
elif model == 'squeezenet':
    original_model = torchvision.models.squeezenet1_0(pretrained=True)
else:
    sys.exit(-1)

for name, param in original_model.named_children():
   print(name)
net = FineTuneModel(original_model, model)
for name, param in net.named_children():
   print(name)

#num_ftrs = net.fc.in_features
#net.fc = nn.Linear(num_ftrs, nclass)

CNN_Train(net, args)


