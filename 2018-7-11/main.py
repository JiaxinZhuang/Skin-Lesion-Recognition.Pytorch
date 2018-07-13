import argparse
import torch.nn as nn
import sys
sys.path.append('./models/')
from models import *
import torchvision

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
parser.add_argument('--desc', default='Resnet')
parser.add_argument('--iterNo', default=4)
args = parser.parse_args()

model = 'resnet50'
if model == 'resnet152':
	original_model = torchvision.models.resnet152(pretrained=True)
elif model == 'resnet50':
	original_model = torchvision.models.resnet50(pretrained=True)
	
for name, param in original_model.named_children():
   print(name)
net = FineTuneModel(original_model, model)
for name, param in net.named_children():
   print(name)

#num_ftrs = net.fc.in_features
#net.fc = nn.Linear(num_ftrs, nclass)

CNN_Train(net, args)


