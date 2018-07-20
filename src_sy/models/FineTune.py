import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch):
        super(FineTuneModel, self).__init__()

        if arch == 'resnet152_3c' or arch =='resnet50_3c':
            # 3 conv layer
            original_model.avgpool = nn.Sequential(
                nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=0,
                               bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=0,
                               bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=0,
                               bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
            )
            self.features = original_model
            self.classifier = nn.Sequential(
                nn.Linear(1000, 7)
            )
            self.modelName = 'resnet152_3c'
        elif arch == 'resnet50' or arch == 'resnet152':
            # Everything except the last linear layer
            self.features = original_model
            self.bn_c = nn.BatchNorm1d(1000)
            self.classifier = nn.Linear(1000, 7)
            self.modelName = 'resnet'
        elif arch == 'inceptionresnetv2':
            # Everything except the last linear layer
            self.features = original_model
            self.classifier = nn.Sequential(
                nn.Linear(1000, 7)
            )
            self.modelName = 'inceptionresnetv2'
        elif arch == 'densenet161':
            # Everything except the last linear layer
            self.features = original_model

            # test
            self.bn_c = nn.BatchNorm1d(1000)

            self.classifier = nn.Sequential(
                nn.Linear(1000, 7)
            )
            self.modelName = 'densenet161'
        elif arch == 'densenet161_m':
            # remove linear layer
            #self.features = original_model.features
            self.classifier = nn.Linear(2208, 7)

            #
            self.modelName = 'densenet161_m'
        elif arch == 'pnasnet5large' or arch == 'nasnetalarge':
            # Everything except the last linear layer
            self.features = original_model

            # test
            self.bn_c = nn.BatchNorm1d(1000)

            self.classifier = nn.Linear(1000, 7)
            self.modelName = arch
        else :
            raise("Finetuning not supported on this architecture yet")

        print('FineTuneModel is {}'.format(self.modelName))

        # Freeze those weights after train some times
        #freeze_cnt = 12-2
        #freeze_cnt = 12-4
        #freeze_cnt = 12-6
        #freeze_cnt = 12-8

        #freeze_cnt = 12-3
        #freeze_cnt = 12-5
        #for index, (name, p) in enumerate(chain(self.features.features.named_children(), self.features.classifier.named_children())):
        #    if index < freeze_cnt:
        #        print('=> Freeze {} layer'.format(name))
        #        p.require_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(F.relu(self.bn_c(f)))

        # remove linear layer
        #features = self.features(x)
        #out = F.relu(features, inplace=True)
        #out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        #y = self.classifier(out)

        # use temperature for softmax instead of 1, we try 50, 100
        #temperature = 50
        #temperature = 100
        #print('=> Using temperature for softmax {}'.format(temperature))
        #y = torch.div(y, temperature)

        return y
