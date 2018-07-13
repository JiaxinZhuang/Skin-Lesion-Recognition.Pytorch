import torch
import torch.nn as nn
import torch.nn.functional as F

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
        elif arch.startswith('inceptionresnetv2'):
            # Everything except the last linear layer
            self.features = original_model
            self.classifier = nn.Sequential(
                nn.Linear(1000, 7)
            )
            self.modelName = 'inceptionresnetv2'
        elif arch.startswith('densenet161'):
            # Everything except the last linear layer
            self.features = original_model
            self.classifier = nn.Sequential(
                nn.Linear(1000, 7)
            )
            self.modelName = 'densenet161'
        else :
            raise("Finetuning not supported on this architecture yet")

        print('FineTuneModel is {}'.format(self.modelName))

        # Freeze those weights
        #for p in self.features.parameters():
        #    p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(F.relu(self.bn_c(f)))
        return y
