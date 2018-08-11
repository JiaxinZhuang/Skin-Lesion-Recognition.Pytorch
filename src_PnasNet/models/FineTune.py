import torch.nn as nn
import torch.nn.functional as F

class FineTuneModel(nn.Module):
    """ Wrap ptrtrain model to do finetuning.

    Attributes:
            forward:
    """

    def __init__(self, original_model, arch, args):
        """init entire train model"""

        super(FineTuneModel, self).__init__()
        self.args = args

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

        elif arch == 'resnet50' or arch == 'resnet152':
            # Everything except the last linear layer
            self.features = original_model
            self.bn_c = nn.BatchNorm1d(1000)
            self.classifier = nn.Linear(1000, 7)

        elif arch == 'densenet161':
            # Everything except the last linear layer
            self.features = original_model

            self.bn_c = nn.BatchNorm1d(1000)
            self.classifier = nn.Sequential(
                nn.Linear(1000, 7)
            )

        elif arch == 'pnasnet5large' or arch == 'nasnetalarge':
            # Everything except the last linear layer
            self.features = original_model

            self.bn_c = nn.BatchNorm1d(1000)
            self.classifier = nn.Linear(1000, 7)
        else :
            raise("Finetuning not supported on this architecture yet")

        self.modelName = arch
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
        if self.args.extract_feature == True:
            y = self.features(x)
        else:
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
