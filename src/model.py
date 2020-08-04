"""Model.

    Model should be formulated here.
"""

import sys
import torch
import torch.nn as nn
import torchvision
from torchsummary import summary

import pretrainedmodels


# from utils.initialization import _kaiming_normal, _xavier_normal, \
#         _kaiming_uniform, _xavier_uniform


class Network(nn.Module):
    """Network
    """
    def __init__(self, backbone="resnet50", num_classes=7, input_channel=3,
                 pretrained=True):
        super(Network, self).__init__()
        if backbone == "resnet50":
            model = ResNet50(num_classes=num_classes,
                             input_channel=input_channel,
                             pretrained=pretrained)
        elif backbone == "resnet18":
            model = ResNet18(num_classes=num_classes,
                             input_channel=input_channel,
                             pretrained=pretrained)
        elif backbone == "PNASNet5Large":
            model = PNASNet5Large(num_classes=num_classes,
                                  input_channel=input_channel,
                                  pretrained=pretrained)
        elif backbone == "NASNetALarge":
            model = NASNetALarge(num_classes=num_classes,
                                 input_channel=input_channel,
                                 pretrained=pretrained)
        else:
            print("Need model")
            sys.exit(-1)
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)

    def print_model(self, input_size, device):
        """Print model structure
        """
        self.model.to(device)
        summary(self.model, input_size)


class ResNet50(nn.Module):
    """AlexNet
    """
    def __init__(self, num_classes, input_channel, pretrained):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(
            *list(torchvision.models.resnet50(pretrained=pretrained).
                  children())[:-1]
            )
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet18(nn.Module):
    """AlexNet
    """
    def __init__(self, num_classes, input_channel, pretrained):
        super(ResNet18, self).__init__()
        self.features = nn.Sequential(
            *list(torchvision.models.resnet18(pretrained=pretrained).
                  children())[:-1]
            )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class PNASNet5Large(nn.Module):
    """PNASNet5Large.
    """
    def __init__(self, num_classes, input_channel, pretrained):
        super(PNASNet5Large, self).__init__()
        model = pretrainedmodels.pnasnet5large(num_classes=1000,
                                               pretrained="imagenet")
        model.last_linear = nn.Linear(model.last_linear.in_features,
                                      num_classes)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out


class NASNetALarge(nn.Module):
    """NASNetALarge.
    """
    def __init__(self, num_classes, input_channel, pretrained):
        super(NASNetALarge, self).__init__()
        model = pretrainedmodels.nasnetalarge(num_classes=1000,
                                              pretrained="imagenet")
        model.last_linear = nn.Linear(model.last_linear.in_features,
                                      num_classes)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out


class Identity(nn.Module):
    """Identity path.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


# def init_weights(net, method, _print):
#    """Initialize weights of the networks.
#
#        weights of conv layers and fully connected layers are both initialzied
#        with Xavier algorithm. In particular, set parameters to random values
#        uniformly drawn from [-a, a], where a = sqrt(6 * (din + dout)), for
#        batch normalization layers, y=1, b=0, all biases initialized to 0
#    """
#    if method == "kaiming_normal":
#        net = _kaiming_normal(net, _print)
#    elif method == "kaiming_uniform":
#        net = _kaiming_uniform(net, _print)
#    elif method == "xavier_uniform":
#        net = _xavier_uniform(net, _print)
#    elif method == "xavier_normal":
#        net = _xavier_normal(net, _print)
#    else:
#        _print("Init weight: Need legal initialization method")
#    return net


if __name__ == "__main__":
    # net = Network(backnone="alexnet")
    # input_size = (3, 224, 224)
    # net = Network(backbone="resnet50", num_classes=200)
    input_size = (3, 331, 331)
    # net = Network(backbone="PNASNet5Large", num_classes=200)
    net = Network(backbone="NASNetALarge", num_classes=200)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.print_model(input_size, device)
