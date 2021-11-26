import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def getImageNet():
    NetModel = models.densenet121(pretrained=True)
    # print(NetModel.classifier)

    for param in NetModel.parameters():
        param.requires_grad = False
        # print(param.shape)

    classifier = nn.Sequential(OrderedDict([
                            ('Linear1_wbq', nn.Linear(1024, 512)),
                            ('relu', nn.ReLU()),
                            ('relu', nn.Dropout2d()),
                            ('Linear2_wbq', nn.Linear(512, 200)),
                            ]))

    NetModel.classifier = classifier

    # fc_inputs = NetModel.fc.in_features
    # NetModel.fc = nn.Sequential(
    #     nn.Linear(fc_inputs, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, 200),
    # )
    # print(denseNetModel.classifier)

    return NetModel


if __name__ == '__main__' : 
    getImageNet()