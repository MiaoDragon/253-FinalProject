"""
This implements the CNN part of the deep network
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from resnet import resnet18


class ResNet(nn.Module):
    def __init__(self, in_channel, out_size):
        super(ResNet, self).__init__()
        self.model = resnet18(num_classes=out_size)
        self.model.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    def forward(self, x):
        x = self.model(x)
        return x

class CNN(nn.Module):
    def __init__(self, in_channel, out_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=5, stride=2,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, out_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
