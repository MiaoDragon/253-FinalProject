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
