"""
This implements the CNN part of the deep network
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
class ResBlock(nn.Module):
    # reference:
    # https://github.com/kuangliu/pytorch-cifar/blob/master/models/googlenet.py
    # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    def __init__(self, in_channel, out_channel, kernel=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, \
                               kernel_size=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.a = nn.ReLU()
        self.downsample_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, \
                                         kernel_size=1, stride=1)
        self.downsample_bn = nn.BatchNorm2d(out_channel)
        # make sure res is same size as after conv
        self.maxpool2d = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.xavier_normal_(self.downsample_conv.weight)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.a(out)
        identity = self.downsample_conv(identity)
        identity = self.maxpool2d(identity)
        identity = self.downsample_bn(identity)
        out += identity
        return out

class ResNet(nn.Module):
    def __init__(self, out_size):
        super(ResNet, self).__init__()
        self.conv1 = ResBlock(in_channel=3, out_channel=16, kernel=5, stride=1, padding=0)
        self.conv2 = ResBlock(in_channel=16, out_channel=32, kernel=5, stride=1, padding=0)
        self.conv3 = ResBlock(in_channel=32, out_channel=64, kernel=5, stride=1, padding=0)
        self.conv4 = ResBlock(in_channel=64, out_channel=32, kernel=3, stride=1, padding=0)
        self.conv5 = ResBlock(in_channel=32, out_channel=16, kernel=3, stride=1, padding=0)
        self.conv6 = ResBlock(in_channel=16, out_channel=16, kernel=3, stride=1, padding=0)
        self.conv7 = ResBlock(in_channel=16, out_channel=16, kernel=3, stride=1, padding=0)
        self.conv8 = ResBlock(in_channel=16, out_channel=16, kernel=3, stride=1, padding=0)
        self.conv9 = ResBlock(in_channel=16, out_channel=16, kernel=3, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=82944, out_features=128)
        self.fc_bn1 = nn.BatchNorm1d(128)
        self.fc_a1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=out_size)
        self.fc_bn2 = nn.BatchNorm1d(out_size)
        self.fc_a2 = nn.ReLU()
    def forward(self, x):
        # x of size: B * S * W * H
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        x = x.view(len(x), -1)
        #print(x.size())
        x = self.fc1(x)
        if len(x) > 1:
            x = self.fc_bn1(x)
        x = self.fc_a1(x)
        x = self.fc2(x)
        if len(x) > 1:
            x = self.fc_bn2(x)
        x = self.fc_a2(x)
        return x
