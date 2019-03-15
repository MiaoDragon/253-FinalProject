"""
Original Policy Gradient only applies to discrete space
Here we give a policy network for continuous space
We achieve this by outputing the mean of a Gaussian distribution
For the baseline case, we simply use N(mu,I) as the gaussian model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from cnn import ResNet
class ValueNet(nn.Module):
    # used for estimating the value
    def __init__(self, obs_num, state_dim, lower, upper, use_cnn=True):
        super(ValueNet, self).__init__()
        # cnn layer for state extraction
        self.use_cnn = use_cnn
        if self.use_cnn:
            self.cnn = ResNet(in_channel=obs_num, out_size=state_dim)
        else:
            self.cnn = None
        # three layer MLP
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)
        self.opt = optim.Adam(self.parameters(), lr=1e-3)
    def forward(self, s):
        if self.use_cnn:
            s = self.cnn(s)
        s = s.view(len(s), -1)  # concatenate obs in Pandulum example
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = self.fc3(s)
        return s
    def set_opt(self, opt=optim.Adam, lr=1e-2):
        self.opt = opt(self.parameters(), lr=lr)
