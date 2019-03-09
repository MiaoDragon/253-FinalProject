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
class BaselineNet(nn.Module):
    def __init__(self, state_dim, action_dim, std=1.):
        super(BaselineNet, self).__init__()
        # cnn layer for state extraction
        self.cnn = ResNet(state_dim)
        # three layer MLP
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.opt = optim.Adam(self.parameters(), lr=1e-3)
        self.std = std
    def forward(self, s):
        #s = self.cnn(s)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = self.fc3(s)  # hamiltonina
        #s = F.softmax(s)
        return s
    def update_std(self, std):
        self.std = std
    def explore(self, s):
        # add stochastic for exploration
        a = self(s)
        dist = torch.distributions.normal.Normal(a, self.std)
        return dist.sample()
    def log_prob(self, s, a):
        # given state and action, output the prob of choosing that action
        mean = self(s).squeeze()
        # we use std=1 for simplicity
        # mean: B * Action_shape
        # a: B * action
        dist = torch.distributions.normal.Normal(mean, self.std)
        return dist.log_prob(a)
    def set_opt(self, opt=optim.Adam):
        self.opt = opt(self.parameters(), lr=1e-3)
