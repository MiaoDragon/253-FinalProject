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
class QValueNet(nn.Module):
    # used for estimating the Q value
    def __init__(self, obs_num, state_dim, action_dim, lower, upper, use_cnn=True):
        super(QValueNet, self).__init__()
        # cnn layer for state extraction
        self.use_cnn = use_cnn
        if self.use_cnn:
            self.cnn = ResNet(in_channel=obs_num, out_size=state_dim)
        else:
            self.cnn = None
        # three layer MLP
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.opt = optim.Adam(self.parameters(), lr=1e-3)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.action_dim = action_dim
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
