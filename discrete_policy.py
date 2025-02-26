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
    def __init__(self, obs_num, state_dim, action_dim, use_cnn=True):
        super(BaselineNet, self).__init__()
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
        self.softmax = nn.Softmax()
        self.action_dim = action_dim
    def forward(self, s):
        if self.use_cnn:
            s = self.cnn(s)
        s = s.view(len(s), -1)  # concatenate obs in Pandulum example
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = self.fc3(s)  # hamiltonina
        #s = F.softmax(s)
        # this to make sure the output is larger than 1
        s = self.softmax(s)
        #print(s)
        return s

    def explore(self, s):
        # add stochastic for exploration
        s = self(s)
        return torch.multinomial(s, len(s))
    def log_prob(self, s, a):
        # given state and action, output the prob of choosing that action
        pred = self(s)
        #print('probability:')
        #print(pred)
        #print('action:')
        #print(a)
        #print('selected prob:')
        #print(torch.gather(pred, 1, a))
        return torch.log(torch.gather(pred, 1, a))

    def set_opt(self, opt=optim.Adam, lr=1e-2):
        self.opt = opt(self.parameters(), lr=lr)
