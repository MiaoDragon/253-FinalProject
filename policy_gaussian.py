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
    def __init__(self, obs_num, state_dim, action_dim, lower, upper, use_cnn=True):
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
        self.std = 1.
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.action_dim = action_dim
        #self.transforms = [torch.distributions.AffineTransform(loc=lower, scale=upper-lower)]
        self.transforms = transforms = [torch.distributions.SigmoidTransform(), \
                                        torch.distributions.AffineTransform(loc=lower, scale=upper-lower)]
    def forward(self, s):
        if self.use_cnn:
            s = self.cnn(s)
        s = s.view(len(s), -1)  # concatenate obs in Pandulum example
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = self.fc3(s)  # hamiltonina
        #s = self.tanh(s) * 2  # -2 to 2
        s = s.view(len(s), self.action_dim)
        #s = F.softmax(s)
        # this to make sure the output is larger than 1
        #s = self.softplus(s) + 1e-5
        #s = self.sigmoid(s) * 50 + 2
        #print(s)
        return s
    def set_std(self, std):
        self.std = std
    def distribution(self, s):
        # return the distribution obtained from input
        s = self(s)
        alpha = s[...,0]
        beta = s[...,1]
        # added softplus
        dist = torch.distributions.normal.Normal(loc=s, scale=self.std)
        #dist = torch.distributions.transformed_distribution.TransformedDistribution(dist, self.transforms)
        return dist
    def explore(self, s):
        # add stochastic for exploration
        s = self(s)
        print('mean:')
        print(s)
        # added softplus
        dist = torch.distributions.normal.Normal(loc=s, scale=self.std)
        action = dist.sample()
        prob = dist.log_prob(action)
        return action, prob
        
    def log_prob(self, s, a):
        # given state and action, output the prob of choosing that action
        s = self(s)
        # we use std=1 for simplicity
        # mean: B * Action_shape
        # a: B * action
        dist = torch.distributions.normal.Normal(loc=s, scale=self.std)
        #dist = torch.distributions.transformed_distribution.TransformedDistribution(dist, self.transforms)
        return dist.log_prob(a)
    def set_opt(self, opt=optim.Adam, lr=1e-2):
        self.opt = opt(self.parameters(), lr=lr)
