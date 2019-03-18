"""
Original Policy Gradient only applies to discrete space
Here we give a policy network for continuous space
We achieve this by outputing the mean of a Gaussian distribution
For the baseline case, we simply use N(mu,I) as the gaussian model
"""

import torch
import torch.distributions as distrib
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from cnn import ResNet


# We should probably use inheritance, especially if this gets more complicated
# later on.
def gaussian_distrib(self, s):
    return distrib.normal.Normal(loc=self(s), scale=self.std)

def beta_distrib(self, s):
    s = self(s)
    alpha = s[..., 0]
    beta = s[..., 1]
    transforms = [distrib.AffineTransform(loc=self.lower,
                                          scale= self.upper - self.lower)]
    return distrib.TransformedDistribution(distrib.beta.Beta(alpha, beta),
                                      transforms)

distributions = {
    'gaussian':(1, gaussian_distrib),
    'beta':(2, beta_distrib)
}

class BaselineNet(nn.Module):
    def __init__(self, obs_num, state_dim, action_dim, lower, upper, use_cnn=True,
               distribution='gaussian'):
        super(BaselineNet, self).__init__()

        self.dist = distribution
        self.param_count, self.distribution = distributions[distribution]
        # bind ("fake method")
        self.distribution = lambda s: self.distribution(self, s)
        # this is only for gaussians right now
        self.std = 1.

        # cnn layer for state extraction
        self.use_cnn = use_cnn
        if self.use_cnn:
            self.cnn = ResNet(in_channel=obs_num, out_size=state_dim)
        else:
            self.cnn = None
        # three layer MLP
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, param_count*action_dim)
        self.opt = optim.Adam(self.parameters(), lr=1e-3)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.action_dim = action_dim
        self.lower = lower
        self.upper = upper

    def set_std(self, std):
        self.std = std

    def forward(self, s):
        if self.use_cnn:
            s = self.cnn(s)
        s = s.view(len(s), -1)  # concatenate obs in Pandulum example
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = self.fc3(s)

        # Reshape for convenience
        if self.param_count > 1:
            s = s.view(len(s), self.action_dim, param_count)
        else:
            s = s.view(len(s), self.action_dim)

        # ?
        if self.dist == 'beta':
            s = self.sigmoid(s) * 50 + 2

        return s

    def explore(self, s):
        dist = self.distribution(s)
        action = dist.sample()
        prob = dist.log_prob(action)
        return action, prob

    def log_prob(self, s, a):
        dist = self.distribution(s)
        return dist.log_prob(a)

    def set_opt(self, opt=optim.Adam, lr=1e-2):
        self.opt = opt(self.parameters(), lr=lr)
