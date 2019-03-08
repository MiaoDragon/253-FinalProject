import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.distributions import normal
import gym
class PolicyNetDiscreteVanila(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetDiscreteVanila, self).__init__()
        # three layer MLP
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.opt = optim.Adam(self.parameters(), lr=1e-3)
    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = self.fc3(s)  # hamiltonina
        s = F.softmax(s)
        return s # probability distribution
    def reward(self, exps, b):
        # given experiences (list of trajectories)
        # compute the reward function
        # each experience: [(s,a,r)]
        # b: baseline
        sum_exps = 0.
        for exp in exps:
            R = 0.
            sum_exp = 0.
            bt = b / len(exp)
            for t in range(len(exp)-1,-1,-1):
                (s,a,r) = exp[t]
                R += r - bt
                sum_exp += torch.log(self(s)[a])*R
            sum_exps += sum_exp
        sum_exps = sum_exps / len(exps)
        return sum_exps


env = gym.make('Pendulum-v0')
# here if action space is continuous, we try to discretize it
action_mapping = []
if hasattr(env.action_space, 'n'):
    policyNet = PolicyNetDiscreteVanila(env.observation_space.shape[0], env.action_space.n)
    action_mapping = range(env.action_space.n)
else:
    # use a mapping
    # here assume action space is only one dimension
    action_num = 10
    discrete_epi = (env.action_space.high[0] - env.action_space.low[0]) / action_num
    action_mapping = [[i*discrete_epi+env.action_space.low[0]] for i in range(action_num)]
    policyNet = PolicyNetDiscreteVanila(env.observation_space.shape[0], action_num)
step = 0
num_exp = 100
while True:
    step += 1
    env.reset()
    # collect data using current policy
    # estimate the average reward on the go
    b = 0.
    exps = []
    #for i_episode in range(100):
    for i_episode in range(num_exp):
        exp = []
        obs = env.reset()
        for i in range(200):
            env.render()
            obs = torch.FloatTensor(obs)
            p_actions = policyNet(obs)
            action_idx = torch.multinomial(p_actions, 1)  # LongTensor
            action = action_mapping[action_idx.item()]
            obs_next, reward, done, info = env.step(action)
            b += reward
            # append it into experience
            obs = obs.detach()
            action_idx = action_idx.detach()
            exp.append( (obs, action_idx, reward) )
            obs = obs_next
            if done:
                break
        exps.append(exp)
    b = b / num_exp
    print('average reward for step %d: %f' % (step-1, b))
    # train network
    policyNet.zero_grad()
    # problem of really long horizon
    J = -policyNet.reward(exps, b)
    J.backward()
    policyNet.opt.step()
