import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.distributions import normal
import gym
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    def reward(self, exps, b, exp_decay):
        # given experiences (list of trajectories)
        # compute the reward function
        # each experience: [(s,a,p(a|s),r)]
        # b: baseline   Here use naive baseline
        # bias the Reward to prefer nearer samples (which is closer to policy now)
        # by weighted average
        # weight = f(n) should be monotonically increasing
        # here try f(n)=n^alpha
        sum_exps = 0.
        weight_sum = 0.
        for exp_i in range(len(exps)):
            exp = exps[exp_i]
            w = float((exp_i+1) ** exp_decay)
            weight_sum += w
            R = 0.
            sum_exp = 0.
            bt = b / len(exp)
            # compute is for entire trajectory
            log_is = 0.
            for t in range(len(exp)):
                (s,a,r,p) = exp[t]
                log_is += torch.log(self(s)[a])-torch.log(p)
            log_is = log_is.detach()
            for t in range(len(exp)-1,-1,-1):
                (s,a,r,p) = exp[t]
                R += r - bt
                sum_exp += torch.log(self(s)[a])*R*torch.exp(log_is)    # IS from 1 to t, so late update
                log_is -= (torch.log(self(s)[a])-torch.log(p))
                log_is = log_is.detach()    # the predicted prob is treated as constant
            sum_exps += sum_exp*w
        sum_exps = sum_exps / weight_sum
        #sum_exps = sum_exps / len(exps)
        return sum_exps

def main(args):
    env = gym.make(args.env)
    #env = gym.make('CartPole-v0')
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
    exps = []  # accumulate this for training
    # use average reward for baseline
    # we can use a weighted average to estimate b as latest ones are more closer to current policy
    # use the equation:
    # b <- alpha * R + (1-alpha) * b, where alpha should be small relative to 1-alpha
    # 1-alpha is the decay rate
    alpha = args.alpha        # effective length: 1/alpha
    b = 0.   # we can use a weighted average to estimate b as latest ones are more closer to current policy

    i_episode = 0
    reward_list = []
    while i_episode < args.max_epi:
        i_episode += 1
        obs = env.reset()
        # collect data using current policy
        # estimate the average reward on the go
        R = 0.
        exp = []
        for i in range(args.max_iter):
            obs = torch.FloatTensor(obs)
            p_actions = policyNet(obs)
            action_idx = torch.multinomial(p_actions, 1)  # LongTensor
            action = action_mapping[action_idx.item()]
            obs_next, reward, done, info = env.step(action)
            R += reward
            obs = obs.detach()
            action_idx = action_idx.detach()
            p_actions = p_actions.detach()
            exp.append( (obs, action_idx, reward, p_actions[action_idx]) )
            obs = obs_next
            if done:
                break
        reward_list.append(R)
        exps.append(exp)
        print('reward for episode %d: %f' % (i_episode, R))
        # here we use R as initialization for practical reason
        if i_episode == 1:
            b = R
        else:
            b = alpha * R + (1-alpha) * b
        print('baseline: %f' % (b))
        policyNet.zero_grad()
        # when episode 1 b will be equal to R, which leads to 0 J, not good
        if i_episode == 1:
            J = -policyNet.reward(exps, 0., args.exp_decay)
        else:
            J = -policyNet.reward(exps, b, args.exp_decay)
        print(J)
        J.backward()
        policyNet.opt.step()
    return reward_list


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CartPole-v0')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--exp_decay', type=float, default=0.5)
parser.add_argument('--max_epi', type=int, default=200)
parser.add_argument('--max_iter', type=int, default=200)
args = parser.parse_args()



for alpha in [1e-2, 1e-1, 0.25, 0.5, 0.75]:
    for exp_decay in [0., 0.25, 0.5, 0.75, 1.]:
        args.alpha = alpha
        args.exp_decay = exp_decay
        fig = plt.figure()
        reward_list = main(args)
        plt.plot(np.arange(1, args.max_epi+1), reward_list)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        fig.savefig('alpha_%f_expdecay_%f.png' % (alpha, exp_decay))
