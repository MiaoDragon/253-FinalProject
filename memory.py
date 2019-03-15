"""
this implements the memory recording past trajectories
each memory records: (s_t, a_t, r_t, log_p(a_t|s_t), Q)
memory is a circular array of array
Q is None if not calculated, otherwise a floating point value
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import scipy.signal
from rl_utility import *
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    ref: https://github.com/openai/spinningup/blob/master/spinup/algos/vpg/core.py
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Memory():
    def __init__(self, capacity, obs_num, computing_device):
        self.capacity = capacity
        self.idx = 0
        self.obs_num = obs_num
        self.obs_list = []
        self.a_list = []
        self.r_list = []
        self.R_list = []
        self.log_p_list = []
        self.b = 0.
        self.computing_device = computing_device
    def remember(self, obs, a, r, log_p, gamma):
        # given new trajectory in the form [(s_t, a_t, r_t, log_p(a_t|s_t))]
        # add this into memory
        self.idx = (self.idx + 1) % self.capacity
        if len(self.obs_list) < self.capacity:
            self.obs_list.append(obs)
            self.a_list.append(a)
            self.r_list.append(r)
            self.log_p_list.append(log_p)
            #R = discount_cumsum(r, gamma)
            #print(R)
            self.R_list.append(discount_cumsum(r, gamma))
        else:
            self.obs_list[self.idx] = obs
            self.a_list[self.idx] = a
            self.r_list[self.idx] = r
            self.R_list[self.idx] = discount_cumsum(r, gamma)
            self.log_p_list[self.idx] = log_p

    def loss_subsample(self, net):
        # this use sub sample to compute loss
        pass
    def loss_traj(self, net, b, clip_upper, clip_lower):
        # this computes loss along each trajectory
        #self.compute_b()
        #b = self.b
        sum_exps = 0.
        for exp_i in range(len(self.obs_list)):
            R = 0.
            sum_exp = 0.
            bt = b / len(self.obs_list[0])
            # compute is for entire trajectory (is: important sampling)
            log_is = 0.
            states = []
            actions = []
            past_log_p = []
            As = []
            # check if q is None, if so, then calculate q for each exp

            for t in range(len(self.obs_list[exp_i])):
                states.append(obs_to_state(self.obs_num, self.obs_list[exp_i][t], self.obs_list[exp_i][:t])) # :t actually uses past exps
                As.append(self.R_list[exp_i][t] - bt * (len(self.obs_list[0])-t))
            states = torch.stack(states)
            actions = torch.stack(self.a_list[exp_i])
            past_log_p = torch.stack(self.log_p_list[exp_i])
            As = torch.tensor(As)
            states = states.to(self.computing_device)
            actions = actions.to(self.computing_device)
            past_log_p = past_log_p.to(self.computing_device)
            As = As.to(self.computing_device)
            # sum probabiliies to obtain joint probaility
            log_probs = net.log_prob(states, actions).sum(dim=1)
            #log_probs = net.log_prob(states, actions).sum(dim=1)

            log_is = (log_probs - past_log_p).detach()

            log_is[log_is>np.log(10.)] = np.log(10.)  # clipping
            importance_w = torch.exp(log_is)
            # clipping is
            # ref: https://arxiv.org/pdf/1707.06347.pdf
            importance_w[importance_w > 1+clip_upper] = 1+clip_upper
            importance_w[importance_w < 1-clip_lower] = 1-clip_lower
            #print('log probability:')
            #print(log_probs)
            #print('importance weight:')
            #print(importance_w)
            # take the lower bound as loss
            exp_loss = (log_probs * As * importance_w).sum()
            exp_loss = exp_loss / len(self.obs_list[0])  # normalize to give smaller loss
            sum_exps += exp_loss
            # print log_probs, log_is to see where it went wrong
            #print('log_probs:')
            #print(log_probs)
            #print('log_is:')
            #print(log_is)
            # convert reward to loss by inserting -
        return -sum_exps / len(self.obs_list)
