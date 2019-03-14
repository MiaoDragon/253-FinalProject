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
from rl_utility import *
class Memory():
    def __init__(self, capacity, obs_num, computing_device):
        self.capacity = capacity
        self.idx = 0
        self.obs_num = obs_num
        self.memory = []
        self.b = 0.
        self.computing_device = computing_device
    def remember(self, traj):
        # given new trajectory in the form [(s_t, a_t, r_t, log_p(a_t|s_t))]
        # add this into memory
        self.idx = (self.idx + 1) % self.capacity
        if len(self.memory) < self.capacity:
            self.memory.append(traj)
        else:
            self.memory[self.idx] = traj
    def compute_b(self):
        # compute baseline
        b = 0.
        for traj in self.memory:
            for exp in traj:
                b += exp[2]
        self.b = b / len(self.memory)

    def loss_subsample(self, net):
        # this use sub sample to compute loss
        pass
    def loss_traj(self, net, b, clip_factor):
        # this computes loss along each trajectory
        #self.compute_b()
        #b = self.b
        sum_exps = 0.
        for exp_i in range(len(self.memory)):
            exp = self.memory[exp_i]
            R = 0.
            sum_exp = 0.
            bt = b / len(exp)
            # compute is for entire trajectory (is: important sampling)
            log_is = 0.
            states = []
            actions = []
            past_log_p = []
            As = []
            # check if q is None, if so, then calculate q for each exp
            if exp[0][4] is None:
                for t in range(len(exp)-1,-1,-1):
                    o, a, r, log_p, A = exp[t]
                    R += r - bt # advantage value
                    exp[t][4] = R
            for t in range(len(exp)):
                o, a, r, log_p, A = exp[t]
                # stack observations into states
                states.append(obs_to_state(self.obs_num, o, exp[:t])) # :t actually uses past exps
                actions.append(a)
                past_log_p.append(log_p)
                As.append(A)

            states = torch.stack(states)
            actions = torch.stack(actions)
            past_log_p = torch.stack(past_log_p)
            As = torch.tensor(As)
            states = states.to(self.computing_device)
            actions = actions.to(self.computing_device)
            past_log_p = past_log_p.to(self.computing_device)
            As = As.to(self.computing_device)
            # sum probabiliies to obtain joint probaility
            log_probs = net.log_prob(states, actions).sum(dim=1)

            log_is = (log_probs - past_log_p).detach()

            log_is[log_is>np.log(10.)] = np.log(10.)  # clipping
            importance_w = torch.exp(log_is)
            # clipping is
            # ref: https://arxiv.org/pdf/1707.06347.pdf
            importance_w[importance_w > 1+clip_factor] = 1+clip_factor
            importance_w[importance_w < 1-clip_factor] = 1-clip_factor
            # take the lower bound as loss
            exp_loss = (log_probs * As * importance_w).sum()
            exp_loss = exp_loss / len(exp)  # normalize to give smaller loss
            sum_exps += exp_loss
            # print log_probs, log_is to see where it went wrong
            #print('log_probs:')
            #print(log_probs)
            #print('log_is:')
            #print(log_is)
            # convert reward to loss by inserting -
        return -sum_exps / len(self.memory)
