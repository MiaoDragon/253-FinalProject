"""
this implements the memory recording past trajectories
each memory records: (s_t, a_t, r_t, log_p(a_t|s_t))
memory is a circular array of array
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

    def loss(self, net):
        self.compute_b()
        b = self.b
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
            for t in range(len(exp)):
                o, a, r, log_p = exp[t]
                # stack observations into states
                states.append(obs_to_state(self.obs_num, o, exp[:t])) # :t actually uses past exps
                actions.append(a)
            states = torch.stack(states)
            actions = torch.stack(actions)
            actions = actions.to(self.computing_device)
            # sum probabiliies to obtain joint probaility
            log_probs = net.log_prob(states, actions).sum(dim=1)
            for t in range(len(exp)):
                _, _, _, log_p = exp[t]
                # added clipping to avoid gradient explosure
                log_is += log_probs[t] - max(log_p.detach().sum(), np.log(1e-5))
            # treat the IS term as data, don't compute gradient w.r.t. it
            log_is = log_is.detach()
            for t in range(len(exp)-1,-1,-1):
                (o, a, r, log_p) = exp[t]
                net_log_prob = log_probs[t]
                R += r - bt
                # future reward * current loglikelihood * past IS
                sum_exp += net_log_prob * R * torch.exp(log_is)
                log_is -= net_log_prob - max(log_p.detach().sum(), np.log(1e-5))
                # the predicted prob is treated as constant
                log_is = log_is.detach()
            sum_exps += sum_exp
            # convert reward to loss by inserting -
        return -sum_exps / len(self.memory)
