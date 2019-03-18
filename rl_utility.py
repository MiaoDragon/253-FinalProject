import numpy as np
import torch
from scipy.misc import imresize

def obs_to_state(obs_num, obs, exp):
    # exp is a list of past experiences in the form (obs, action, reward, log_prob) up to previous time
    # each obs is a tensor
    # if length of exp is not enough for obs_num, then pad with the initial obs
    obs_list = [obs]
    # only need remaining from past experience
    obs_num = obs_num - 1
    exp_num = len(exp)
    eff_obs_num = min(obs_num, exp_num)
    for i in range(eff_obs_num):
        obs_list.append(exp[-i])
    for i in range(obs_num - eff_obs_num):
        # the rest use the initial obs
        # if exp has length 0, then just use obs
        if exp_num == 0:
            obs_list.append(obs)
        else:
            obs_list.append(exp[0])
    state = torch.stack(obs_list)
    return state

def preprocess(obs):
    # obs is numpy array
    # this works for 400x600 obs
    obs = obs[:350]
    obs = imresize(obs, (96,96))
    #obs = obs[:int(350/400*obs.shape[0])]
    obs = np.dot(obs[...,:3], [0.299, 0.587, 0.114]) / 255.
    return obs
