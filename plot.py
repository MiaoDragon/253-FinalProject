import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from discrete_policy import BaselineNet
from memory import Memory
import argparse
import gym
import os
import random
from sl_utility import *
from rl_utility import *
from plot_util import *
import sys
#import cv2
def main(args):
    # ----- load seed when there is saved one -----
    if os.path.exists(args.model_path):
        print('loading previous seed...')
        seed = load_seed(args.model_path)
    # obtain action bound
    #upper_action = env.action_space.high
    #lower_action = env.action_space.low
    # this is needed when we unnormalize the network output [0,1] to this
    # ----- metrics -----
    epi_reward = []
    train_loss = []
    # ----- cuda gpu -----
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        computing_device = torch.device("cuda")
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        print("CUDA NOT supported")
    epi_reward, train_loss = load_loss(args.model_path)
    plot(epi_reward, train_loss, name=args.save_name, smooth=args.smooth)
parser = argparse.ArgumentParser()
#parser.add_argument('--env', type=str, default='CarRacing-v0')
parser.add_argument('--model_path', type=str, default='../model/baseline.pkl')
parser.add_argument('--save_name', type=str, default='cartPole')
parser.add_argument('--smooth', type=int, default=False)
args = parser.parse_args()
reward_list = main(args)
