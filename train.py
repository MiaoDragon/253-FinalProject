import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from policy import BaselineNet
from memory import Memory
import argparse
import gym
import os
import random
from sl_utility import *
from rl_utility import *
import sys
#import cv2
def main(args):
    seed = args.seed
    # ----- load seed when there is saved one -----
    if os.path.exists(args.model_path):
        print('loading previous seed...')
        seed = load_seed(args.model_path)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = gym.make(args.env)
    # obtain action bound
    upper_action = env.action_space.high
    lower_action = env.action_space.low
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
    # ----- model & optimizer ------
    if args.use_cnn:
        policyNet = BaselineNet(obs_num=args.obs_num, state_dim=64, action_dim=len(env.action_space.high), use_cnn=True)
    else:
        policyNet = BaselineNet(obs_num=args.obs_num, state_dim=args.obs_num*env.observation_space.shape[0], \
                                action_dim=len(env.action_space.high), use_cnn=False)
    if os.path.exists(args.model_path):
        print('loading previous model...')
        load_net_state(policyNet, args.model_path)
    policyNet = policyNet.to(computing_device)
    policyNet.set_opt(lr=args.learning_rate)
    if os.path.exists(args.model_path):
        print('loading optimizer state...')
        load_opt_state(policyNet.opt, args.model_path)
        old_epi_reward, old_train_loss = load_loss(args.model_path)
        epi_reward += old_epi_reward
        train_loss += old_train_loss

    memory = Memory(capacity=args.memory_capacity, obs_num=args.obs_num, computing_device=computing_device)
    # --- standard deviation for random sampling ---
    start_std = args.start_std
    final_std = args.final_std
    std_decay_epi = args.max_epi * args.std_decay_epi_ratio
    # linear decay of std
    std_decay = (start_std - final_std) / std_decay_epi
    std = start_std
    policyNet.update_std(std)
    total_reward = 0.
    # may consider adding random generation of data
    for i_episode in range(args.max_epi):
        obs = env.reset()
        # collect data using current policy
        # estimate the average reward on the go
        R = 0.
        exp = []
        for i in range(args.max_iter):
            #print('iteration: %d' % (i))
            if args.use_cnn:
                obs = env.render(mode='rgb_array')
                obs = preprocess(obs)  # for image, use this
            #cv2.imshow('hi', obs)
            obs = torch.FloatTensor(obs)
            obs = obs.to(computing_device)
            state = obs_to_state(args.obs_num, obs, exp).unsqueeze(0)
            action = policyNet.explore(state)
            action = action[0]
            # unnormalize the action by bound
            #print(action)
            perform_action = action.detach().data.cpu().numpy()
            perform_action = perform_action * (upper_action - lower_action) + lower_action
            log_prob = policyNet.log_prob(state, action)
            obs_next, reward, done, info = env.step(perform_action)
            R += reward
            obs = obs.detach()
            # sum each dim of log_prob to get joint prob
            exp.append( [obs, action, reward, log_prob.detach().data.sum(), None] )
            obs = obs_next
            if done:
                break
        memory.remember(exp)
        print('reward for episode %d: %f' % (i_episode, R))
        total_reward += R
        policyNet.zero_grad()
        # when episode 1 b will be equal to R, which leads to 0 J, not good
        # also pass the average reward
        J = memory.loss_traj(policyNet, total_reward / (i_episode+1), args.clip_factor)
        print('loss: %f' % (J))
        J.backward()
        policyNet.opt.step()
        # save reward and loss
        epi_reward.append(R)
        train_loss.append(J.detach().data.item())
        # update std
        std = std - std_decay
        policyNet.update_std(std)
        # print the thing
        sys.stdout.flush()
        # save model after several epochs
        if i_episode % args.save_epi == 0:
            save_state(policyNet, policyNet.opt, epi_reward, train_loss, seed, args.model_path)
    return epi_reward


parser = argparse.ArgumentParser()
#parser.add_argument('--env', type=str, default='CarRacing-v0')
parser.add_argument('--env', type=str, default='CarRacing-v0')
parser.add_argument('--max_epi', type=int, default=1000)
parser.add_argument('--max_iter', type=int, default=1000)
parser.add_argument('--save_epi', type=int, default=100)
parser.add_argument('--memory_capacity', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--obs_num', type=int, default=4)
parser.add_argument('--model_path', type=str, default='../model/baseline.pkl')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--start_std', type=float, default=5.)
parser.add_argument('--final_std', type=float, default=0.2)
parser.add_argument('--std_decay_epi_ratio', type=float, default=0.7)
parser.add_argument('--use_cnn', type=int, default=True)
parser.add_argument('--clip_factor', type=float, default=0.5, help='this makes sure the importance factor is within 1-alpha to 1+alpha')
args = parser.parse_args()
reward_list = main(args)
