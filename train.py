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
    policyNet = BaselineNet(args.obs_num*env.observation_space.shape[0], len(env.action_space.high))
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
    for name, param in policyNet.named_parameters():
        print(name)
        print(param)
    # may consider adding random generation of data
    for i_episode in range(args.max_epi):
        obs = env.reset()
        # collect data using current policy
        # estimate the average reward on the go
        R = 0.
        exp = []
        for i in range(args.max_iter):
            #obs = preprocess(obs)  # for image, use this
            obs = torch.FloatTensor(obs)
            obs = obs.to(computing_device)
            state = obs_to_state(args.obs_num, obs, exp).unsqueeze(0)
            action = policyNet.explore(state)
            action = action[0]
            print(action)
            perform_action = action.detach().data.cpu().numpy()
            log_prob = policyNet.log_prob(state, action)
            obs_next, reward, done, info = env.step(perform_action)
            R += reward
            obs = obs.detach()
            exp.append( (obs, action, reward, log_prob) )
            obs = obs_next
            if done:
                break
        memory.remember(exp)
        print('reward for episode %d: %f' % (i_episode, R))
        policyNet.zero_grad()
        # when episode 1 b will be equal to R, which leads to 0 J, not good
        J = memory.loss(policyNet)
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
parser.add_argument('--env', type=str, default='Pendulum-v0')
parser.add_argument('--max_epi', type=int, default=50000)
parser.add_argument('--max_iter', type=int, default=1000)
parser.add_argument('--save_epi', type=int, default=500)
parser.add_argument('--memory_capacity', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--obs_num', type=int, default=4)
parser.add_argument('--model_path', type=str, default='../model/baseline.pkl')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--start_std', type=float, default=5.)
parser.add_argument('--final_std', type=float, default=0.1)
parser.add_argument('--std_decay_epi_ratio', type=float, default=0.7)
args = parser.parse_args()
reward_list = main(args)
