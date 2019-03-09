import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from policy import BaselineNet
from memory import Memory
import argparse
import gym
from scipy.misc import imresize
#import scipy
def main(args):
    env = gym.make(args.env)
    #env = gym.make('CartPole-v0')
    # cuda gpu
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        computing_device = torch.device("cuda")
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        print("CUDA NOT supported")

    action_mapping = []
    policyNet = BaselineNet(env.observation_space.shape[0], len(env.action_space.high))
    policyNet = policyNet.to(computing_device)
    policyNet.set_opt()
    memory = Memory(capacity=200, computing_device=computing_device)
    b = 0.   # we can use a weighted average to estimate b as latest ones are more closer to current policy
    i_episode = 0
    reward_list = []
    start_std = 5.
    final_std = 0.1
    std_decay_epi = args.max_epi * 0.8
    # linear decay of std
    std_decay = (start_std - final_std) / std_decay_epi
    std = start_std
    policyNet.update_std(std)
    # may consider adding random generation of data
    while i_episode < args.max_epi:
        i_episode += 1
        obs = env.reset()
        # collect data using current policy
        # estimate the average reward on the go
        R = 0.
        exp = []
        for i in range(args.max_iter):
            #obs = env.render(mode='rgb_array')
            #obs = imresize(obs, (96,96,3))
            #obs = obs / 255.
            # change from H*W*C to C*H*W
            #obs = torch.FloatTensor(obs).permute(2,0,1).unsqueeze(0)
            obs = torch.FloatTensor(obs)
            obs = obs.to(computing_device)
            action = policyNet.explore(obs)
            action = action[0]
            perform_action = [action.detach().data.cpu().numpy()]
            log_prob = policyNet.log_prob(obs, action)
            obs_next, reward, done, info = env.step(perform_action)
            R += reward
            obs = obs.detach()
            exp.append( (obs, action, reward, log_prob) )
            obs = obs_next
            if done:
                break
        reward_list.append(R)
        memory.remember(exp)
        print('reward for episode %d: %f' % (i_episode, R))
        policyNet.zero_grad()
        # when episode 1 b will be equal to R, which leads to 0 J, not good
        J = memory.loss(policyNet)
        #print(J)
        J.backward()
        policyNet.opt.step()
        # update std
        std = std - std_decay
        policyNet.update_std(std)
        # save model after several epochs
        #if i_episode % save_epi == 0:

    return reward_list


parser = argparse.ArgumentParser()
#parser.add_argument('--env', type=str, default='CarRacing-v0')
parser.add_argument('--env', type=str, default='Pendulum-v0')
parser.add_argument('--max_epi', type=int, default=25000)
parser.add_argument('--max_iter', type=int, default=200)
args = parser.parse_args()
reward_list = main(args)
#print(reward_list)
