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
from PIL import Image
import cv2
from skimage.transform import resize
##convert to grayscale
def preprocess(observation):
    observation = observation[:350]
    return np.dot(observation[...,:3], [0.299, 0.587, 0.114])
#import scipy
def main(args):
    env = gym.make(args.env).unwrapped
    #env = gym.make('CartPole-v0')
    # here if action space is continuous, we try to discretize it
    action_mapping = []
    policyNet = BaselineNet(32, len(env.action_space.high))
    memory = Memory(capacity=200)
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
            obs = env.render(mode='rgb_array')
            obs = preprocess(obs)
            # display the obs frame every 15 iterations
            # obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_CUBIC)
            # if i%15 == 0 :
            #     img = Image.fromarray(obs)
            #     img.show()
            obs = resize(obs, (96,96,1),anti_aliasing=True)
            obs = obs / 255.
            # obs = imresize(obs, (96,96,3))
            # change from H*W*C to C*H*W
            obs = torch.FloatTensor(obs).permute(2,0,1).unsqueeze(0)
            # action = policyNet.explore(obs)
            # action = action[0]
            # perform_action = action.detach().data.cpu().numpy()
            perform_action = env.action_space.sample()
            #if action[0] < -1.:
            #    action[0] = -1.
            #if action[1] < 0.:
            #    action[1] = 0.
            #if action[2] < 0.:
            #    action[2] = 0.
            # log_prob = policyNet.log_prob(obs, action)
            # log_prob = policyNet.log_prob(obs, torch.FloatTensor(perform_action))
            log_prob = 0
            obs_next, reward, done, info = env.step(perform_action)
            obs_next = preprocess(obs_next)
            R += reward
            obs = obs.detach()
            # exp.append( (obs, action, reward, log_prob) )
            exp.append( (obs, perform_action, reward, log_prob) )
            obs = obs_next
            if done:
                break

        reward_list.append(R)
        memory.remember(exp)
        print('reward for episode %d: %f' % (i_episode, R))
        policyNet.zero_grad()
        # when episode 1 b will be equal to R, which leads to 0 J, not good
        J = memory.loss(policyNet)
        print(J)
        J.backward()
        policyNet.opt.step()
    return reward_list


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CarRacing-v0')
#parser.add_argument('--env', type=str, default='Pendulum-v0')
parser.add_argument('--max_epi', type=int, default=200)
parser.add_argument('--max_iter', type=int, default=200)
args = parser.parse_args()
reward_list = main(args)
print(reward_list)
