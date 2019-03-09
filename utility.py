import torch
import torch.nn as nn
import numpy as np
def save_state(net, opt, train_loss, avg_minibatch_loss, val_loss, seed, fname):
    # save model state, optimizer state, train_loss, val_loss, random_seed
    states = {
        'state_dict': net.state_dict(),
        'optimizer': opt.state_dict(),
        'train_loss': train_loss,
        'avg_minibatch_loss': avg_minibatch_loss,
        'val_loss': val_loss,
        'seed': seed
    }
    torch.save(states, fname)

def load_net_state(net, fname):
    checkpoint = torch.load(fname)
    net.load_state_dict(checkpoint['state_dict'])

def load_opt_state(opt, fname):
    checkpoint = torch.load(fname)
    opt.load_state_dict(checkpoint['optimizer'])

def load_loss(fname):
    checkpoint = torch.load(fname)
    return checkpoint['train_loss'], checkpoint['avg_minibatch_loss'], checkpoint['val_loss']

def load_seed(fname):
    checkpoint = torch.load(fname)
    return checkpoint['seed']
