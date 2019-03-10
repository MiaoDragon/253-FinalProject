import time
import seaborn as sn
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

import os

def _convert_to_xy(data):
    if isinstance(data, dict):
        x,y = zip(*(sorted(data.items())))
    elif isinstance(data, list):
        x = np.arange(n)+1
        y = data
    elif isinstance(data, np.ndarray):
        x = np.arange(n)+1
        y = list(data) # Assume 1D
    return (x,y)

# Data params should be lists or 1D arrays or dicts. `name` is a string
# identifier for the output figure PNG; if not provided, it will default to
# using the current datetime.
def plot(epi_reward, train_loss, name=None):
    fig = plt.figure(figsize=(8,4), dpi=80)
    ax = fig.add_subplot(111)
    title = 'loss over epochs'
    xlabel, ylabel = 'epochs', 'loss'
    nbins = 10

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    x,y = _convert_to_xy(epi_reward)
    plt.plot(x,y,label='episode reward')
    n = max(x)

    x,y = _convert_to_xy(train_loss)
    plt.plot(x,y,label='training loss')
    n = max(max(x),n)

    ticks = (np.arange(nbins) + 1) * n//nbins
    plt.xticks(ticks)

    ax.set_ylim(bottom=0)
    ax.margins(0)
    ax.legend()

    if name is None:
        name = time.strftime("%m-%d-%Y_%H-%M-%S")

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/run_'+name+'.png')
