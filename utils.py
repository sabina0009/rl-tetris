import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_learning_curve(x,scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def plot_multiple_learning_curves(figure_file, *results_files):
    colours = ['b','g', 'r', 'c', 'm', 'y', 'k']
    j=0
    for file in results_files:
        scores = np.loadtxt(file, dtype=int)
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        x = [n for n in range(len(scores))]
        plt.plot(x, running_avg, color=colours[j], label=f'{file[8:-4]}')
        plt.title('Running average of previous 100 scores')
        plt.legend()
        j += 1
    plt.savefig(figure_file)

def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs