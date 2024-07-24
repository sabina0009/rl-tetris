import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def plot_multiple_learning_curves(x, *score_lists, figure_file):
    colours = ['b','g', 'r', 'c', 'm', 'y', 'k']
    i=0
    for scores in score_lists:
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        plt.plot(x, running_avg, color=colours[i], label=f'Plot {i}')
        plt.title('Running average of previous 100 scores')
        plt.legend()
        i += 1
    plt.savefig(figure_file)