
# Code based on https://www.youtube.com/watch?v=hlv79rcHws0&t=1977s and 
# https://github.com/lweitkamp/option-critic-pytorch

import numpy as np
import matplotlib.pyplot as plt

import gymnasium
import gym_simpletetris
from fourrooms import Fourrooms
import numpy as np
import torch

def plot_learning_curve(x, scores, figure_file):
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

def plot_average_learning_curve(filename, runs):
    results_files = [f'results/{filename}/{filename}-{i}.txt' for i in range(runs)]
    scores = []
    for i in range(runs):
        score_history = np.loadtxt(results_files[i], dtype=int)
        scores.append(score_history)
    run_lengths = [len(score_history) for score_history in scores]
    min_len = min(run_lengths)
    x = [n for n in range(min_len)]
    avg_scores = []
    for i in range(min_len):
        avg = sum([scores[j][i] for j in range(runs)]) / runs
        avg_scores.append(avg)
    np.savetxt(f'results/{filename}/average.txt', avg_scores, fmt='%d')
    running_avg = np.zeros(len(avg_scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(avg_scores[max(0, i-100):(i+1)])
    print(f'...plotting average or {runs} runs...')
    plt.plot(x, running_avg)
    plt.title(f'{filename} average of {runs} runs')
    plt.savefig(f'plots/{filename}/average.png')

def make_env(environment, board_size):
        
    if environment == 'Tetris':
        if board_size == '20x10':
            env = gymnasium.make('SimpleTetris-v0', reward_step=True)
            input_dims = 200
            return env, input_dims 
        
        if board_size == '8x4':
            env = gymnasium.make('SimpleTetris-v0', height=8, width=4)
            input_dims = 32
            return env, input_dims 
        
    elif environment == 'CartPole':
        env = gymnasium.make('CartPole-v1')
        input_dims = env.observation_space.shape[0]
        return env, input_dims 
    
    elif environment == 'FourRooms':
        env = Fourrooms()
        input_dims = env.observation_space.shape[0]
        return env, input_dims

def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs

def save_models(option_critic, option_critic_prime, process_num, filename):
    print('... saving models ...')
    option_critic.save_checkpoint(f'option_critic {filename} {process_num}')
    option_critic_prime.save_checkpoint(f'option_critic_prime {filename} {process_num}')

def load_models(option_critic, option_critic_prime, process_num, filename):
    print('... loading models ...')
    option_critic.load_checkpoint(f'option_critic {filename} {process_num}')
    option_critic_prime.load_checkpoint(f'option_critic_prime {filename} {process_num}')

def get_column_heights(board):
    heights = []
    for column in board:
        height = 0
        column = column[::-1]
        for i in range(len(column)):
            if column[i] == 1:
                height = i + 1
        heights.append(height)
    return heights
