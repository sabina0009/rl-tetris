
import gymnasium as gym
import gym_simpletetris
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve, plot_average_learning_curve
import time

N = 2048
batch_size = 32
n_epochs = 10
alpha = 0.0003
filename = 'tetris-PPO-8x4'

def run_worker(process_num, score_history, lines_cleared_score):
    env = gym.make('SimpleTetris-v0', height=8, width=4)
    agent = Agent(n_actions=env.action_space.n, filename=filename, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=[env.observation_space.shape[0] * env.observation_space.shape[1]])
    #n_games = 1000
    max_steps = 200*N
    figure_file = f'plots/{filename}{process_num}.png'
    best_score = env.reward_range[0]
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    episode = 0
    start_time = time.time()
    np.random.seed(process_num)
    #for i in range(n_games):
    while n_steps < max_steps:
        observation, info = env.reset()
        observation = observation.flatten()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation_ = observation_.flatten()
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history[process_num].append(score)
        lines_cleared_score[process_num].append(info['lines_cleared'])
        avg_score = np.mean(score_history[process_num][-100:])
        episode += 1

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        time_elapsed = time.time() - start_time
        hours = time_elapsed // 3600
        time_elapsed = time_elapsed % 3600
        minutes = time_elapsed // 60
        seconds = time_elapsed % 60
        if episode % 50 == 0:
            print('process_num', process_num, 'episode', episode, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters, 'runtime %d:%d:%.1f' % (hours, minutes, seconds))
    
            np.savetxt(f'results/{filename}{process_num}.txt', score_history[process_num], fmt='%d')
            np.savetxt(f'results/lines_scores/{filename}{process_num}.txt', lines_cleared_score[process_num], fmt='%d')


            x = [i+1 for i in range(len(score_history[process_num]))]
            plot_learning_curve(x, score_history[process_num], figure_file)
            plot_learning_curve(x, lines_cleared_score[process_num],  f'plots/lines_scores/{filename}{process_num}.png')


    pass

threads = 5
score_history = [[] for i in range(threads)]
lines_cleared_score = [[] for i in range(threads)]
import torch.multiprocessing as mproc
import threading
if __name__ == '__main__':
    #share the network weights between the processes
    processes = []
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    ROLLING_EVENT.set()
    for process_num in range(threads):
            p = mproc.Process(target=run_worker, args=(process_num, score_history, lines_cleared_score))
            p.start()
            processes.append(p)
    for p in processes:
            p.join()

    plot_average_learning_curve(filename, threads)