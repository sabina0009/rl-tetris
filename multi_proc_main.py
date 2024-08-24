
import gymnasium as gym
import gym_simpletetris
import numpy as np
from dqn_torch import Agent
from utils import plot_learning_curve, plot_average_learning_curve
import time

N = 20
batch_size = 64
alpha = 0.0003
upd_freq =200
filename = 'tetris-DQNtargetnet'

def run_worker(process_num, score_history):
    env = gym.make('SimpleTetris-v0', reward_step=True)
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha,
                    input_dims=[env.observation_space.shape[0] * env.observation_space.shape[1]])
    #n_games = 1000
    max_steps = 150000
    figure_file = f'plots/{filename}{process_num}.png'
    best_score = env.reward_range[0]
    avg_score = 0
    episode = 0
    start_time = time.time()
    #for i in range(n_games):
    while agent.num_steps < max_steps:
        observation, _ = env.reset()
        observation = observation.flatten()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation_ = observation_.flatten()
            score += reward
            agent.remember(observation, action, observation_, reward, done)
            agent.learn()
            if agent.num_steps % upd_freq == 0:
                agent.update_target_policy()
            observation = observation_
        score_history[process_num].append(score)
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
        print('episode', episode, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', agent.num_steps, 'runtime %d:%d:%.1f' % (hours, minutes, seconds))
    
    np.savetxt(f'results/{filename}{process_num}.txt', score_history[process_num], fmt='%d')

    x = [i+1 for i in range(len(score_history[process_num]))]
    plot_learning_curve(x, score_history[process_num], figure_file)

    pass

threads = 5
score_history = [[] for _ in range(threads)]
import torch.multiprocessing as mproc
import threading
if __name__ == '__main__':
    #share the network weights between the processes
    processes = []
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    ROLLING_EVENT.set()
    for process_num in range(threads):
            p = mproc.Process(target=run_worker, args=(process_num, score_history))
            p.start()
            processes.append(p)
    for p in processes:
            p.join()

    plot_average_learning_curve(filename, threads)