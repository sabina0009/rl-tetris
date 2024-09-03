
import gymnasium as gym
import gym_simpletetris
import minigrid
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve, plot_average_learning_curve, make_env
import time
import os

import torch.multiprocessing as mproc
import threading

from minigrid.wrappers import FlatObsWrapper

def run_worker(process_num, score_history, N, batch_size, n_epochs, args, filename):
    alpha = 0.0003

    plot_path = f'plots/{filename}'
    results_path = f'results/{filename}'

    env, input_dims = make_env(args.environment, args.boardsize)
    input_dims = [input_dims]

    agent = Agent(n_actions=env.action_space.n, filename=filename, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs,
                    input_dims=input_dims)
    #n_games = 1000
    max_steps = args.steps
    figure_file = f'plots/{filename}-{process_num}.png'
    best_score = env.reward_range[0]
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    episode = 0
    start_time = time.time()
    np.random.seed(process_num)
    #for i in range(n_games):
    while n_steps < max_steps:
        if args.environment == 'FourRooms':
            observation = env.reset()
        else:
            observation, _ = env.reset()
        if args.environment == 'Tetris':
            observation = observation.flatten()
        done = False
        score = 0
        ep_len = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            if args.environment == 'FourRooms':
                 observation_, reward, done, _ = env.step(action)
            else:
                observation_, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            if args.environment == 'Tetris':
                observation_ = observation_.flatten()
            n_steps += 1
            score += reward
            ep_len += 1
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        if args.environment == 'FourRooms':
             score = ep_len
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

        print('process_num', process_num, 'episode', episode, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            'time_steps', n_steps, 'learning_steps', learn_iters, 'runtime %d:%d:%.1f' % (hours, minutes, seconds))
    
    try:
         os.mkdir(plot_path)
    except:
         pass
    
    try:
         os.mkdir(results_path)
    except:
         pass
    
    
    np.savetxt(f'{results_path}/{filename}-{process_num}.txt', score_history[process_num], fmt='%d')

    x = [i+1 for i in range(len(score_history[process_num]))]
    plot_learning_curve(x, score_history[process_num], f'{plot_path}/{filename}-{process_num}.png')



    pass

def run_PPO(args, filename):

    if args.version == 0:
        N = 2048
        batch_size = 32
        n_epochs = 10
    
    elif args.version == 1:
        N = 20
        batch_size = 64
        n_epochs = 4
         
    threads = args.runs
    score_history = [[] for i in range(threads)]

    processes = []
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    ROLLING_EVENT.set()
    for process_num in range(threads):
            p = mproc.Process(target=run_worker, args=(process_num, score_history, N, batch_size, n_epochs, args, filename))
            p.start()
            processes.append(p)
    for p in processes:
            p.join()

    plot_average_learning_curve(filename, threads)

