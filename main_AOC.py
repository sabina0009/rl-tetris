
# Code based on https://github.com/lweitkamp/option-critic-pytorch
# Code modified in collaboration with project supervisor, Andreas Theophilou

import numpy as np
import argparse
import torch
from copy import deepcopy

from aoc import OptionCriticFeatures, OptionCriticConv
from aoc import critic_loss as critic_loss_fn
from aoc import actor_loss as actor_loss_fn

from experience_replay import ReplayBuffer
from utils import make_env, to_tensor,  save_models
from utils import plot_learning_curve, plot_average_learning_curve
from utils import get_column_heights
from logger import Logger

import time
import os
import gymnasium as gym
import gym_simpletetris
import torch.multiprocessing as mproc
import threading

import time

# parser = argparse.ArgumentParser(description="Option Critic PyTorch")
# parser.add_argument('--env', default='CartPole-v0', help='ROM to run')
# parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
# parser.add_argument('--frame-skip', default=4, type=int, help='Every how many frames to process')
# parser.add_argument('--learning-rate',type=float, default=.001, help='Learning rate')
# parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
# parser.add_argument('--epsilon-start',  type=float, default=1.0, help=('Starting value for epsilon.'))
# parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
# parser.add_argument('--epsilon-decay', type=float, default=20000, help=('Number of steps to minimum epsilon.'))
# parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
# parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
# parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes.'))
# parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
# parser.add_argument('--termination-reg', type=float, default=0.01, help=('Regularization to decrease termination prob.'))
# parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
# parser.add_argument('--num-options', type=int, default=4, help=('Number of options to create.'))
# parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')

# parser.add_argument('--max_steps_ep', type=int, default=10000000, help='number of maximum steps per episode.')
# parser.add_argument('--max_steps_total', type=int, default=100000, help='number of maximum steps to take.') # bout 4 million
# parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
# parser.add_argument('--seed', type=int, default=0, help='Random seed for numpy, torch, random.')
# parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
# parser.add_argument('--exp', type=str, default=None, help='optional experiment name')
# parser.add_argument('--switch-goal', type=bool, default=False, help='switch goal after 2k eps')


def run(process_num, score_history, plot_path, results_path, filename, arguments):

    optimal_eps=0.05
    learning_rate=.001
    epsilon_start=1.0
    epsilon_min=.1
    epsilon_decay=20000
    max_history=10000
    batch_size=128
    freeze_interval=200
    update_frequency=4
    num_options=arguments.options
    temp=1

    avg_score = 0

    max_steps_ep=10000000
    max_steps_total=arguments.steps
    cuda=True
    seed=0

    env, in_features = make_env(arguments.environment, arguments.boardsize)
    is_atari = False
    best_score = env.reward_range[0] 


    option_critic = OptionCriticConv if is_atari else OptionCriticFeatures
    device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')

    additional_feature_len = num_options 

    option_critic = option_critic(
        in_features=in_features + additional_feature_len,
        num_actions=env.action_space.n,
        num_options=num_options,
        temperature=temp,
        eps_start=epsilon_start,
        eps_min=epsilon_min,
        eps_decay=epsilon_decay,
        eps_test=optimal_eps,
        device=device
    )
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)

    optim = torch.optim.RMSprop(option_critic.parameters(), lr=learning_rate)

    np.random.seed(seed)
    torch.manual_seed(seed)
    #env.seed(args.seed)

    buffer = ReplayBuffer(capacity=max_history, seed=seed)
    steps = 0 ;
    start_time = time.time()
    episode = 0

    while steps < max_steps_total:

        rewards = 0 ; option_lengths = {opt:[] for opt in range(num_options)}

        obs, _   = env.reset()

        # Get Tetris features
        num_holes = env.engine.holes 
        piece_height = sum(np.any(env.engine.board, axis=0)) 
        heights = get_column_heights(env.engine.board) 
        aggregate_heights = sum(np.any(env.engine.board, axis=0)) 
        lines = env.engine.lines_cleared
        bumpiness = sum([abs(heights[i]-heights[i+1]) for i in range(len(heights)-1)])
        if additional_feature_len == 2:
            additional_features = np.array([num_holes, piece_height]) 
        if additional_feature_len == 4:
            additional_features = np.array([num_holes, aggregate_heights, lines, bumpiness]) 
        # Added features to the state representation
        obs = np.concatenate([obs.flatten(), additional_features]).flatten() 

        state = option_critic.get_state(to_tensor(obs))
        greedy_option  = option_critic.greedy_option(state)
        current_option = 0

        done = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0
        while not done and ep_steps < max_steps_ep:
            epsilon = option_critic.epsilon

            # Predict option termination and find next option
            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(num_options) if np.random.rand() < epsilon else greedy_option

                curr_op_len = 0
    
            # Get next action and step environment
            action, logp, entropy = option_critic.choose_action(obs, current_option)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Get feature and add to state representation
            if additional_feature_len == 2:
                num_holes = env.engine.holes 
                piece_height = sum(np.any(env.engine.board, axis=0))
                additional_features = np.array([num_holes, piece_height])
            if additional_feature_len == 4:
                if env.engine.has_dropped:
                    num_holes = env.engine.holes 
                    heights = get_column_heights(env.engine.board)
                    aggregate_heights = sum(np.any(env.engine.board, axis=0)) 
                    lines = env.engine.lines_cleared
                    bumpiness = sum([abs(heights[i]-heights[i+1]) for i in range(len(heights)-1)])
                additional_features = np.array([num_holes, aggregate_heights, lines, bumpiness]) 
            next_obs = np.concatenate([next_obs.flatten(), additional_features]).flatten() 

            buffer.push(obs, current_option, reward, next_obs, done)
            rewards += reward +1

            actor_loss, critic_loss = None, None
            if len(buffer) > batch_size:
                actor_loss = actor_loss_fn(obs, current_option, logp, entropy, \
                    reward, done, next_obs, option_critic, option_critic_prime)
                loss = actor_loss

                if steps % update_frequency == 0:
                    data_batch = buffer.sample(batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch)
                    loss += critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            state = option_critic.get_state(to_tensor(next_obs))
            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)

            # update global steps etc
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs

        score_history[process_num].append(rewards)
        score_history[process_num].append(rewards)
        avg_score = np.mean(score_history[process_num][-100:])
        episode += 1

        if avg_score > best_score:
            best_score = avg_score
            save_models(option_critic, option_critic_prime, process_num, filename)

        time_elapsed = time.time() - start_time
        hours = time_elapsed // 3600
        time_elapsed = time_elapsed % 3600
        minutes = time_elapsed // 60
        seconds = time_elapsed % 60

        print('process_num', process_num, ' | episode', episode, ' | score %.1f' % rewards, ' | avg score %.1f' % avg_score,
                ' | time_steps', steps, ' | runtime %d:%d:%.1f' % (hours, minutes, seconds))

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


def run_AOC(arguments, filename):
    threads = args.runs
    score_history = [[] for i in range(threads)]

    plot_path = f'plots/{filename}'
    results_path = f'results/{filename}'

    processes = []
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    ROLLING_EVENT.set()
    for process_num in range(threads):
            p = mproc.Process(target=run, args=(process_num, score_history, plot_path, results_path, filename, arguments))
            p.start()
            processes.append(p)
    for p in processes:
            p.join()

    plot_average_learning_curve(filename, threads)
