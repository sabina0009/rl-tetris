import os
import numpy as np
import torch
from copy import deepcopy

from option_critic import OptionCriticFeatures, OptionCriticConv
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn

from experience_replay import ReplayBuffer
from utils import make_env, to_tensor, save_models

from utils import plot_learning_curve, plot_average_learning_curve
from utils import get_column_heights
import time

env_name = 'tetris20x10'
max_steps = 200000
num_options = 4

filename=f'4 options - aggregate height - version 3 - {max_steps} steps'
plot_path = f'plots/{filename}'
results_path = f'results/{filename}'
lines_path = f'results/{filename}/lines_cleared'

def run(process_num, score_history, lines_cleared, num_options):
    env, is_atari = make_env(env_name)
    option_critic = OptionCriticConv if is_atari else OptionCriticFeatures
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    additional_feature_len = num_options #added code

    learning_rate = 0.0005
    max_history = 10000
    max_steps_ep = 10000
    batch_size = 32
    update_frequency = 4
    freeze_interval = 200

    best_score = env.reward_range[0]
    avg_score = 0
    
    gamma = 0.99
    termination_reg = 0.01
    entropy_reg = 0.01

    if env_name == 'tetris20x10':
        in_features = 200 + additional_feature_len
    elif env_name == 'tetris8x4':
        in_features = 32 + additional_feature_len
    else:
        in_features = env.observation_space[0]

    option_critic = option_critic(
        in_features= in_features,
        num_actions=env.action_space.n,
        num_options=num_options,
        temperature=1.0,
        eps_start=1.0,
        eps_min=0.1,
        eps_decay=20000,
        eps_test=0.05,
        device=device
    )
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)

    optim = torch.optim.RMSprop(option_critic.parameters(), lr=learning_rate)

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    #env.seed(args.seed)

    buffer = ReplayBuffer(capacity=max_history, seed=seed)

    steps = 0 
    episode = 0
    start_time = time.time()

    while steps < max_steps:

        rewards = 0 ; option_lengths = {opt:[] for opt in range(num_options)}

        obs, _   = env.reset()

        num_holes = env.engine.holes #added code
        lines_cleared = env.engine.lines_cleared
        piece_height = sum(np.any(env.engine.board, axis=0)) #added code
        heights = get_column_heights(env.engine.board)
        aggregate_height = sum(heights)
        bumpiness = sum([abs(heights[i]-heights[i+1]) for i in range(len(heights)-1)])
        if additional_feature_len == 2:
            additional_features = np.array([num_holes, aggregate_height]) #added code
        elif additional_feature_len == 4:
            additional_features =  np.array([num_holes, aggregate_height, lines_cleared, bumpiness])
        obs = np.concatenate([obs.flatten(), additional_features]).flatten() #added code

        state = option_critic.get_state(to_tensor(obs))
        greedy_option  = option_critic.greedy_option(state)
        current_option = 0

        done = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0
        while not done and ep_steps < max_steps_ep:
            epsilon = option_critic.epsilon

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0
    
            action, logp, entropy = option_critic.get_action(obs, current_option, env)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if env.engine.has_dropped:
                num_holes = env.engine.holes #added code
                lines_cleared = env.engine.lines_cleared
                piece_height = sum(np.any(env.engine.board, axis=0)) #added code
                heights = get_column_heights(env.engine.board)
                aggregate_height = sum(heights)
                bumpiness = sum([abs(heights[i]-heights[i+1]) for i in range(len(heights)-1)])
            if additional_feature_len == 2:
                additional_features = np.array([num_holes, aggregate_height]) #added code
            elif additional_feature_len == 4:
                additional_features =  np.array([num_holes, aggregate_height, lines_cleared, bumpiness])
            next_obs = np.concatenate([next_obs.flatten(), additional_features]).flatten() #added code

            buffer.push(obs, current_option, reward, next_obs, done)
            rewards += reward

            actor_loss, critic_loss = None, None
            if len(buffer) > batch_size:
                actor_loss = actor_loss_fn(obs, current_option, logp, entropy, \
                    reward, done, next_obs, option_critic, option_critic_prime, gamma, termination_reg, entropy_reg)
                loss = actor_loss

                if steps % update_frequency == 0:
                    data_batch = buffer.sample(batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, gamma)
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
        lines_history[process_num].append(lines_cleared)
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
    
    try:
         os.mkdir(lines_path)
    except:
         pass

    np.savetxt(f'{results_path}/{filename}-{process_num}.txt', score_history[process_num], fmt='%d')
    np.savetxt(f'{lines_path}/{filename}-{process_num}.txt', lines_history[process_num], fmt='%d')

    x = [i+1 for i in range(len(score_history[process_num]))]
    plot_learning_curve(x, score_history[process_num], f'{plot_path}/{filename}-{process_num}.png')

    pass

threads = 5
score_history = [[] for i in range(threads)]
lines_history = [[] for i in range(threads)]
import torch.multiprocessing as mproc
import threading
if __name__ == '__main__':
    #share the network weights between the processes
    processes = []
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    ROLLING_EVENT.set()
    for process_num in range(threads):
            p = mproc.Process(target=run, args=(process_num, score_history, lines_history, num_options))
            p.start()
            processes.append(p)
    for p in processes:
            p.join()

    plot_average_learning_curve(plot_path, results_path, filename, threads)