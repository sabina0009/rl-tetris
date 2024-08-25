import numpy as np
import torch
from copy import deepcopy

from option_critic import OptionCriticFeatures, OptionCriticConv
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn

from experience_replay import ReplayBuffer
from utils import make_env, to_tensor, save_models

from utils import plot_learning_curve, plot_average_learning_curve
import time

filename='tetris-ppoc'

def run(process_num, score_history):
    env_name = 'SimpleTetris-v0'
    env, is_atari = make_env(env_name)
    option_critic = OptionCriticConv if is_atari else OptionCriticFeatures
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    learning_rate = 0.0005
    max_history = 10000
    max_steps = 10000
    max_steps_ep = 10000
    num_options = 2
    batch_size = 3
    N = 20
    n_epochs = 4

    best_score = env.reward_range[0]
    avg_score = 0
    
    gamma = 0.99
    gae_lambda = 0.95
    entropy_reg = 0.01
    eta = 0.01
    policy_clip = 0.2

    option_critic = option_critic(
        in_features=env.observation_space.shape[0] if env_name != 'SimpleTetris-v0' else 200,
        num_actions=env.action_space.n,
        num_options=num_options,
        temperature=1.0,
        eps_start=1.0,
        eps_min=0.1,
        eps_decay=20000,
        eps_test=0.05,
        device=device
    )

    optim = torch.optim.Adam(option_critic.parameters(), lr=learning_rate)

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    #env.seed(args.seed)

    buffer = ReplayBuffer(capacity=max_history, seed=seed)

    steps = 0 
    episode = 0
    learn_iters = 0
    start_time = time.time()

    while steps < max_steps:

        rewards = 0 ; option_lengths = {opt:[] for opt in range(num_options)}

        obs, _   = env.reset()
        if env_name == 'SimpleTetris-v0':
            obs = obs.flatten()
        state = option_critic.get_state(to_tensor(obs))
        current_option = 0
        termination_cost = 0

        done = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0
        while not done and ep_steps < max_steps_ep:

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = option_critic.get_next_option(state)
                curr_op_len = 0
                termination_cost = eta
            else:
                termination_cost = 0

            action, logp, val = option_critic.choose_action(obs, current_option)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if env_name == 'SimpleTetris-v0':
                next_obs = next_obs.flatten()

            steps += 1
            ep_steps += 1
            curr_op_len += 1
            reward = reward - termination_cost
            rewards += reward

            buffer.push(obs, current_option, action, logp, val, reward, done)

            actor_loss, critic_loss = None, None

            if steps % N == 0:
                for _ in range(n_epochs):
                    obs_arr, option_arr, action_arr, old_prob_arr, vals_arr,\
                        reward_arr, dones_arr, batches = buffer.sample(batch_size)
                    
                    values = vals_arr
                    advantage = np.zeros(len(reward_arr), dtype=np.float32)

                    for t in range(len(reward_arr)-1):
                        discount = 1
                        a_t = 0
                        for k in range(t, len(reward_arr)-1):
                            a_t += discount*(reward_arr[k] + gamma*values[k+1]*\
                                    (1-int(dones_arr[k])) - values[k])
                            discount *= gamma*gae_lambda
                        advantage[t] = a_t
                    advantage = torch.tensor(advantage).to(device)

                    values = torch.tensor(values).to(device)

                    for batch in batches:
                        obss = torch.tensor(obs_arr[batch], dtype=torch.float).to(device)
                        states = option_critic.get_state(obss).squeeze()
                        old_probs = torch.tensor(old_prob_arr[batch]).to(device)
                        options = torch.tensor(option_arr[batch]).to(device)
                        actions = torch.tensor(action_arr[batch]).to(device)
                        dones = torch.tensor(dones_arr[batch]).to(device)
                        dones = dones.long()

                        actor_loss = actor_loss_fn(option_critic, batch, states, old_probs, options, actions, dones, \
                                                   advantage, policy_clip, termination_cost, entropy_reg)
                        critic_loss = critic_loss_fn(option_critic, obss, options, advantage, values, batch)

                        total_loss = actor_loss + 0.5*critic_loss
                        optim.zero_grad()
                        total_loss.backward()
                        optim.step()
                    
                buffer.clear_memory()
                learn_iters += 1

            obs = next_obs
            state = option_critic.get_state(to_tensor(obs))
            option_termination = option_critic.predict_option_termination(state, current_option)

        score_history[process_num].append(rewards)
        avg_score = np.mean(score_history[process_num][-100:])
        episode += 1

        if avg_score > best_score:
            best_score = avg_score
            option_critic.save_checkpoint(f'option_critic_{process_num}')

        time_elapsed = time.time() - start_time
        hours = time_elapsed // 3600
        time_elapsed = time_elapsed % 3600
        minutes = time_elapsed // 60
        seconds = time_elapsed % 60

        print('process_num', process_num, ' | episode', episode, ' | score %.1f' % rewards, ' | avg score %.1f' % avg_score,
                ' | time_steps', steps, ' | learning steps', learn_iters, ' | runtime %d:%d:%.1f' % (hours, minutes, seconds))

        if episode % 50 == 0:
            np.savetxt(f'results/{filename}-{process_num}.txt', score_history[process_num], fmt='%d')
            x = [i+1 for i in range(len(score_history[process_num]))]
            plot_learning_curve(x, score_history[process_num], f'plots/{filename}-{process_num}.png')

    pass

threads = 1
score_history = [[] for i in range(threads)]
import torch.multiprocessing as mproc
import threading
if __name__ == '__main__':
    #share the network weights between the processes
    processes = []
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    ROLLING_EVENT.set()
    for process_num in range(threads):
            p = mproc.Process(target=run, args=(process_num, score_history))
            p.start()
            processes.append(p)
    for p in processes:
            p.join()

    plot_average_learning_curve(filename, threads)