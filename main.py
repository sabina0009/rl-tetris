import gymnasium as gym
import gym_simpletetris
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve
import time

if __name__ == '__main__':
    env = gym.make('SimpleTetris-v0', reward_step=True)
    N = 20
    batch_size = 64
    n_epochs = 4
    n_options = 2
    alpha = 0.0003
    agent = Agent(n_options = n_options, n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=[env.observation_space.shape[0] * env.observation_space.shape[1]])
    n_games = 10000
    filename = 'tetris-option-critic-10000games'
    figure_file = f'plots/{filename}.png'
    best_score = env.reward_range[0]
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    start_time = time.time()

    option_lengths = {i:[] for i in range(n_options)}

    for i in range(n_games):

        observation, _ = env.reset()
        observation = observation.flatten()
        done = False
        score = 0

        current_option = 0
        state = agent.option_critic.get_state(observation)
        option_termination = True
        curr_op_len = 0

        while not done:

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = agent.option_critic.get_next_option(state)
                curr_op_len = 0

            action, prob, val = agent.choose_action(observation, current_option)

            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation_ = observation_.flatten()
            n_steps += 1
            curr_op_len += 1
            score += reward
            agent.remember(observation, current_option, action, prob, val, reward, done)

            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1

            observation = observation_
            state = agent.option_critic.get_state(observation)
            option_termination = agent.option_critic.predict_option_termination(state, current_option)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        time_elapsed = time.time() - start_time
        hours = time_elapsed // 3600
        time_elapsed = time_elapsed % 3600
        minutes = time_elapsed // 60
        seconds = time_elapsed % 60
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters, 'runtime %d:%d:%.1f' % (hours, minutes, seconds))
    
    np.savetxt(f'results/{filename}.txt', score_history, fmt='%d')

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
