import gymnasium as gym
import gym_simpletetris
import numpy as np
from dqn_torch import Agent
from utils import plot_learning_curve
import time

if __name__ == '__main__':
    env = gym.make('SimpleTetris-v0', reward_step=True)
    n_options = 2
    batch_size = 64
    alpha = 0.0003
    agent = Agent(n_options=n_options, n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha,
                    input_dims=[env.observation_space.shape[0] * env.observation_space.shape[1]])
    #n_games = 1000
    max_steps = 100000
    filename = 'tetris-dqn-option-critic'
    figure_file = f'plots/{filename}.png'
    best_score = env.reward_range[0]
    score_history = []
    avg_score = 0
    episode = 0
    start_time = time.time()
    #for i in range(n_games):
    while agent.num_steps < max_steps:
        observation, _ = env.reset()
        observation = observation.flatten()
        done = False
        score = 0
        current_option = 0
        option_termination = True
        state = agent.policy_network.forward(observation)
        greedy_option = agent.policy_network.get_q(state).argmax(dim=-1).item()
        while not done:
            current_option, action, log_p, entropy = agent.choose_action(observation, current_option, option_termination, greedy_option)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation_ = observation_.flatten()
            score += reward
            agent.remember(observation, current_option, action, observation_, reward, done)
            agent.learn(observation, current_option, log_p, entropy, reward, done, observation_)
            if agent.num_steps % 200 == 0:
                agent.update_target_policy()
            observation = observation_
            state = agent.policy_network.forward(observation)
            option_termination, greedy_option = agent.policy_network.predict_option_termination(state, current_option)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
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
    
    np.savetxt(f'results/{filename}.txt', score_history, fmt='%d')

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
