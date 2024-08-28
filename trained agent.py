import gymnasium as gym
import gym_simpletetris
import numpy as np
from ppo_torch import Agent

    
env = gym.make('SimpleTetris-v0', reward_step = True)
N = 2048
batch_size = 32
n_epochs = 10
alpha = 0.0003
filename = 'tetris'
agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, filename=filename,
                alpha=alpha, n_epochs=n_epochs, 
                input_dims=[env.observation_space.shape[0] * env.observation_space.shape[1]])
agent.load_models()
n_games = 10
scores = []

for i in range(n_games):
    observation, _ = env.reset()
    observation = observation.flatten()
    done = False
    score = 0
    while not done:
        env.render()
        action, _, _ = agent.choose_action(observation)
        observation_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        observation_ = observation_.flatten()
        score += reward
        observation = observation_
    print(score)
