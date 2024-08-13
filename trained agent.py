import gymnasium as gym
import gym_simpletetris
import numpy as np
from ppo_torch import Agent
import time
from utils import to_tensor
    
env = gym.make('SimpleTetris-v0', reward_step = True, height = 8, width = 4)
N = 20
batch_size = 64
n_epochs = 4
alpha = 0.0003
agent = Agent(n_options=2, n_actions=env.action_space.n, batch_size=batch_size, 
                alpha=alpha, n_epochs=n_epochs, 
                input_dims=[env.observation_space.shape[0] * env.observation_space.shape[1]])
agent.load_models()
n_games = 10
scores = []

for i in range(n_games):
    observation, _ = env.reset()
    observation = observation.flatten()
    state = agent.option_critic.get_state(observation)
    current_option = agent.option_critic.get_next_option(state)
    done = False
    score = 0
    while not done:
        env.render()
        obs = to_tensor(observation).to(agent.option_critic.device)
        state = agent.option_critic.get_state(obs)
        if agent.option_critic.predict_option_termination(state, current_option):
            current_option = agent.option_critic.get_next_option(state)
        action, _, _ = agent.choose_action(obs, current_option)
        observation_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        observation_ = observation_.flatten()
        score += reward
        observation = observation_
    print(score)