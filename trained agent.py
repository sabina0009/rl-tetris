import gym
import gym_simpletetris
import numpy as np
from ppo_torch import Agent

    
env = gym.make('SimpleTetris-v0', reward_step = True)
N = 20
batch_size = 64
n_epochs = 4
alpha = 0.0003
agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                alpha=alpha, n_epochs=n_epochs, 
                input_dims=[env.observation_space.shape[0] * env.observation_space.shape[1]])
agent.load_models()
n_games = 10

for i in range(n_games):
    observation = env.reset()
    observation = observation.flatten()
    done = False
    score = 0
    while not done:
        env.render()
        #action, _, _ = agent.choose_action(observation)
        action = env.action_space.sample()
        observation_, reward, done, info = env.step(action)
        observation_ = observation_.flatten()
        score += reward
        observation = observation_
    print(f"Game: {i}, Score: {score}")
