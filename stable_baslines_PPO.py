import gymnasium
import gym
import gym_simpletetris
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter


log_dir = "/tmp/sb3_log/log"

env = gym.make('SimpleTetris-v0', reward_step=True)
env = Monitor(env, log_dir)

env.observation_space = gymnasium.spaces.Box(0.0, 1.0, (10, 20), float)
env.action_space = gymnasium.spaces.Discrete(7)

model = PPO("MlpPolicy", env, learning_rate=0.0003, n_steps=20, batch_size=64, n_epochs=4, gamma=0.99,
                   gae_lambda=0.95, clip_range=0.2, verbose=1)
model.learn(total_timesteps=100000)


obs = env.reset()
done = False
while not done:
    action, state_ = model.predict(obs)
    action = int(action)
    obs, reward, done, info = env.step(action)
    env.render()