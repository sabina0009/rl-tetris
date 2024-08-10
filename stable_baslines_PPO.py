import numpy as np
import gymnasium
import gym_simpletetris
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from utils import plot_learning_curve

log_dir = "tmp/sb3_log/log"
filename = "sb3-tetris-DQN"
figure_file = f'plots/{filename}.png'


env = gymnasium.make('SimpleTetris-v0', reward_step=True)
env = Monitor(env, log_dir)

model = DQN("MlpPolicy", env, learning_rate=0.0003, batch_size=64, gamma=0.99, verbose=1)
model.learn(total_timesteps=100000)

score_history = env.get_episode_rewards()
np.savetxt(f'results/{filename}.txt', score_history, fmt='%d')
x = [i+1 for i in range(len(score_history))]
plot_learning_curve(x, score_history, figure_file)

vec_env = model.get_env()
obs = vec_env.reset()
done = False
while not done:
    action, state_ = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    env.render("human")