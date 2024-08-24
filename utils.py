import gym
import gymnasium
import gym_simpletetris
import numpy as np
import torch

from gym.wrappers import AtariPreprocessing, TransformReward
from gym.wrappers import FrameStack as FrameStack_

from fourrooms import Fourrooms

import matplotlib.pyplot as plt


class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]


class FrameStack(FrameStack_):
    def __init__(self, env, k):
        FrameStack_.__init__(self, env, k)

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

def make_env(env_name):

    if env_name == 'fourrooms':
        return Fourrooms(), False
    
    if env_name == 'SimpleTetris-v0':
        env = gymnasium.make('SimpleTetris-v0', reward_step=True)
        return env, False 

    env = gym.make(env_name)
    is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
    if is_atari:
        env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, terminal_on_life_loss=True)
        env = TransformReward(env, lambda r: np.clip(r, -1, 1))
        env = FrameStack(env, 4)
    return env, is_atari

def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def plot_multiple_learning_curves(figure_file, *results_files):
    colours = ['b','g', 'r', 'c', 'm', 'y', 'k']
    j=0
    for file in results_files:
        scores = np.loadtxt(file, dtype=int)
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        x = [n for n in range(len(scores))]
        plt.plot(x, running_avg, color=colours[j], label=f'{file[8:-4]}')
        plt.title('Running average of previous 100 scores')
        plt.legend()
        j += 1
    plt.savefig(figure_file)


