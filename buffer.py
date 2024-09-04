# Rollout buffer for PPOC based on https://github.com/lweitkamp/option-critic-pytorch

import numpy as np
import random
from collections import deque

class RolloutBuffer(object):
    def __init__(self, capacity, seed=42):
        self.capacity = capacity

        self.rng = random.SystemRandom(seed)
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, option, action, logp, val, reward, done):
        self.buffer.append((obs, option, action, logp, val, reward, done))

    def sample(self, batch_size):
        batch_start = np.arange(0, len(self.buffer), batch_size)
        indicies = np.arange(len(self.buffer), dtype=np.int64)
        self.rng.shuffle(indicies)
        batches = [indicies[i:i+batch_size] for i in batch_start]
        obs, option, action, logp, val, reward, done = zip(*self.buffer)
        return np.array(obs), np.array(option), np.array(action), np.array(logp),\
              np.array(val), np.array(reward), np.array(done), batches

    def __len__(self):
        return len(self.buffer)
    
    def clear_memory(self):
        self.buffer = deque(maxlen=self.capacity)

