import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from collections import deque
import random
from math import exp


class ReplayMemory:
    def __init__(self, max_len):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []

        self.max_len = max_len

    def __len__(self):
        return len(self.states)

    def store_memory(self, state, action, next_state, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

        if len(self.states) > self.max_len:
            self.states.pop(0)
            self.actions.pop(0)
            self.next_states.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)

    def sample(self, batch_size):
        n_states = len(self.states)
        batch = np.random.choice(n_states, batch_size, replace=False)

        return np.array(self.states), np.array(self.actions), np.array(self.next_states), \
            np.array(self.rewards), np.array(self.dones), batch

    
class DeepNeuralNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=64, fc2_dims=64, chkpt_dir='tmp/dqn'):
        super(DeepNeuralNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'nn_torch_dqn')
        self.nn = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
            )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, amsgrad=True)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        return self.nn(state)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            tau=0.005, batch_size=64, eps_start=0.9, eps_end=0.05, eps_decay=1000):
        self.gamma = gamma
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.num_steps = 0
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]

        self.policy_network = DeepNeuralNetwork(n_actions, input_dims, alpha)
        self.memory = ReplayMemory(10000)
       
    def remember(self, state, action, next_state, reward, done):
        self.memory.store_memory(state, action, next_state, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.policy_network.save_checkpoint()


    def load_models(self):
        print('... loading models ...')
        self.policy_network.load_checkpoint()

    @property
    def epsilon(self):
        eps = self.eps_end + (self.eps_start - self.eps_end) * exp(-self.num_steps / self.eps_decay)
        self.num_steps += 1
        return eps
        

    def choose_action(self, observation):
        sample = random.random()
        state = T.tensor([observation]).to(self.policy_network.device)
        if sample > self.epsilon:
            with T.no_grad():
                actions = self.policy_network.forward(state)
            return T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
            return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states_arr, actions_arr, next_states_arr, rewards_arr, dones_arr, batch = self.memory.sample(self.batch_size)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        states = T.tensor(states_arr[batch], dtype=T.float).to(self.policy_network.device)
        actions = actions_arr[batch]
        next_states = T.tensor(next_states_arr[batch], dtype=T.float).to(self.policy_network.device)
        rewards = T.tensor(rewards_arr[batch], dtype=T.float).to(self.policy_network.device)
        dones = T.tensor(dones_arr[batch]).to(self.policy_network.device)
        dones = dones.long()

        q_eval = self.policy_network.forward(states)[batch_index, actions]
        q_next = self.policy_network.forward(next_states)

        q_target = rewards + self.gamma * T.max(q_next, dim=1)[0] * (1 - dones)

        self.policy_network.zero_grad()
        loss = self.policy_network.loss(q_target, q_eval).to(self.policy_network.device)
        loss.backward()
        self.policy_network.optimizer.step()



