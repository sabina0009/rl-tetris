import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Bernoulli

from collections import deque
import random
from math import exp


class ReplayMemory:
    def __init__(self, max_len):
        self.states = []
        self.options = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []

        self.max_len = max_len

    def __len__(self):
        return len(self.states)

    def store_memory(self, state, option, action, next_state, reward, done):
        self.states.append(state)
        self.options.append(option)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

        if len(self.states) > self.max_len:
            self.states.pop(0)
            self.options.pop(0)
            self.actions.pop(0)
            self.next_states.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)

    def sample(self, batch_size):
        n_states = len(self.states)
        batch = np.random.choice(n_states, batch_size, replace=False)

        return np.array(self.states), np.array(self.options), np.array(self.actions), \
            np.array(self.next_states), np.array(self.rewards), np.array(self.dones), batch

    
class DeepQNetwork(nn.Module):
    def __init__(self, n_actions, n_options, input_dims, alpha,
            fc1_dims=64, fc2_dims=64, chkpt_dir='tmp/dqn'):
        super(DeepQNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'nn_torch_dqn')
        self.features = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
            )
        self.Q = nn.Linear(fc2_dims, n_options)
        self.terminations =  nn.Linear(fc2_dims, n_options)
        self.options_W = nn.Parameter(T.zeros(n_options, fc2_dims, n_actions))
        self.options_b = nn.Parameter(T.zeros(n_options, n_actions))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, amsgrad=True)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, obs):
        obs = np.asarray(obs)
        obs = T.from_numpy(obs).float()
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)

        return state
    
    def get_q(self, state):
        return self.Q(state)
    
    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample().item()
        q_val = self.get_q(state)
        next_greedy_option = q_val.argmax(dim=-1).item()

        return bool(option_termination), next_greedy_option

    def get_terminations(self, state):
        return self.terminations(state).sigmoid()

    def get_action_logits(self, state, current_option): 
        return state.detach() @ self.options_W[current_option] + self.options_b[current_option]

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, n_options, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            tau=0.005, batch_size=64, eps_start=0.9, eps_end=0.05, eps_decay=1000, temperature = 1,
            update_frequency = 4, termination_reg=0.01, entropy_reg = 0.01):
        self.gamma = gamma
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.temperature = temperature
        self.update_frequency = update_frequency
        self.termination_reg = termination_reg
        self.entropy_reg = entropy_reg
        self.num_steps = 0
        self.n_actions = n_actions
        self.n_options = n_options
        self.action_space = [i for i in range(n_actions)]

        self.policy_network = DeepQNetwork(n_actions, n_options, input_dims, alpha)
        self.target_network = DeepQNetwork(n_actions, n_options, input_dims, alpha)
        self.memory = ReplayMemory(10000)
       
    def remember(self, state, option, action, next_state, reward, done):
        self.memory.store_memory(state, option, action, next_state, reward, done)

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

    def choose_action(self, observation, current_option, option_termination, next_greedy_option):

        state = self.policy_network.forward(observation)

        if option_termination:
            if np.random.rand() > self.epsilon:
                current_option = next_greedy_option
            else:
                current_option = np.random.choice(self.n_options)

        logits = self.policy_network.get_action_logits(state, current_option)
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()
            
        return current_option, action.item(), logp, entropy

    def learn(self, obs, current_option, logp, entropy, reward, done, next_obs):
        if len(self.memory) < self.batch_size:
            return
        
        state = self.policy_network.forward(obs)
        next_state = self.policy_network.forward(next_obs)
        next_state_prime = self.target_network.forward(next_obs)

        q_val = self.policy_network.get_q(state).detach().squeeze()
        terminations = self.policy_network.get_terminations(state)
        next_terminations = self.policy_network.get_terminations(next_state)
        q_next_prime = self.target_network.get_q(next_state_prime).detach().squeeze()

        option_term_prob = terminations[:, current_option]
        next_option_term_prob = next_terminations[:, current_option].detach()

        actor_q_target = reward + (1 - done) * self.gamma * \
            ((1 - next_option_term_prob) * q_next_prime[current_option] + next_option_term_prob  * q_next_prime.max(dim=-1)[0])

        termination_loss = option_term_prob * (q_val[current_option].detach() - q_val.max(dim=-1)[0].detach() + self.termination_reg) * (1 - done)
        policy_loss = -logp * (actor_q_target.detach() - q_val[current_option]) - self.entropy_reg * entropy
        actor_loss = termination_loss + policy_loss
        loss = actor_loss
        
        if self.num_steps % self.update_frequency == 0:
            states_arr, options_arr, _, next_states_arr, rewards_arr, dones_arr, batch = self.memory.sample(self.batch_size)

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            obss = T.tensor(states_arr[batch], dtype=T.float).to(self.policy_network.device)
            options = options_arr[batch]
            next_obss = T.tensor(next_states_arr[batch], dtype=T.float).to(self.policy_network.device)
            rewards = T.tensor(rewards_arr[batch], dtype=T.float).to(self.policy_network.device)
            dones = T.tensor(dones_arr[batch]).to(self.policy_network.device)
            dones = dones.long()

            states = self.policy_network.forward(obss).squeeze(0)
            q_vals = self.policy_network.get_q(states)

            next_states = self.policy_network.forward(next_obss).squeeze(0)
            next_termination_probs = self.policy_network.get_terminations(next_states)

            next_states_prime = self.target_network.forward(next_obss).squeeze(0)
            q_next_prime = self.target_network.get_q(next_states_prime)

            next_option_term_probs = next_termination_probs[batch_index, options]

            critic_q_target = rewards + self.gamma * (1 - dones) * \
                ((1 - next_option_term_probs) * q_next_prime[batch_index, options] + next_option_term_probs * q_next_prime.max(dim=-1)[0]) 

            critic_loss = (q_vals[batch_index, options] - critic_q_target.detach()).pow(2).mul(0.5).mean()
            loss += critic_loss

        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()

        self.num_steps += 1

    def update_target_policy(self):
        target_net_state_dict = self.target_network.state_dict()
        policy_net_state_dict = self.policy_network.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau \
                + target_net_state_dict[key]*(1-self.tau)
        
        self.target_network.load_state_dict(target_net_state_dict)

