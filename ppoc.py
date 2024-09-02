import os
import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

from math import exp
import numpy as np

from utils import to_tensor


class OptionCriticConv(nn.Module):
    def __init__(self,
                in_features,
                num_actions,
                num_options,
                temperature=1.0,
                eps_start=1.0,
                eps_min=0.1,
                eps_decay=int(1e6),
                eps_test=0.05,
                device='cpu',
                testing=False,
                chkpt_dir = 'tmp/ppoc'):

        super(OptionCriticConv, self).__init__()

        self.in_channels = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.magic_number = 7 * 7 * 64
        self.device = device
        self.testing = testing

        self.chkpt_dir = chkpt_dir

        self.temperature = temperature
        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test  = eps_test
        self.num_steps = 0
        
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.modules.Flatten(),
            nn.Linear(self.magic_number, 512),
            nn.ReLU()
        )

        self.Q            = nn.Linear(512, num_options)                 # Policy-Over-Options
        self.terminations = nn.Linear(512, num_options)                 # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 512, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)
    
    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()
    
    def get_terminations(self, state):
        return self.terminations(state).sigmoid() 

    def get_action(self, state, option):
        logits = state.data @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy
    
    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps
    
    def save_checkpoint(self, name):
        checkpoint_file = os.path.join(self.chkpt_dir, name)
        torch.save(self.state_dict(), checkpoint_file)
    
    def load_checkpoint(self, name):
        checkpoint_file = os.path.join(self.chkpt_dir, name)
        self.load_state_dict(torch.load(self.checkpoint_file))


class OptionCriticFeatures(nn.Module):
    def __init__(self,
                in_features,
                num_actions,
                num_options,
                temperature=1.0,
                eps_start=1.0,
                eps_min=0.1,
                eps_decay=int(1e6),
                eps_test=0.05,
                device='cpu',
                testing=False,
                gamma = 0.99,
                termination_reg = 0.01, 
                entropy_reg = 0.01,
                chkpt_dir = 'tmp/ppoc',
                fc1_dims = 32,
                fc2_dims = 64):

        super(OptionCriticFeatures, self).__init__()

        self.in_features = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.chkpt_dir = chkpt_dir

        self.temperature = temperature
        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test  = eps_test
        self.num_steps = 0

        self.gamma = gamma
        self.termination_reg = termination_reg
        self.entropy_reg = entropy_reg
        
        self.features = nn.Sequential(
            nn.Linear(in_features, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(fc2_dims, num_options),
            nn.Softmax(dim=-1)
            )
        self.terminations = nn.Linear(fc2_dims, num_options)                
        self.options_W = nn.Parameter(torch.zeros(num_options, fc2_dims, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.critic = nn.Sequential(
                nn.Linear(in_features, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, num_options)
        )

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state
    
    def get_option_dist(self, state):
        option_dist =  self.actor(state)
        option_dist = Categorical(option_dist)

        return option_dist
    
    def get_next_option(self, state):
        option_dist = self.get_option_dist(state)
        option = option_dist.sample()

        return option.item()
    
    def get_value(self, obs, option):
        if obs.ndim < 2:
            obs = to_tensor(obs).to(self.device)
            return self.critic(obs)[option].to(self.device)
        else:
            return self.critic(obs)[:, option].to(self.device)

    def get_Q(self, state):
        return self.Q(state)
    
    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        
        next_option = self.get_next_option(state)

        return bool(option_termination.item()), next_option
    
    def get_terminations(self, state):
        return self.terminations(state).sigmoid() 

    def get_action_dist(self, state, option):
        logits = state.data @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        return action_dist
    
    def choose_action(self, obs, option):
        state = self.get_state(to_tensor(obs))

        dist = self.get_action_dist(state, option)
        value = self.get_value(to_tensor(obs), option)
        action = dist.sample()

        action = dist.sample()
        logp = dist.log_prob(action)

        return action.item(), logp.item(), value.item()

    def save_checkpoint(self, name):
        print('... saving models ...')
        checkpoint_file = os.path.join(self.chkpt_dir, name)
        torch.save(self.state_dict(), checkpoint_file)
    
    def load_checkpoint(self, name):
        print('...loading models ...')
        checkpoint_file = os.path.join(self.chkpt_dir, name)
        self.load_state_dict(torch.load(checkpoint_file))


def critic_loss(option_critic, obs, options, advantage, values, batch):
    critic_value = option_critic.get_value(obs, options)

    critic_value = torch.squeeze(critic_value)

    returns = advantage[batch] + values[batch]
    critic_loss = (returns-critic_value)**2
    critic_loss = critic_loss.mean()

    return critic_loss

def actor_loss(option_critic, batch, states, old_probs, options, actions, dones, advantage, policy_clip, termination_cost, entropy_reg):
    dist = option_critic.get_action_dist(states, options)

    new_probs = dist.log_prob(actions)
    prob_ratio = new_probs.exp() / old_probs.exp()
    #prob_ratio = (new_probs - old_probs).exp()
    weighted_probs = advantage[batch] * prob_ratio
    weighted_clipped_probs = torch.clamp(prob_ratio, 1-policy_clip,
            1+policy_clip)*advantage[batch]
    policy_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

    entropy = dist.entropy()
    policy_loss -= entropy_reg * entropy.mean().detach()

    term_prob = option_critic.get_terminations(states)[:,options].detach()
    termination_loss = term_prob*(advantage[batch]+termination_cost)*(1-dones)
    termination_loss = termination_loss.mean()

    option_dist = option_critic.get_option_dist(states)
    option_logp = option_dist.log_prob(options).detach()
    option_loss = -option_logp*advantage[batch]
    option_loss = option_loss.mean()

    actor_loss = policy_loss + option_loss + termination_loss

    return actor_loss
