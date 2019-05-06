# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:20:02 2019

@author: hb2506
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent_dir.agent import Agent
from environment import Environment
from torch.distributions import Normal
torch.manual_seed(9487)

class Actor(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        mu = F.softmax(x, dim=1)
        sigma = F.softmax(x, dim=1)
        return mu, sigma

class Critic(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        env.seed(9487)
        self.actor_net = Actor(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        self.critic_net = Critic(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        if args.test_pg:
            self.load('pg.cpt')
            
        # discounted reward
        self.gamma = 0.99 
        
        # add
        self.counter = 0
        self.training_step = 0
        self.buffer = []
        self.buffer_capacity = 1000
        
        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress
        
        # optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=3e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=3e-3)
        self.eps = np.finfo(np.float32).eps.item()
        
        # saved rewards and actions
        self.rewards = []
        self.saved_actions = []
        self.saved_log_probs = []
    
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.actor_net.state_dict(), save_path)
        torch.save(self.critic_net.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        self.actor_net.load_state_dict(torch.load(load_path))
        self.critic_net.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards = []
        self.saved_actions = []
        self.saved_log_probs = []

    def make_action(self, state, test=False):
        # TODO:
        # Use your model to output distribution over actions and sample from it.
        # HINT: google torch.distributions.Categorical
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            mu, sigma = self.actor_net(state)
        m = Normal(mu, sigma)
        action = m.sample()
        action_log_prob = m.log_prob(action)
        action = action.clamp(-2, 2)
        return action.item(), action_log_prob.item()
    
    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()
    
    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter+=1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1
        # TODO:
        # discount your saved reward
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)    
        # TODO:
        # compute loss
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()


    def train(self):
        avg_reward = None # moving average of reward
        episode_reward = 0
        save_reward = list()
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                self.saved_actions.append(action)
                self.rewards.append(reward)
                
            save_reward.append(episode_reward)
            episode_reward = 0
            
            # for logging 
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            
            # update model
            self.update()

            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
            
            if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                self.save('pg.cpt')
                break
        np.save('PG_RewardCurve', save_reward)