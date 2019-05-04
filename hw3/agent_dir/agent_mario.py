import numpy as np
np.random.seed(9487)
import torch
torch.cuda.manual_seed_all(9487)
from torch.distributions import Categorical
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm_

from a2c.environment_a2c import make_vec_envs
from a2c.storage import RolloutStorage
from a2c.actor_critic import ActorCritic

from collections import deque
import os

use_cuda = torch.cuda.is_available()

class AgentMario:
    def __init__(self, env, args):

        # Hyperparameters
        self.lr = 7e-4
        self.gamma = 0.9
        self.hidden_size = 512
        self.update_freq = 5
        self.n_processes = 16
        self.seed = 7122
        self.max_steps = 1e7
        self.grad_norm = 0.5
        self.entropy_weight = 0.05

        #######################    NOTE: You need to implement
        self.recurrent = True # <- ActorCritic._forward_rnn()
        #######################    Please check a2c/actor_critic.py
        
        self.display_freq = 4000
        self.save_freq = 100000
        self.save_dir = './checkpoints/'
        self.model_name = 'a2c'
        if args.test_mario:
            self.load(self.save_dir + self.model_name + '.cpt')

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        self.envs = env
        if self.envs == None:
            self.envs = make_vec_envs('SuperMarioBros-v0', self.seed,
                    self.n_processes)
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.obs_shape = self.envs.observation_space.shape
        self.act_shape = self.envs.action_space.n

        self.rollouts = RolloutStorage(self.update_freq, self.n_processes,
                self.obs_shape, self.act_shape, self.hidden_size) 
        self.model = ActorCritic(self.obs_shape, self.act_shape,
                self.hidden_size, self.recurrent).to(self.device)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.lr, 
                eps=1e-5)

        self.hidden = None
        self.init_game_setting()
   
    def _update(self):
        # TODO: Compute returns
        # R_t = reward_t + gamma * R_{t+1}
        R = torch.zeros(self.n_processes, 1).to(self.device)
        discounted_rewards = []
        for i in range(self.rollouts.n_steps):
            step = self.rollouts.step - i - 1
            reward = self.rollouts.rewards[step]
            R = reward + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.cat(discounted_rewards)

        # TODO:
        # Compute actor critic loss (value_loss, action_loss)
        # OPTIONAL: You can also maxmize entropy to encourage exploration
        # loss = value_loss + action_loss (- entropy_weight * entropy)
        value_loss = []
        action_loss = []
        entropys = []
        for step in range(self.rollouts.n_steps):
            obs = self.rollouts.obs[step]
            hiddens = self.rollouts.hiddens[step]
            masks = self.rollouts.masks[step]
            actions = self.rollouts.actions[step]
            values, action_probs, hiddens = self.model(obs, hiddens, masks)
            m = torch.distributions.Categorical(action_probs)
            log_probs = m.log_prob(actions)
            entropy = m.entropy().mean()
            advantages = discounted_rewards[step] - values
            value_loss.append(advantages.pow(2).mean())
            action_loss.append(-(advantages.detach() * log_probs).mean())
            entropys.append(entropy)
        value_loss = torch.cat(value_loss).sum()
        action_loss = torch.cat(action_loss).sum()
        entropys = torch.cat(entropys).sum()
        loss = value_loss + action_loss - self.entropy_weight * entropys

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        
        # TODO:
        # Clear rollouts after update (RolloutStorage.reset())
        self.rollouts.reset()

        return loss.item()

    def _step(self, obs, hiddens, masks):
        with torch.no_grad():
            # TODO:
            # Sample actions from the output distributions
            # HINT: you can use torch.distributions.Categorical
            values, action_probs, hiddens = self.model(obs, hiddens, masks)
            m = torch.distributions.Categorical(action_probs)
            actions = m.sample()

        obs, rewards, dones, infos = self.envs.step(actions.cpu().numpy())
        
        # TODO:
        # Store transitions (obs, hiddens, actions, values, rewards, masks)
        # You need to convert arrays to tensors first
        # HINT: masks = (1 - dones)
        obs = torch.from_numpy(obs).to(self.device)
        actions = actions.unsqueeze(-1)
        rewards = torch.from_numpy(rewards).to(self.device).unsqueeze(-1)
        dones = torch.from_numpy(np.array(dones).astype(np.float32)).to(self.device)
        masks = (torch.ones(dones.size()).to(self.device) - dones).unsqueeze(-1)
        self.rollouts.insert(obs, hiddens, actions, values, rewards, masks)

    def train(self):

        print('Start training')
        running_reward = deque(maxlen=10)
        episode_rewards = torch.zeros(self.n_processes, 1).to(self.device)
        total_steps = 0
        avg_rewards = []
        
        # Store first observation
        obs = torch.from_numpy(self.envs.reset()).to(self.device)
        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)
        
        while True:
            # Update once every n-steps
            for step in range(self.update_freq):
                self._step(
                    self.rollouts.obs[step],
                    self.rollouts.hiddens[step],
                    self.rollouts.masks[step])

                # Calculate episode rewards
                episode_rewards += self.rollouts.rewards[step]
                for r, m in zip(episode_rewards, self.rollouts.masks[step + 1]):
                    if m == 0:
                        running_reward.append(r.item())
                episode_rewards *= self.rollouts.masks[step + 1]

            loss = self._update()
            total_steps += self.update_freq * self.n_processes

            # Log & save model
            if len(running_reward) == 0:
                avg_reward = 0
            else:
                avg_reward = sum(running_reward) / len(running_reward)
            avg_rewards.append(avg_reward)

            if total_steps % self.display_freq == 0:
                print('Steps: %d/%d | Avg reward: %f'%
                        (total_steps, self.max_steps, avg_reward))
                np.save('./results/' + self.model_name + '_avg_rewards.npy', np.array(avg_rewards))
            
            if total_steps % self.save_freq == 0:
                self.save_model(self.model_name + '.cpt')
            
            if total_steps >= self.max_steps:
                break

    def save_model(self, filename):
        torch.save(self.model, os.path.join(self.save_dir, filename))

    def load_model(self, path):
        self.model = torch.load(path)

    def init_game_setting(self):
        if self.recurrent:
            self.hidden = torch.zeros(1, self.hidden_size).to(self.device)

    def make_action(self, observation, test=False):
        # TODO: Use you model to choose an action
        return action
