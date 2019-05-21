import torch
from torch.distributions import Categorical
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm_

from a2c.environment_a2c import make_vec_envs
from a2c.storage import RolloutStorage
from a2c.actor_critic import ActorCritic

import math
import random
from collections import deque
import os
import numpy as np

torch.cuda.manual_seed_all(9487)
random.seed(9487)
#np.random.seed(9487)

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
        self.save_freq = 5
        self.save_dir = './mario/'

        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01

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
        
        if args.test_mario:
            self.load_model(self.save_dir + 'mario.cpt')

        self.hidden = None
        self.init_game_setting()
        
#        self.steps = 0 # num. of passed steps. this may be useful in controlling exploration
#        self.EPS_START = 0.9
#        self.EPS_END = 0.05
#        self.EPS_DECAY = self.max_steps / 1000
   
    def _update(self):
        # TODO: Compute returns
        # R_t = reward_t + gamma * R_{t+1}
        next_value = self.make_action(self.rollouts)
        self.rollouts.returns[-1] = next_value
        for step in reversed(range(self.rollouts.rewards.size(0))):
            self.rollouts.returns[step] = self.rollouts.returns[step + 1] * \
                self.gamma * self.rollouts.masks[step + 1] + self.rollouts.rewards[step]

        # TODO:
        # Compute actor critic loss (value_loss, action_loss)
        # OPTIONAL: You can also maxmize entropy to encourage exploration
        # loss = value_loss + action_loss (- entropy_weight * entropy)
        obs_shape = self.rollouts.obs.size()[2:]
        action_shape = self.rollouts.actions.size()[-1]
        num_steps, num_processes, _ = self.rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.evaluate_actions(
            self.rollouts.obs[:-1].view(-1, *obs_shape),
            self.rollouts.hiddens[0].view(-1, self.hidden_size),
            self.rollouts.masks[:-1].view(-1, 1),
            self.rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = self.rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()
        loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

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
#        sample = random.random()
#        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
#            math.exp(-1. * self.steps / self.EPS_DECAY)
#
#        # if explore, you randomly samples one action
#        # else, use your model to predict action
#        if sample > eps_threshold:
        with torch.no_grad():
            # TODO:
            # Sample actions from the output distributions
            # HINT: you can use torch.distributions.Categorical
            value_preds, actions, self.hidden = self.model(obs, hiddens, masks)
            dist = Categorical(actions)
            actions = dist.sample()
#        else:
#            with torch.no_grad():
#                value_preds, actions, self.hidden = self.model(obs, hiddens, masks)
#            actions = torch.tensor([random.randrange(self.act_shape) for _ in range(self.n_processes)], dtype=torch.long)

        obs, rewards, dones, infos = self.envs.step(actions.cpu().numpy())
        
        # TODO:
        # Store transitions (obs, hiddens, actions, values, rewards, masks)
        # You need to convert arrays to tensors first
        # HINT: masks = (1 - dones)
        
#        for info in infos:
#            if 'episode' in info.keys():
#                episode_rewards.append(info['episode']['r'])

        # If done then clean the history of observations.
        obs= torch.from_numpy(obs).to(self.device)
        rewards= torch.from_numpy(rewards).to(self.device).unsqueeze(-1)
        actions = actions.unsqueeze(-1)
        masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                   for done_ in dones])
        
        self.rollouts.insert(obs, self.hidden, actions, value_preds, rewards, masks)
        
        
    def train(self):

        print('Start training')
        running_reward = deque(maxlen=10)
        episode_rewards = torch.zeros(self.n_processes, 1).to(self.device)
        total_steps = 0

        max_avg_reward = 0
        save_reward = list()
        
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
                    
            save_reward.append(avg_reward)

            if total_steps % self.display_freq == 0:
                print('Steps: %d/%d | Avg reward: %f'%
                        (total_steps, self.max_steps, avg_reward))
            
#            if max_avg_reward == 0:
#                max_avg_reward = avg_reward
#                self.save_model('mario.cpt')
#                print("save_model")
#            else:
#                if avg_reward >= max_avg_reward:
#                    max_avg_reward = avg_reward
#                    self.save_model('mario.cpt')
#                    print("save_model")
                    
            if total_steps % self.save_freq == 0:
                np.save('mario_save_reward', save_reward)
                self.save_model('mario.cpt')
                
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
        if test:
            with torch.no_grad():
                obs = torch.from_numpy(observation).to(self.device).unsqueeze(0)
                masks = torch.ones(1, 1).to(self.device)
                values, action_probs, self.hidden = self.model(obs, self.hidden, masks)
                m = torch.distributions.Categorical(action_probs)
                action = m.sample()
                # action = action_probs.argmax(dim=-1)
            return action.cpu().numpy()[0]
        # TODO: Use you model to choose an action
        with torch.no_grad():
            action, _, _ = self.model(observation.obs[-1], observation.hiddens[-1], observation.masks[-1])

        return action
    
    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.model(inputs, rnn_hxs, masks)
        dist = Categorical(actor_features)

        action_log_probs = dist.log_prob(action.squeeze(-1))
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
