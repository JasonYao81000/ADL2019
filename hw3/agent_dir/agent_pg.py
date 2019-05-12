import numpy as np
np.random.seed(9487)
import torch
torch.cuda.manual_seed_all(9487)
import torch.nn as nn
import torch.nn.functional as F

from agent_dir.agent import Agent
from environment import Environment

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_num),
            nn.Softmax(dim=-1)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Memory
        self.actions = []
        self.states = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state, action=None, evaluate=False):
        # if evaluate is True then we also need to pass an action for evaluation
        # else we return a new action from distribution
        action_prob = self.action_layer(state)
        state_value = self.value_layer(state)
        
        action_distribution = torch.distributions.Categorical(action_prob)
        
        if not evaluate:
            action = action_distribution.sample()
            self.actions.append(action)
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)

        if evaluate:
            return action_distribution.entropy().mean()
        else:
            return action.item()

    def clearMemory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.ppo = args.ppo
        self.model = PolicyNet(state_dim=self.env.observation_space.shape[0],
                                action_num=self.env.action_space.n,
                                hidden_dim=64).cuda()
        self.old_model = PolicyNet(state_dim=self.env.observation_space.shape[0],
                                action_num=self.env.action_space.n,
                                hidden_dim=64).cuda() if self.ppo else self.model
        self.model_name = 'pg-ppo' if self.ppo else 'pg'
        
        if args.test_pg:
            self.load('./checkpoints/' + self.model_name + '-best.cpt')

        # discounted reward
        self.gamma = 0.99 
        # clip parameter for PPO
        self.eps_clip = 0.2

        # training hyperparameters
        self.num_episodes = 10000 # total training episodes
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 1000 # frequency to save the model
        self.ppo_update_epochs = 5 # ppo step epochs per update

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        self.old_model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.model.clearMemory()
        self.old_model.clearMemory()

    def make_action(self, state, test=False):
        state = torch.from_numpy(state).cuda().float()
        state = state if self.ppo else state.unsqueeze(0)
        action = self.old_model(state, evaluate=False)
        return action

    def update(self):
        R = 0
        rewards = []
        # Discount future rewards back to the present using gamma
        for r in self.old_model.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        
        # turn rewards to pytorch tensor and standardize
        rewards = torch.Tensor(rewards).cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        
        if self.ppo:
            # convert list in tensor
            old_states = torch.tensor(self.old_model.states).cuda().detach()
            old_actions = torch.tensor(self.old_model.actions).cuda().detach()
            old_logprobs = torch.tensor(self.old_model.logprobs).cuda().detach()
            
            # Optimize policy model for n epochs:
            for _ in range(self.ppo_update_epochs):
                # Evaluating old actions and values
                entropy = self.model(old_states, old_actions, evaluate=True)

                # Finding the ratio (pi_theta / pi_theta__old):
                logprobs = self.model.logprobs[0].cuda()
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss:
                state_values = self.model.state_values[0]
                advantages = rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 1.0 * nn.MSELoss()(state_values, rewards) - 0.01 * entropy
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                
                self.model.clearMemory()
        
            # Copy new weights into old policy model:
            self.old_model.load_state_dict(self.model.state_dict())
        else:
            # compute loss
            policy_loss = []
            for log_prob, reward in zip(self.old_model.logprobs, rewards):
                policy_loss.append(-log_prob * reward)
            
            # Update network weights
            self.optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).cuda().sum()
            policy_loss.backward()
            self.optimizer.step()
        
        self.old_model.clearMemory()

    def train(self):
        avg_reward = None # moving average of reward
        episode_rewards = []
        best_avg_reward = -9487
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while(not done):
                # Running the old policy model
                action = self.make_action(state)
                state_n, reward, done, _ = self.env.step(action)
                
                # Saving the state and reward
                self.old_model.states.append(state)
                self.old_model.rewards.append(reward)
                state = state_n
                episode_reward += reward

            # for logging 
            last_reward = np.sum(self.old_model.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            episode_rewards.append(episode_reward)
            
            # update model
            self.update()

            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
                np.save('./results/' + self.model_name + '_episode_rewards.npy', np.array(episode_rewards))
            
            # save the model
            if epoch % self.save_freq == 0:
                self.save('./checkpoints/' + self.model_name + '.cpt')

            # save the best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                self.save('./checkpoints/' + self.model_name + '-best.cpt')