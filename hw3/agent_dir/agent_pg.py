import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent_dir.agent import Agent
from environment import Environment
from torch.distributions import Categorical
#torch.manual_seed(9487)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PolicyNet_PPO(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet_PPO, self).__init__()
        self.affine = nn.Linear(state_dim, hidden_dim)
        
        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_num),
                nn.Softmax()
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
                )
        
        # Memory:
        self.actions = []
        self.states = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        
    def forward(self, state, action=None, evaluate=False):
        # if evaluate is True then we also need to pass an action for evaluation
        # else we return a new action from distribution
        if not evaluate:
            state = torch.from_numpy(state).float().to(device)
        
        state_value = self.value_layer(state)
        
        action_probs = self.action_layer(state)
        action_distribution = Categorical(action_probs)
        
        if not evaluate:
            action = action_distribution.sample()
            self.actions.append(action)
            
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        if evaluate:
            return action_distribution.entropy().mean()
        
        if not evaluate:
            return action.item()
        
    def clearMemory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
        
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
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
        self.ppo = args.ppo
        self.lr = 1e-3
        self.betas = (0.9, 0.999)
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.display_freq = 10
        self.eps = np.finfo(np.float32).eps.item()
        self.K_epochs = 1
        self.num_episodes = 10000
        self.n_update = 10
        self.reward_thred = 200
        
        if self.ppo:
            self.policy = PolicyNet_PPO(state_dim = self.env.observation_space.shape[0],
                                   action_num= self.env.action_space.n,
                                   hidden_dim=64).to(device)
            self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                                  lr=self.lr, betas=self.betas)
            self.policy_old = PolicyNet_PPO(state_dim = self.env.observation_space.shape[0],
                                   action_num= self.env.action_space.n,
                                   hidden_dim=64).to(device)    
            if args.test_pg:
                self.load('./pg/pg_ppo.cpt')
            self.MseLoss = nn.MSELoss()
        else:
            self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                                   action_num= self.env.action_space.n,
                                   hidden_dim=64)
            if args.test_pg:
                self.load('./pg/pg.cpt')
            # discounted reward       
            # optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)        
            # saved rewards and actions
            self.rewards = []
            self.saved_actions = []
            self.saved_log_probs = []
    
    
    def save(self, save_path):
        print('save model to', save_path)
        if self.ppo:
            torch.save(self.policy.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        if self.ppo:
            self.policy_old.load_state_dict(torch.load(load_path))
        else:
            self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards = []
        self.saved_actions = []
        self.saved_log_probs = []

    def make_action(self, state, test=False):
        # TODO:
        # Use your model to output distribution over actions and sample from it.
        # HINT: google torch.distributions.Categorical
        if self.ppo:
            action = self.policy_old(state)
            return action
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            probs = self.model(state)
            m = Categorical(probs)
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
            return action.item()

    def update(self):
        # TODO:
        # discount your saved reward
        if self.ppo:
            rewards = []
            discounted_reward = 0
            for reward in reversed(self.policy_old.rewards):
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)           
            # Normalizing the rewards:
            rewards = torch.tensor(rewards).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)            
            # convert list in tensor
            old_states = torch.tensor(self.policy_old.states).to(device).detach()
            old_actions = torch.tensor(self.policy_old.actions).to(device).detach()
            old_logprobs = torch.tensor(self.policy_old.logprobs).to(device).detach()            
            # Optimize policy for K epochs:
            for _ in range(self.K_epochs):
                # Evaluating old actions and values :
                dist_entropy = self.policy(old_states, old_actions, evaluate=True)               
                # Finding the ratio (pi_theta / pi_theta__old):
                logprobs = self.policy.logprobs[0].to(device)
                ratios = torch.exp(logprobs - old_logprobs.detach())                    
                # Finding Surrogate Loss:
                state_values = self.policy.state_values[0].to(device)
                advantages = rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()               
                self.policy.clearMemory()            
            self.policy_old.clearMemory()            
            # Copy new weights into old policy:
            self.policy_old.load_state_dict(self.policy.state_dict()) 
        else:
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
            avg_reward = None
            episode_reward = 0
            save_reward = list()
            for epoch in range(self.num_episodes):
                state = self.env.reset()
                done = False
                if self.ppo:
                    running_reward = []
                    while(not done):
                        # Running policy_old:
                        action = self.make_action(state)
                        state_n, reward, done, _ = self.env.step(action)
                        
                        # Saving state and reward:
                        self.policy_old.states.append(state)
                        self.policy_old.rewards.append(reward)
                        
                        state = state_n
                        episode_reward += reward
                        running_reward.append(reward)
                    last_reward = np.sum(running_reward)
                else:
                    self.init_game_setting()
                    while(not done):
                        action = self.make_action(state)
                        state, reward, done, _ = self.env.step(action)
                        episode_reward += reward
                        self.saved_actions.append(action)
                        self.rewards.append(reward)
                    last_reward = np.sum(self.rewards)
                
                avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
                save_reward.append(episode_reward)
                episode_reward = 0
                # update after n episodes
                if epoch % self.n_update:
                    self.update()
                
                if self.ppo:
                    # log
                    np.save('PG_PPO_RewardCurve', save_reward)
                    if avg_reward > self.reward_thred:
                        self.save('pg_ppo.cpt')
                        break
                else:
                    np.save('PG_RewardCurve', save_reward)
                    if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                        self.save('pg.cpt')
                        break
                    
                if epoch % self.display_freq == 0:
                    print('Epochs: %d/%d | Avg reward: %f '%
                           (epoch, self.num_episodes, avg_reward))
                    running_reward = 0
                
                