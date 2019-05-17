import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_size, recurrent):
        super(ActorCritic, self).__init__()
        
        self.recurrent = recurrent
        self.hidden_size = hidden_size
        
        self.head = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 7 * 7, hidden_size),
            nn.ReLU()
        )

        if self.recurrent:
            self.rnn = nn.GRU(hidden_size, hidden_size)
        
        self.actor = nn.Linear(hidden_size, act_shape)
        self.critic = nn.Linear(hidden_size, 1)
        
        self.reset_parameters()

    def reset_parameters(self):
        
        def _weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.head.apply(_weights_init)

        if self.recurrent:
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
        
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight)
        nn.init.constant_(self.actor.bias, 0)
        nn.init.constant_(self.critic.bias, 0)

    def _forward_rnn(self, x, hiddens, masks):
        '''
        Args:
            x: observations -> (n_steps * n_processes, hidden_size)
            hiddens: hidden states of 1st step -> (n_processes, hidden_size)
            masks: whether to reset hidden state -> (n_steps * n_processes, 1)
        Returns:
            x: outputs of RNN -> (n_steps * n_processes, hidden_size)
            hiddens: hidden states of last step -> (n_processes, hidden_size)
        '''
        if x.size(0) == hiddens.size(0):
            # n_steps = 1
            x, hiddens = self.rnn(x.unsqueeze(0), (hiddens * masks).unsqueeze(0))
            x = x.squeeze(0)
            hiddens = hiddens.squeeze(0)
        else:
            # TODO
            # step 1: Unflatten the tensors to (n_steps, n_processes, -1)
            n_processes = hiddens.size(0)
            n_steps = int(x.size(0) / n_processes)
            x = x.view(n_steps, n_processes, x.size(1))
            masks = masks.view(n_steps, n_processes, masks.size(1))
            
            # step 2: Run a for loop through time to forward rnn
            # HINT: You must set hidden states to zeros when masks == 0 in the loop 
            outputs = []
            for i in range(n_steps):
                output, hiddens = self.rnn(x[i, :, :].unsqueeze(0), (hiddens * masks[i, :, :]).unsqueeze(0))
                hiddens = hiddens.squeeze(0)
                outputs.append(output.squeeze(0))
            
            # step 3: Flatten the outputs
            x = torch.cat(outputs, dim=0)
            x = x.view(n_steps * n_processes, -1)
        
        return x, hiddens

    def forward(self, inputs, hiddens, masks):
        x = self.head(inputs / 255.0)
        if self.recurrent:
            x, hiddens = self._forward_rnn(x, hiddens, masks)
        
        values = self.critic(x)
        action_probs = self.actor(x)
        action_probs = F.softmax(action_probs, -1)
        
        return values, action_probs, hiddens

        
