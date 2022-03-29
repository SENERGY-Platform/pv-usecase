import numpy as np

from statistics import median

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

 


class Policy(nn.Module):
    def __init__(self, state_size, action_size=2):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        nn.init.kaiming_normal_(self.fc1, nonlinearity='relu')

        self.fc2 = nn.Linear(64, 64)
        nn.init.kaiming_normal_(self.fc2, nonlinearity='relu')
        
        self.fc3 = nn.Linear(64, action_size)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class Agent:
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        self.initial_weather_data = None
        self.power_list = []
        self.reward = -np.Inf
        
    def save_weather_data(weather_data):
        self.initial_weather_data = weather_data
        
    def update_power_list(new_power_value):
        self.power_list.append(new_power_value)
        
    def act(self, policy):
        state = self.initial_weather_data
        state = torch.from_numpy(state).float().unsqueeze(0)
        if self.use_cuda:
            state = state.cuda()
        probs = policy(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def get_reward(self, action, history_power_mean):
        median = median(self.power_list)
        
        if action==1:
            reward = median-history_power_mean
        elif action==0:
            reward = history_power_mean-median
            
        return reward
    
    def learn(self, policy, history_power_mean, optimizer):
        action, log_prob = self.act(policy)
        self.reward = self.get_reward(action, history_power_mean)
        
        policy_loss = -self.reward*log_prob
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()