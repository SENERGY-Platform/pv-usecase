import numpy as np

from statistics import median

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

 


class Policy(nn.Module):
    def __init__(self, state_size):
        super().__init__()

        self.fc1 = nn.Linear(state_size, 16)
        nn.init.kaiming_normal_(self.fc1, nonlinearity='relu')
        
        self.fc2 = nn.Linear(16, 2)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class Agent:
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        self.initial_weather_data = None
        self.power_list = []
        self.action = None
        self.log_prob = None
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
        self.action = m.sample()
        self.log_prob = m.log_prob(self.action)
    
    def get_reward(self, action, history_power_mean):
        median = median(self.power_list)
        
        if action==1:
            reward = median-history_power_mean
        elif action==0:
            reward = history_power_mean-median
            
        return reward
    
    def learn(self, history_power_mean, optimizer):
        self.reward = self.get_reward(self.action, history_power_mean)
        
        policy_loss = -self.reward*self.log_prob
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
