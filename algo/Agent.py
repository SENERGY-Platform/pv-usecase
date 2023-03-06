import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

 


class Policy(nn.Module):
    def __init__(self, state_size):
        super().__init__()

        self.fc1 = nn.Linear(state_size, 16)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        
        self.fc2 = nn.Linear(16, 2)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        y = F.softmax(self.fc2(x), dim=1)
        return y


class Agent:
    def __init__(self):
        self.initial_weather_data = None
        self.initial_time = None
        self.power_list = []
        self.action = None
        self.log_prob = None
        self.reward = None
        
    def save_weather_data(self,weather_data):
        self.initial_weather_data = weather_data
        
    def update_power_list(self,time, new_power_value):
        self.power_list.append((time, new_power_value))
        
    def act(self, policy):
        state = self.initial_weather_data
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = policy(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        log_prob_action = m.log_prob(action)
        return  action, log_prob_action

    def get_reward(self, action, history):
        agents_power_mean = sum([power_value for _, power_value in self.power_list])/len([power_value for _, power_value in self.power_list])
        history_mean = sum(history)/len(history)
        
        if action.item()==1:    # 'YES'
            reward = agents_power_mean-history_mean
        elif action.item()==0:  # 'NO'
            reward = history_mean-agents_power_mean
            
        return reward
    
    def learn(self, reward, log_prob, optimizer):
        policy_loss = -reward*log_prob
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
