"""
   Copyright 2022 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

__all__ = ("Operator", )

import util
from . import aux_functions, Agent
from collections import deque
import pickle
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import os
import random
import astral
from astral import sun
import pytz
import datetime
import matplotlib.pyplot as plt


class Operator(util.OperatorBase):
    def __init__(self, energy_src_id, weather_src_id, lat=51.34, long=12.38, power_history_start_stop='2', buffer_len='48', p_1='1', p_0='1', history_modus='daylight', power_td=0.17, weather_dim=6, data_path="data"):
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        
        self.lat = lat
        self.long = long
        self.observer = astral.Observer(latitude=self.lat, longitude=self.long)

        self.energy_src_id = energy_src_id
        self.weather_src_id = weather_src_id

        self.weather_same_timestamp = []

        self.power_history_start_stop = int(power_history_start_stop)

        self.buffer_len = int(buffer_len)
        self.history_power_len = int(10000/float(power_td)) # power_td is the time difference between two consecutive power values in minutes
        self.replay_buffer = deque(maxlen=self.buffer_len)
        self.power_history = deque(maxlen=self.history_power_len) 
        self.daylight_power_history = deque(maxlen=int(self.history_power_len/2))
        self.history_modus = history_modus

        self.p_1 = int(p_1)
        self.p_0 = int(p_0)
        
        self.agents = []
        self.policy = Agent.Policy(state_size=weather_dim) # If we keep track of time, temp, humidity, uv-index, precipitation and clouds we have weather_dim=6.
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)

        self.power_lists = []
        self.actions = []
        self.rewards = []
        self.weather_data = []
        self.agents_data = []

        self.power_lists_file = f'{data_path}/{self.energy_src_id}_{self.weather_src_id}_power_lists_{self.p_1}_{self.p_0}_{self.history_modus}_{self.power_history_start_stop}.pickle'
        self.actions_file = f'{data_path}/{self.energy_src_id}_{self.weather_src_id}_actions_{self.p_1}_{self.p_0}_{self.history_modus}_{self.power_history_start_stop}.pickle'
        self.rewards_file = f'{data_path}/{self.energy_src_id}_{self.weather_src_id}_rewards_{self.p_1}_{self.p_0}_{self.history_modus}_{self.power_history_start_stop}.pickle'
        self.weather_file = f'{data_path}/{self.energy_src_id}_{self.weather_src_id}_weather_{self.p_1}_{self.p_0}_{self.history_modus}_{self.power_history_start_stop}.pickle'
        self.agents_data_file = f'{data_path}/{self.energy_src_id}_{self.weather_src_id}_agents_data_{self.p_1}_{self.p_0}_{self.history_modus}_{self.power_history_start_stop}.pickle'

        self.model_file = f'{data_path}/{self.energy_src_id}_{self.weather_src_id}_model_{self.p_1}_{self.p_0}_{self.history_modus}_{self.power_history_start_stop}.pt'

        self.power_forecast_plot_file = f'{data_path}/{self.energy_src_id}_{self.weather_src_id}_histogram_{self.p_1}_{self.p_0}_{self.history_modus}_{self.power_history_start_stop}.png'

        #if os.path.exists(self.model_file):
        #    self.policy.load_state_dict(torch.load(self.model_file))

    def run_new_weather(self, new_weather_data):
        new_weather_array = aux_functions.preprocess_weather_data(new_weather_data)
        self.weather_data.append(new_weather_array)
        new_weather_input = np.mean(new_weather_array, axis=0)

        self.policy.eval()      # The current policy is used for prediction.
        with torch.no_grad():
            input = torch.from_numpy(new_weather_input).float().unsqueeze(0)
            output = torch.argmax(self.policy(input)).item()
        self.policy.train()

        self.agents.append(Agent.Agent())
        if len(self.replay_buffer)==self.buffer_len:
            random.shuffle(self.replay_buffer)
            for agent in self.replay_buffer:
                agent.action, agent.log_prob = agent.act(self.policy)
                if self.history_modus=='all':
                    agent.reward = agent.get_reward(agent.action, self.p_1, self.p_0, self.power_history)
                elif self.history_modus=='daylight':
                    agent.reward = agent.get_reward(agent.action, self.p_1, self.p_0, self.daylight_power_history)
                agent.learn(agent.reward, agent.log_prob, self.optimizer)

        torch.save(self.policy.state_dict(), self.model_file)
        with open(self.weather_file, 'wb') as f:
            pickle.dump(self.weather_data, f)
        
        newest_agent = self.agents[-1]
        newest_agent.save_weather_data(new_weather_input)
        newest_agent.initial_time = pd.to_datetime(new_weather_data[0]['weather_time'])
    
        if output==0:
            return {"value": 0}
        elif output==1:
            return {"value": 1}

    def run_new_power(self, new_power_data):
        time, new_power_value = aux_functions.preprocess_power_data(new_power_data)
        if new_power_value != None:
            self.power_history.append(new_power_value)

        for i, agent in enumerate(self.agents):
            if agent.initial_time + pd.Timedelta(2,'hours') >= time:
                if new_power_value != None:
                    agent.update_power_list(time, new_power_value)
            elif agent.initial_time + pd.Timedelta(2,'hours') < time:
                oldest_agent = self.agents.pop(i)
                self.agents_data.append(oldest_agent)
                if len(self.replay_buffer)==self.buffer_len and oldest_agent.power_list != []:
                    oldest_agent.action, oldest_agent.log_prob = oldest_agent.act(self.policy)
                    if self.history_modus=='all':
                        oldest_agent.reward = oldest_agent.get_reward(oldest_agent.action, self.p_1, self.p_0, self.power_history)
                    elif self.history_modus=='daylight':
                        oldest_agent.reward = oldest_agent.get_reward(oldest_agent.action, self.p_1, self.p_0, self.daylight_power_history)
                    oldest_agent.learn(oldest_agent.reward, oldest_agent.log_prob, self.optimizer)
                    self.power_lists.append(oldest_agent.power_list)
                    self.actions.append(oldest_agent.action)
                    self.rewards.append(oldest_agent.reward)
                if oldest_agent.power_list != []:
                    self.replay_buffer.append(oldest_agent) 

        sunrise = pd.to_datetime(sun.sunrise(self.observer, date=time, tzinfo='UTC'))
        sunset = pd.to_datetime(sun.sunset(self.observer, date=time, tzinfo='UTC')) 
        if (sunrise+pd.Timedelta(self.power_history_start_stop, 'hours')<time) and (time+pd.Timedelta(self.power_history_start_stop, 'hours')<sunset):
            if new_power_value != None:
               self.daylight_power_history.append(new_power_value)

        with open(self.power_lists_file, 'wb') as f:
            pickle.dump(self.power_lists, f)
        with open(self.actions_file, 'wb') as f:
            pickle.dump(self.actions, f)
        with open(self.rewards_file, 'wb') as f:
            pickle.dump(self.rewards, f)
        with open(self.agents_data_file, 'wb') as f:
            pickle.dump(self.agents_data, f)


    def create_power_forecast(self, new_weather_data):
        self.policy.eval() 
        power_forecast = []
        new_weather_array = aux_functions.preprocess_weather_data(new_weather_data)
        new_weather_forecasted_for = [pd.to_datetime(datapoint['forecasted_for']).tz_localize(None) for datapoint in new_weather_data]
        for i in range(0,len(new_weather_array),3):
            new_weather_input = np.mean(new_weather_array[i:i+3], axis=0)
            with torch.no_grad():
                input = torch.from_numpy(new_weather_input).float().unsqueeze(0)
                probability = float(self.policy(input).squeeze()[1])
            power_forecast.append((new_weather_forecasted_for[i],probability))
        fig, ax = plt.subplots(1,1,figsize=(30,30))
        ax.plot([timestamp for timestamp,_ in power_forecast],[probability for _,probability in power_forecast])
        plt.savefig(self.power_forecast_plot_file)
        self.policy.train()
        return power_forecast
        


    def run(self, data, selector):
        if os.getenv("DEBUG") is not None and os.getenv("DEBUG").lower() == "true":
            print(selector + ": " + str(data))
        if selector == 'weather_func':
            if len(self.weather_same_timestamp)<47:
                self.weather_same_timestamp.append(data)
            elif len(self.weather_same_timestamp)==47:
                self.weather_same_timestamp.append(data)
                new_weather_data = self.weather_same_timestamp
                output = self.run_new_weather(new_weather_data[0:3])
                power_forecast = self.create_power_forecast(new_weather_data)
                self.weather_same_timestamp = []
                return [{'timestamp':timestamp.strftime('%Y-%m-%d %X'), 'value': probability} for timestamp, probability in power_forecast]
        elif selector == 'power_func':
            self.run_new_power(data)
                
