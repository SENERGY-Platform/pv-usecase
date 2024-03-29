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
from timezonefinder import TimezoneFinder


class Operator(util.OperatorBase):
    def __init__(self, lat, long, power_history_start_stop='2', buffer_len='48', weather_dim=6, data_path="data"):
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        self.data_path = data_path
        
        self.lat = float(lat)
        self.long = float(long)
        self.observer = astral.Observer(latitude=self.lat, longitude=self.long)

        tf = TimezoneFinder()
        self.timezone=tf.certain_timezone_at(lng=self.long, lat=self.lat)

        self.weather_same_timestamp = []

        self.power_history_start_stop = int(power_history_start_stop)

        self.buffer_len = int(buffer_len)
        self.history_power_len = pd.Timedelta(7,'days')
        self.replay_buffer = deque(maxlen=self.buffer_len)
        self.power_history = []
        self.daylight_power_history = []

        self.agents = []
        self.policy = Agent.Policy(state_size=weather_dim) # If we keep track of time, temp, humidity, uv-index, precipitation and clouds we have weather_dim=6.
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)

        self.model_file = f'{data_path}/model_{self.power_history_start_stop}.pt'
        self.replay_buffer_file = f'{data_path}/replay_buffer_{self.power_history_start_stop}.pickle'
        self.daylight_power_history_file = f'{data_path}/daylight_power_history_{self.power_history_start_stop}.pickle'

        if os.path.exists(self.replay_buffer_file):
            with open(self.replay_buffer_file, 'rb') as f:
                if os.path.getsize(self.replay_buffer_file) > 0:
                    self.replay_buffer = pickle.load(f)

        if os.path.exists(self.daylight_power_history_file):
            with open(self.daylight_power_history_file, 'rb') as f:
                if os.path.getsize(self.daylight_power_history_file) > 0:
                    self.daylight_power_history = pickle.load(f)

        if os.path.exists(self.model_file):
            if os.path.getsize(self.model_file) > 0:
                self.policy.load_state_dict(torch.load(self.model_file))


    def run_new_weather(self, new_weather_data):
        weather_time, new_weather_array = aux_functions.preprocess_weather_data(new_weather_data)
        new_weather_input = np.mean(new_weather_array, axis=0)

        self.policy.eval()      # The current policy is used for prediction.
        with torch.no_grad():
            input = torch.from_numpy(new_weather_input).float().unsqueeze(0)
            output = torch.argmax(self.policy(input)).item()
        self.policy.train()

        sunrise, sunset = aux_functions.get_sunrise_sunset(self.observer, weather_time)
        
        if weather_time < sunset+pd.Timedelta(3,'hours') and weather_time > sunrise-pd.Timedelta(3,'hours'):
            self.agents.append(Agent.Agent())
            newest_agent = self.agents[-1]
            newest_agent.save_weather_data(new_weather_input)
            newest_agent.initial_time = pd.to_datetime(new_weather_data[0]['weather_time'])
        
        if len(self.replay_buffer)==self.buffer_len:
            random.shuffle(self.replay_buffer)
            for agent in self.replay_buffer:
                action, log_prob = agent.act(self.policy)
                reward = agent.get_reward(action, [power for _, power in self.daylight_power_history])
                agent.learn(reward, log_prob, self.optimizer)

        torch.save(self.policy.state_dict(), self.model_file)
        
        if output==0:
            return {"value": 0}
        elif output==1:
            return {"value": 1}

    def run_new_power(self, new_power_data):
        time, new_power_value = aux_functions.preprocess_power_data(new_power_data,self.timezone)
        if new_power_value != None:
            self.power_history.append((time,new_power_value))
            if time-self.power_history[0][0] > self.history_power_len:
                del self.power_history[0]
        sunrise, sunset = aux_functions.get_sunrise_sunset(self.observer, time)
        if (sunrise+pd.Timedelta(self.power_history_start_stop, 'hours')<time) and (time+pd.Timedelta(self.power_history_start_stop, 'hours')<sunset):
            if new_power_value != None:
               self.daylight_power_history.append((time,new_power_value))
               if time-self.daylight_power_history[0][0] > self.history_power_len:
                   del self.daylight_power_history[0]
        with open(self.daylight_power_history_file, 'wb') as f:
            pickle.dump(self.daylight_power_history, f)

        old_agents = []
        old_indices = []
        
        for i, agent in enumerate(self.agents):
            if agent.initial_time + pd.Timedelta(2,'hours') >= time:
                if new_power_value != None:
                    agent.update_power_list(time, new_power_value)
            elif agent.initial_time + pd.Timedelta(2,'hours') < time:
                agents_initial_time_string = agent.initial_time.strftime('%Y-%m-%d %X')
                with open(f'{self.data_path}/agent_{agents_initial_time_string}_{self.power_history_start_stop}.pickle', 'wb') as f:
                    pickle.dump(agent, f)
                old_agents.append(agent)
                old_indices.append(i)

        old_indices = sorted(old_indices, reverse=True)
        for index in old_indices:
            del self.agents[index]
        for old_agent in old_agents:
            if len(self.replay_buffer)==self.buffer_len and old_agent.power_list != []:
                old_agent.action, old_agent.log_prob = old_agent.act(self.policy)
                old_agent.reward = old_agent.get_reward(old_agent.action, [power for _, power in self.daylight_power_history])
                old_agent.learn(old_agent.reward, old_agent.log_prob, self.optimizer)
            if old_agent.power_list != [] and self.daylight_power_history != []:
                aux_functions.update_replay_buffer(self.replay_buffer, old_agent, [power for _, power in self.daylight_power_history])

        with open(self.replay_buffer_file, 'wb') as f:
            pickle.dump(self.replay_buffer, f)

    def create_power_forecast(self, new_weather_data):
        self.policy.eval() 
        power_forecast = []
        _, new_weather_array = aux_functions.preprocess_weather_data(new_weather_data)
        new_weather_forecasted_for = [pd.to_datetime(datapoint['forecasted_for']) for datapoint in new_weather_data]
        for i in range(0,len(new_weather_array),3):
            new_weather_input = np.mean(new_weather_array[i:i+3], axis=0)
            with torch.no_grad():
                input = torch.from_numpy(new_weather_input).float().unsqueeze(0)
                probability = float(self.policy(input).squeeze()[1])
            power_forecast.append((new_weather_forecasted_for[i],probability))
        self.policy.train()
        return power_forecast
        
    def run(self, data, selector):
        print(selector + ": " + str(data))
        if selector == 'weather_func':
            if len(self.weather_same_timestamp)<47:
                self.weather_same_timestamp.append(data)
            elif len(self.weather_same_timestamp)==47:
                self.weather_same_timestamp.append(data)
                new_weather_data = self.weather_same_timestamp
                _ = self.run_new_weather(new_weather_data[0:3])
                power_forecast = self.create_power_forecast(new_weather_data)
                self.weather_same_timestamp = []
                if len(self.replay_buffer)==self.buffer_len:
                    print("PV-Operator-Output:", [{'timestamp':timestamp.strftime('%Y-%m-%d %X')+'Z', 'value': probability} for timestamp, probability in power_forecast])
                    return [{'timestamp':timestamp.strftime('%Y-%m-%dT%X')+'Z', 'value': probability} for timestamp, probability in power_forecast]
        elif selector == 'power_func':
            self.run_new_power(data)
