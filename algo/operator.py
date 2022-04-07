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


class Operator(util.OperatorBase):
    def __init__(self, energy_src_id, weather_src_id):
        self.energy_src_id = energy_src_id
        self.weather_src_id = weather_src_id

        self.weather_same_timestamp = []

        self.replay_buffer = deque(maxlen=50)
        self.power_history = deque(maxlen=history_power_td)
        
        self.agents = deque(maxlen=4)
        self.policy = Agent.policy(state_size=weather_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)

        self.rewards = []




    def run_new_weather(self, new_weather_data):
        new_weather_array = aux_functions.preprocess_weather_data(new_weather_data)

        self.policy.eval()      # The current policy is used for prediction.
        with torch.no_grad():
            input = torch.Tensor(new_weather_array)
            output = self.policy(input)
        self.policy.train()

    
        if len(self.agents) < 4:
            self.agents.append(Agent.Agent())
        elif len(self.agents) == 4:
            oldest_agent = self.agents.popleft()
            self.agents.append(Agent.Agent())
            oldest_agent.action, oldest_agent.log_prob = oldest_agent.act(self.policy)
            oldest_agent.reward = oldest_agent.get_reward(oldest_agent.action, history_power_mean=sum(self.power_history)/len(self.power_history))
            oldest_agent.learn(oldest_agent.reward, oldest_agent.log_prob, self.optimizer)
            del oldest_agent
            
        torch.save(self.policy.state_dict(), 'policy.pt')
        with open('rewards.pickle', 'wb') as f:
            pickle.dump(self.rewards, f)
            
           
        newest_agent = self.agents[-1]
        newest_agent.save_weather_data(new_weather_array)
        newest_agent.act(self.policy)
    
        return output

    def run_new_power(self, new_power_data):
        new_power_value = aux_functions.preprocess_power_data(new_power_data)

        self.history_power.append(new_power_value)

        for agent in self.agents:
            agent.update_power_list(new_power_value)

    def run(self, data, selector):

        if selector['name'] == 'weather_func':
            if self.weather_same_timestamp != []:
                if data['time'] == self.weather_same_time_stamp[-1]['time']:
                    self.weather_same_timestamp.append(data)
                elif data['time'] != self.weather_same_timestamp[-1]['time']:
                    new_weather_data = self.weather_same_timestamp
                    self.weather_same_timestamp = []
                    output = self.run_new_weather(new_weather_data)
                    return output
        elif selector['name'] == 'power_func':
            self.run_new_power()
