import Agent
import aux_functions
from collections import deque
import pickle
import numpy as np
import torch
import torch.optim as optim

def init_algo(data_path, history_power_td=60000, weather_dim=6):
    agents = deque(maxlen=4) 
    policy = Agent.Policy(state_size=weather_dim)        
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    history_power = deque(maxlen=history_power_td)      
    replay_buffer = deque(maxlen=50)                    

    rewards = []

    data_path = data_path

    return agents, policy, optimizer, history_power, replay_buffer, rewards, data_path


def run_new_power(new_power_data): 
    new_power_value = aux_functions.preprocess_power_data(new_power_data)

    history_power.append(new_power_value)
        
    for agent in agents:
        agent.update_power_list(new_power_value) 


def run_new_weather(new_weather_data):
    new_weather_array = aux_functions.preprocess_weather_data(new_weather_data)
    new_weather_input = np.mean(new_weather_array, axis=0) 

    policy.eval()     
    with torch.no_grad():
        input = torch.Tensor(new_weather_input)
        output = policy(input)
    policy.train()

    
    if len(agents) < 4:
        agents.append(Agent.Agent())
    elif len(agents) == 4:
        oldest_agent = agents.popleft()
        agents.append(Agent.Agent())
        oldest_agent.action, oldest_agent.log_prob = oldest_agent.act(policy)
        oldest_agent.reward = oldest_agent.get_reward(oldest_agent.action, history_power_mean=sum(history_power)/len(history_power))
        oldest_agent.learn(oldest_agent.reward, oldest_agent.log_prob, optimizer)
            
    replay_buffer.append(agents[-1])

    for agent in replay_buffer:
        agent.learn(agent.reward, agent.log_prob, optimizer)
    
    torch.save(policy.state_dict(), data_path+'/policy.pt')
    with open(data_path+'/rewards.pickle', 'wb') as f:
        pickle.dump(rewards, f)
            
           
    newest_agent = agents[-1]
    newest_agent.save_weather_data(new_weather_input)
    newest_agent.act(policy)
    
    return output

# The global variables get initialized. "rewards" can be saved on disk. All other variables have to be kept in memory.

agents, policy, optimizer, history_power, replay_buffer, rewards, data_path = init_algo() 

if NEW_POWER_STATUS: # As often as possible
    run_new_power(new_power_data)

if NEW_WEATHER_STATUS: # Every 30min
    proposed_action = run_new_weather(new_weather_data)


