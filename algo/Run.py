import Agent
import aux_functions

from collections import deque

import pickle

import numpy as np

import torch
import torch.optim as optim





def init_algo():             # This function initializes all global variables of the algorithm.
    agents = deque(maxlen=8) # This list will contain the 8 agents that will learn in parallel.

    history_power = []       # This list will contain the power values from the last 1/2/4 weeks.

    policy = Agent.Policy(state_size=8)        # The policy that has to be learned.
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    rewards = []             # This list tracks all rewards given to all of the agents.

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    
    if use_cuda:
        policy = policy.cuda()

    history_time_delta = None # The time delta over which all power values are saved to compute the power threshold. 

    return agents, history_power, policy, optimizer, rewards, use_cuda, history_time_delta


def run_new_power(): # This function is run after a new solar power value is input.
    new_power = aux_functions.load_power_data()
    history_power = aux_functions.update_history_power(history_power, new_power, history_time_delta) # Update the power history from the last 1/2/4 weeks.
        
    for agent in agents:
        agent.update_power_list(new_power) # Update the power list for every agent. (The power list of an agent is used for computation of the reward.)


def run_new_weather(): # This function is run after a new weather data point is input; i.e. once each 15min/30min.
    new_weather = aux_functions.load_weather_data()

    if len(agents) < 8:
        agents.append(Agent.Agent(use_cuda))
    elif len(agents) == 8:
        oldest_agent = agents.popleft()
        agents.append(Agent.Agent(use_cuda))
        history_power_mean = np.mean(np.array(history_power))
        oldest_agent.learn(policy, history_power_mean, optimizer)
        rewards.append(oldest_agent.reward)
            
        torch.save(policy.state_dict(), '...policy.pt')
        with open('...rewards.pickle', 'wb') as f:
            pickle.dump(rewards, f)
            
        del oldest_agent
            
    newest_agent = agents[-1]
    newest_agent.save_weather_data(new_weather)
    newest_agent.act(policy)
    
    return newest_agent.action

# The global variables get initialized. "rewards" can be saved on disk. All other variables have to be kept in memory.

agents, history_power, policy, optimizer, rewards, use_cuda, history_time_delta = init_algo() 


NEW_POWER_STATUS = bool
NEW_WEATHER_STATUS = bool

if NEW_POWER_STATUS: # As often as possible
    run_new_power()

if NEW_WEATHER_STATUS: # Every 15min/30min
    proposed_action = run_new_weather()


