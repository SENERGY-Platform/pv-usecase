import Agent
import aux_functions
from collections import deque
import pickle
import numpy as np
import torch
import torch.optim as optim

def init_algo(history_power_td=60000, weather_dim=6,data_path):             # This function initializes all global variables of the algorithm.
    agents = deque(maxlen=4) # This list will contain the 4 agents that will learn in parallel.
    policy = Agent.Policy(state_size=weather_dim)        # The policy that has to be learned.
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    history_power = deque(maxlen=history_power_td)      # This list will contain the power values from the last 1/2/4 weeks.
    replay_buffer = deque(maxlen=50)                    

    rewards = []             # This list tracks all rewards given to all of the agents.

    data_path = data_path

    return agents, policy, optimizer, history_power, replay_buffer, rewards, data_path


def run_new_power(new_power): # This function is run after a new solar power value is input.
    #new_power = aux_functions.load_power_data()
    history_power = aux_functions.update_history_power(history_power, new_power, history_time_delta) # Update the power history from the last 1/2/4 weeks.
        
    for agent in agents:
        agent.update_power_list(new_power) # Update the power list for every agent. (The power list of an agent is used for computation of the reward.)


def run_new_weather(new_weather_data): # This function is run after a new weather data point is input; i.e. once each 15min/30min.

    policy.eval()      # The current policy is used for prediction.
    with torch.no_grad():
        input = torch.Tensor(new_weather_data)
        if use_cuda:
            input.cuda()
        output = policy(input)
    policy.train()

    
    if len(agents) < 4:
        agents.append(Agent.Agent(use_cuda))
    elif len(agents) == 4:
        oldest_agent = agents.popleft()
        agents.append(Agent.Agent(use_cuda))
        oldest_agent.action, oldest_agent.log_prob = oldest_agent.act(policy)
        oldest_agent.reward = oldest_agent.get_reward(oldest_agent.action, history_power_mean=sum(history_power)/len(history_power))
        oldest_agent.learn(oldest_agent.reward, oldest_agent.log_prob, optimizer)
        del oldest_agent
            
    torch.save(policy.state_dict(), 'policy.pt')
    with open('rewards.pickle', 'wb') as f:
        pickle.dump(rewards, f)
            
           
    newest_agent = agents[-1]
    newest_agent.save_weather_data(new_weather)
    newest_agent.act(policy)
    
    return output

# The global variables get initialized. "rewards" can be saved on disk. All other variables have to be kept in memory.

agents, history_power, policy, optimizer, rewards, use_cuda, history_time_delta = init_algo() 


NEW_POWER_STATUS = bool
NEW_WEATHER_STATUS = bool

if NEW_POWER_STATUS: # As often as possible
    run_new_power()

if NEW_WEATHER_STATUS: # Every 15min/30min
    proposed_action = run_new_weather()


