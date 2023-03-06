import numpy as np
import pandas as pd
from astral import sun

def preprocess_power_data(new_power_data):
    time=pd.to_datetime(new_power_data['energy_time']).tz_localize(None)
    power = new_power_data['energy']
    return time, power


def preprocess_weather_data(new_weather_data):

    # Normalization of time (min=0, max=24), relative humidity (min=0, max=100), uv-index (min=0, max~10), cloud area fraction
    # (min=0, max=100) is easy. For temperature we choose min=-10, max=30 and for precipitation amount: min=0, max=10.

    aux_list = [[pd.to_datetime(data_point['weather_time']).tz_localize(None).hour/24, (data_point['instant_air_temperature']+10)/40, data_point['instant_relative_humidity']/100,
                 data_point['instant_ultraviolet_index_clear_sky']/10, data_point['1_hours_precipitation_amount']/10,
                 data_point['instant_cloud_area_fraction']/100] for data_point in new_weather_data]

    weather_array = np.array(aux_list)

    return pd.to_datetime(new_weather_data[0]['weather_time']).tz_localize(None), weather_array

def get_sunrise_sunset(observer, time):
    sunrise = pd.to_datetime(sun.sunrise(observer, date=time, tzinfo='UTC')).tz_localize(None)
    sunset = pd.to_datetime(sun.sunset(observer, date=time, tzinfo='UTC')).tz_localize(None)
    return sunrise, sunset

def update_replay_buffer(replay_buffer, agent, history):
    high_power_counter = 0
    history_mean = sum(history)/len(history)

    for old_agent in replay_buffer:
        old_agents_power_mean = sum([power_value for _, power_value in old_agent.power_list])/len([power_value for _, power_value in old_agent.power_list])
        if old_agents_power_mean >= history_mean:
            high_power_counter += 1

    low_power_counter = len(replay_buffer) - high_power_counter

    agents_power_mean = sum([power_value for _, power_value in agent.power_list])/len([power_value for _, power_value in agent.power_list])
    if agents_power_mean-history_mean >= 0:
        agents_type = 'high'
    elif agents_power_mean-history_mean < 0:
        agents_type = 'low'

    if high_power_counter <= low_power_counter and agents_type == 'high':
        replay_buffer.append(agent)
    elif high_power_counter > low_power_counter and agents_type == 'low':
        replay_buffer.append(agent)