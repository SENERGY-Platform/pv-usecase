import numpy as np

def load_power_data():

    # Load and preprocess power data here.

    return power


def load_weather_data():

    # Load and preprocess weather data here.

    return (time,(temp, humid, uv, rain_prob, cloud, time_since_rise, time_until_sunset))


def update_history_power(history_power, new_power_value, history_time_delta): # Updates the history of the last power values.
    if len(history_power) < history_time_delta:
        history_power.append(new_power_value)
    elif len(history_power) == history_time_delta:
        history_power.append(new_power_value)
        history_power = history_power[1:]
    return history_power
