import numpy as np
import pandas as pd

def preprocess_power_data(new_power_data):

    time = new_power_data['energy_time']
    power = new_power_data['energy']
    return power


def preprocess_weather_data(new_weather_data):

    aux_list = [[pd.to_datetime(data_point['weather_time']).tz_localize(None).hour, data_point['instant_air_temperature'], data_point['instant_relative_humidity'],
                 data_point['instant_ultraviolet_index_clear_sky'], data_point['1_hours_precipitation_amount'],
                 data_point['instant_cloud_area_fraction']] for data_point in new_weather_data]

    weather_array = np.array(aux_list)

    return weather_array
