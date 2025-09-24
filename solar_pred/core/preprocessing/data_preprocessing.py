import pandas as pd
from typing import Tuple

from .inverter_preprocessing import preprocess_inverter
from .weather_preprocessing import preprocess_weather
from .sunset_sunrise_preprocessing import preprocess_sunset_sunrise
from ._utils_preprocess import filter_daylight_hours, merge_datasets
from ._feature_engineering import engineer_features

def preprocess_datasets(weather: pd.DataFrame, sunset_sunrise: pd.DataFrame, inverter=None):
    weather_preprocessed = preprocess_weather(weather)
    sunset_sunrise_preprocessed = preprocess_sunset_sunrise(sunset_sunrise)

    # Get rid of nighttime hours
    weather_preprocessed = filter_daylight_hours(weather_preprocessed, sunset_sunrise_preprocessed)

    # Engineer features
    weather_preprocessed = engineer_features(weather_preprocessed)

    # If inverter data is provided, preprocess it, merge with weather data and return. This is done so that we can use the function for inference too where we don't have inverter data.
    if inverter is not None:
        inverter_preprocessed = preprocess_inverter(inverter)
        merged_dataset = merge_datasets(weather_preprocessed, inverter_preprocessed, method='inner')
        return merged_dataset 
    else:
        return weather_preprocessed