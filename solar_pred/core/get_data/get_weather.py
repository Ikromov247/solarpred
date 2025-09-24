import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime
from typing import  Dict


def _fetch_weather_data(latitude: float, longitude: float, url: str, params: Dict) -> pd.DataFrame:
    """
    Helper function to fetch and process weather data from Open-Meteo API.
    
    Args:
        latitude (float): The latitude of the location
        longitude (float): The longitude of the location
        url (str): API endpoint URL
        params (Dict): Parameters for the API request
        
    Returns:
        pd.DataFrame: DataFrame containing hourly weather data
    """
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make the API request
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Process hourly data
    hourly = response.Hourly()
    
    # Create the base hourly_data dictionary with timestamp
    hourly_data = {"timestamp": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}

    # Add all variables to the hourly_data dictionary
    for i, variable in enumerate(params["hourly"]):
        hourly_data[variable] = hourly.Variables(i).ValuesAsNumpy()

    # Create DataFrame
    hourly_dataframe = pd.DataFrame(data=hourly_data)

    # Localize the timestamps to the correct timezone and then make it timezone-naive
    hourly_dataframe['timestamp'] = hourly_dataframe['timestamp'].dt.tz_convert('Asia/Tokyo')
    hourly_dataframe['timestamp'] = hourly_dataframe['timestamp'].dt.tz_localize(None)

    return hourly_dataframe


def get_weather_data_by_date(latitude: float, longitude: float, 
                           start_date, 
                           end_date) -> pd.DataFrame:
    """
    Get weather data for a specified location and time period using explicit dates.
    
    Args:
        latitude (float): The latitude of the location
        longitude (float): The longitude of the location
        start_date (Union[datetime, str]): Start date for weather data
        end_date (Union[datetime, str]): End date for weather data
        
    Returns:
        pd.DataFrame: DataFrame containing hourly weather data
    """
    # Convert string dates to datetime
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    present_day = datetime.now().date()

    # Calculate the difference in days from present
    days_from_present_start = (present_day - start_date).days
    days_from_present_end = (end_date - present_day).days

    # Define common parameters for both APIs
    hourly_params = [
                    'global_tilted_irradiance_instant', 
                    'global_tilted_irradiance', 
                    'cloud_cover_mid', 
                    'cloud_cover_high',
                    'uv_index',
                    'diffuse_radiation',
                    'direct_radiation_instant'
                    ]

    common_params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "Asia/Tokyo",
        "hourly": hourly_params
    }

    # Determine which API to use based on the date range
    if days_from_present_start > 92:
        # Use historical forecast API for dates more than 92 days in the past
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        params = {
            **common_params,
            "start_date": start_date,
            "end_date": min(end_date, present_day)
        }
    else:
        # Use regular forecast API for recent past and future dates
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            **common_params,
            "past_days": min(days_from_present_start, 92) if days_from_present_start > 0 else 0,
            "forecast_days": min(days_from_present_end, 16) if days_from_present_end > 0 else 0
        }

    return _fetch_weather_data(latitude, longitude, url, params)


def get_weather_data_for_df(latitude: float, longitude: float, df: pd.DataFrame) -> pd.DataFrame:
    """
    Get weather data for a specified location based on the date range of a DataFrame.
    
    Args:
        latitude (float): The latitude of the location
        longitude (float): The longitude of the location
        df (pd.DataFrame): DataFrame with datetime index
        
    Returns:
        pd.DataFrame: DataFrame containing hourly weather data
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index")

    start_date = df.index.min()
    end_date = df.index.max()
    
    start_date = str(start_date)
    end_date = str(end_date)
    
    return get_weather_data_by_date(latitude, longitude, start_date, end_date)
