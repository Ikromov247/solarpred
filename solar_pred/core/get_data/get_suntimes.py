import ephem
from datetime import timedelta
import pandas as pd

def get_sunset_sunrise(lat, lon, altitude, date, timezone):
    """
    Calculate sunset and sunrise times for a specific location and date.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        altitude (float): Altitude in meters
        date (datetime.date): Date to calculate for
        timezone (tzinfo): Timezone object
        
    Returns:
        tuple: (sunrise_local, sunset_local) datetime objects
    """
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.elevation = altitude
    observer.date = date
    observer.pressure = 0
    observer.horizon = '-0:34'

    sun = ephem.Sun()
    
    sunrise = observer.next_rising(sun)
    sunset = observer.next_setting(sun)

    sunrise_local = ephem.localtime(sunrise).astimezone(timezone).replace(tzinfo=None)
    sunset_local = ephem.localtime(sunset).astimezone(timezone).replace(tzinfo=None)

    return sunrise_local, sunset_local

def get_suntimes_from_inverter(latitude, longitude, altitude, timezone, inverter_df):
    """
    Get sunrise and sunset times for the date range in the inverter dataframe.
    
    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude
        altitude (float): Altitude in meters
        timezone (tzinfo): Timezone object
        inverter_df (pd.DataFrame): Inverter data with datetime index
        
    Returns:
        pd.DataFrame: DataFrame with dates, sunrise, and sunset times
    """
    if inverter_df.empty:
        raise ValueError("Inverter DataFrame is empty")
        
    start_date = inverter_df.index.min().date()
    end_date = inverter_df.index.max().date()
    
    return get_suntimes_by_date(latitude, longitude, altitude, timezone, start_date, end_date)

def get_suntimes_by_date(latitude, longitude, altitude, timezone, start_date, end_date):
    """
    Get sunrise and sunset times for a specific date range.
    
    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude
        altitude (float): Altitude in meters
        timezone (tzinfo): Timezone object
        start_date (datetime.date): Start date
        end_date (datetime.date): End date
        
    Returns:
        pd.DataFrame: DataFrame with dates, sunrise, and sunset times
    """
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
        
    if start_date > end_date:
        raise ValueError("Start date must be before or equal to end date")

    data = []
    current_date = start_date

    while current_date <= end_date:
        sunrise, sunset = get_sunset_sunrise(latitude, longitude, altitude, current_date, timezone)
        data.append({
            'timestamp': str(current_date),
            'sunrise': sunrise.strftime('%H:%M:%S'),
            'sunset': sunset.strftime('%H:%M:%S')
        })
        
        current_date += timedelta(days=1)

    return pd.DataFrame(data)