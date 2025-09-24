import pandas as pd

def preprocess_sunset_sunrise(sunset_sunrise_data: pd.DataFrame) -> pd.DataFrame:
    # Create a clean copy of the relevant columns
    sunset_sunrise_df = sunset_sunrise_data.copy()
    
    # Convert Unix timestamps to datetime
    for col in ['sunrise', 'sunset']:
        # since sunset and sunrise have only time and no date, the date is added using the timestamp column
        sunset_sunrise_df[col] = pd.to_datetime(sunset_sunrise_df['timestamp'] + ' ' + sunset_sunrise_df[col])
    
    # Convert timestamp to datetime and set it as the index
    sunset_sunrise_df['timestamp'] = pd.to_datetime(sunset_sunrise_df['timestamp'])
    sunset_sunrise_df = sunset_sunrise_df.set_index('timestamp')

    return sunset_sunrise_df