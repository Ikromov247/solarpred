import pandas as pd

def preprocess_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    # Create a copy of the input DataFrame to avoid modifying the original
    weather_df = weather_df.copy()

    # Convert datetime to pandas datetime and set it as index
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    weather_df = weather_df.set_index('timestamp')

    return pd.DataFrame(weather_df)