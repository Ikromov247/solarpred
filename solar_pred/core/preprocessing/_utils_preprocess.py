import pandas as pd


def merge_datasets(df1, df2, method='inner'):
    
    # Perform an inner merge with the weather data using inner join. We use inner joing because we can't train or make predictions without weather data.
    merged_df = pd.merge(df1, df2, 
                         left_index=True, right_index=True, 
                         how=method)

    # Sort the index to ensure chronological order
    merged_df = merged_df.sort_index()

    return merged_df

def filter_daylight_hours(df, sunset_sunrise):
    df = df.copy()
    sunset_sunrise = sunset_sunrise.copy()

    df_index = pd.Index(df.index.date)
    sunset_sunrise_index = pd.Index(sunset_sunrise.index.date)

    # Keep only the days present in sunset_sunrise_df
    df = df[df_index.isin(sunset_sunrise_index)]

    # Function to check if a timestamp is within daylight hours
    def is_daylight(timestamp):
        date = str(timestamp.date())
        sunrise = sunset_sunrise.loc[date, 'sunrise']
        sunset = sunset_sunrise.loc[date, 'sunset']

        # Add an hour to sunrise and subtract an hour from sunset, otherwise the timestamps are not filtered properly
        sunrise -= pd.Timedelta(hours=1)
        sunset += pd.Timedelta(hours=1)

        return sunrise <= timestamp <= sunset

    # Apply the filter
    filtered_data = df[df.index.map(is_daylight)]

    return pd.DataFrame(filtered_data)

