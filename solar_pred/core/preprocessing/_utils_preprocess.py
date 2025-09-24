from pytz import timezone
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple


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

