import pandas as pd
import pytz
from datetime import date, timedelta

from solar_pred.core.logging_config import get_logger
from solar_pred.core.preprocessing import preprocess_datasets
from solar_pred.core.get_data import get_suntimes_by_date, get_suntimes_from_inverter, get_weather_data_by_date, get_weather_data_for_df



class DataProcessor:
    DATE_STRFORMAT = "%Y%m%d%H%M%S"
    TIMEZONE = pytz.timezone('Asia/Seoul')

    @staticmethod
    def preprocess_training_input(training_input):
        """
        Process raw inverter data
        Fetch weather data based on inv data dates
        merge two data streams
        scale
        return 
        """
        panel_metadata = training_input.panel_metadata.model_dump()
        panel_output = pd.DataFrame([data.model_dump() for data in training_input.panel_output])
        # go through the preprocessing pipeline
        panel_output.loc[:, 'timestamp'] = pd.to_datetime(panel_output['timestamp'], format=DataProcessor.DATE_STRFORMAT)
        panel_output.set_index("timestamp", inplace=True)
        panel_output_resampled = panel_output.resample("1h").mean()
        panel_output_resampled.dropna(axis=0, inplace=True)
        

        # If you don't have the inverter data, you can use the the other function, to get the sunset and sunrise times for the period specified by the start and end dates.
        sunset_sunrise_raw_df = get_suntimes_from_inverter(
            latitude=panel_metadata['latitude'], 
            longitude=panel_metadata['longitude'], 
            altitude=panel_metadata['altitude'],
            timezone=DataProcessor.TIMEZONE,
            inverter_df=panel_output_resampled
        )

        weather_raw_df = get_weather_data_for_df(
            latitude=panel_metadata['latitude'], 
            longitude=panel_metadata['longitude'], 
            df=panel_output_resampled
        )
        merged_dataset = preprocess_datasets(
            weather=weather_raw_df, 
            sunset_sunrise=sunset_sunrise_raw_df, 
            inverter=panel_output_resampled
        )
        
        merged_dataset.dropna(axis=0, inplace=True)
        return merged_dataset


    @staticmethod
    def preprocess_inference_input(inference_input):
        # take dates for prediction
        # fetch weather data
        # fetch suntimes
        # process the data
        # return

        panel_metadata = inference_input.model_dump()
        start_date, end_date = get_prediction_dates(panel_metadata['predict_days'])
        sunset_sunrise_raw_df = get_suntimes_by_date(
            latitude=panel_metadata['latitude'], 
            longitude=panel_metadata['longitude'], 
            altitude=panel_metadata['altitude'],
            timezone=DataProcessor.TIMEZONE,
            start_date=start_date,
            end_date=end_date
        )
        weather_raw_df = get_weather_data_by_date(
            latitude=panel_metadata['latitude'], 
            longitude=panel_metadata['longitude'], 
            start_date=start_date,
            end_date=end_date
        )
        weather_df = preprocess_datasets(weather_raw_df, sunset_sunrise_raw_df)
        return weather_df



def get_prediction_dates(days)->tuple[date, date]:
    # tomorrow's date is the default start date
    start_date = date.today() + timedelta(days=1)

    # tomorrow + days = end date
    end_date = start_date + timedelta(days=days)
    return start_date, end_date