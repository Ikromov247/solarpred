import pandas as pd

def preprocess_inverter(inverter_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to clean and preprocess inverter data.

    Parameters:
    - inverter_data: pandas.DataFrame, the input DataFrame from an API.

    Returns:
    - inverter_data: pandas.DataFrame, the cleaned DataFrame.
    """
    inverter_data_df = inverter_data_df.copy()

    # Select and reorder columns
    inverter_data_df = inverter_data_df['solar_power'].to_frame()

    # Drop rows with NaN values
    inverter_data_df = inverter_data_df.dropna()

    # Convert 'solar_power' to float
    inverter_data_df['solar_power'] = inverter_data_df['solar_power'].astype(float)

    inverter_data_resampled = inverter_data_df.resample('H').mean()

    # Remove duplicate indices
    inverter_data_resampled = inverter_data_resampled.groupby(level=0).first()


    # # # Drop data points where generated power is less than 18.72 kW
    # inverter_data_resampled = inverter_data_resampled[inverter_data_resampled['solar_power'] >= 18.72]

    return pd.DataFrame(inverter_data_resampled)