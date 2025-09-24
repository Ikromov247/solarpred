import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering on the dataset."""
    df = df.copy()

    df['global_tilted_irradiance_instant_squared'] = df['global_tilted_irradiance_instant']**2 / 100
    
    # Add cyclical time features
    df.loc[:, 'hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df.loc[:, 'hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df.loc[:, 'month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df.loc[:, 'month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    df.dropna(inplace=True)
    
    return df