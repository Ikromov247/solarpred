import numpy as np

def train_val_split(train_set, model_CONFIG):
    val_size = model_CONFIG['val_size']
    separate_val_set = model_CONFIG['separate_val_set']

    # Get the validation set. If separate_val_set is True, then we need to separate the validation set from the training set.
    val_set = train_set.iloc[-val_size:]
    if separate_val_set:
        train_index = [element for element in train_set.index if element not in val_set.index]
        train_set = train_set.loc[train_index]

    return train_set, val_set

def normalize_train_val(train_set, val_set, scaler_X, scaler_y, model_CONFIG):
    # Split the data into features and target
    X_train = train_set[model_CONFIG['features_to_use']].values
    y_train = train_set[model_CONFIG['target_col']].values
    X_val = val_set[model_CONFIG['features_to_use']].values
    y_val = val_set[model_CONFIG['target_col']].values
    
    # Normalize if the model_CONFIG['normalize'] is True, otherwise return the data as is
    if model_CONFIG['normalize']:
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)

        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    return X_train, X_val, y_train, y_val