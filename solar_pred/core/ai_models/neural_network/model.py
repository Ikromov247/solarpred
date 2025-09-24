import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import random
import copy
import pickle
import pandas as pd

from core.ai_models._models_general import train_val_split, normalize_train_val, output_boundaries
from solar_pred.core.exceptions import TrainSizeError, TestSizeErorr
from core.logging_config import get_logger

torch.serialization.add_safe_globals([StandardScaler])


# Custom loss function that represents our target metric.
class PercentageErrorLoss(nn.Module):
    def __init__(self, scale_factor=187.2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, predictions, targets):
        percentage_errors = torch.abs(predictions - targets) / self.scale_factor
        return torch.mean(percentage_errors) #* 100

# Set seeds for deterministic results
def set_seed(seed, full_determinism):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if full_determinism:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

class NeuralNetwork(nn.Module):
    def __init__(self, model_CONFIG: dict):
        # Set seeds for better reproducibility and if deterministic is true make it fully deterministic.
        set_seed(42, model_CONFIG['deterministic'])
        
        super(NeuralNetwork, self).__init__()
        self.model_CONFIG = model_CONFIG
        self.target_col = model_CONFIG['target_col']
        self.scaler_X = copy.deepcopy(model_CONFIG['scaler'])
        self.scaler_y = copy.deepcopy(model_CONFIG['scaler'])
        
        self.features_to_use = model_CONFIG['features_to_use']
        self.num_features = len(self.features_to_use)
        self.device = model_CONFIG['device']

        # Improved architecture
        self.fc1 = nn.Linear(self.num_features, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        # Dropout layer
        self.dropout = nn.Dropout(model_CONFIG['dropout_rate'])

        self.learning_rate = model_CONFIG['learning_rate']
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        self.is_trained = False

        self.to(self.device)

    def forward(self, x):

        # Rest of the network
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return x

    def prepare_train_data(self, train_set):    
        
        # Check if the training set is too small
        min_train_size = self.model_CONFIG['val_size'] + 1
        if len(train_set) < min_train_size:
            raise TrainSizeError(f"The training set is too small - {len(train_set)} rows. It must contain at least {min_train_size} rows.")

        # Split the data into training and validation sets
        train_set, val_set = train_val_split(train_set, self.model_CONFIG)

        # save unprocessed train and val sets for visualization
        self.train_split = train_set
        self.val_split = val_set

        # Normalize the data
        X_train, X_val, y_train, y_val = normalize_train_val(train_set, val_set, self.scaler_X, self.scaler_y, self.model_CONFIG)

        # Return the normalized data
        train_sets = (X_train, y_train)
        val_sets = (X_val, y_val)

        return train_sets, val_sets

    def fit_model(self, train_set):

        # Prepare the data
        train_sets, val_sets = self.prepare_train_data(train_set)

        # Train the model
        X_train, y_train = train_sets
        
        self.train()  # Set the model to training mode
        criterion = PercentageErrorLoss()
        for epoch in range(self.model_CONFIG['n_epochs']):
            total_loss = 0
            for i in range(0, len(X_train), self.model_CONFIG['batch_size']):
                batch_X = X_train[i:i+self.model_CONFIG['batch_size']]
                batch_y = y_train[i:i+self.model_CONFIG['batch_size']]
                
                states = torch.FloatTensor(batch_X).to(self.device)
                targets = torch.FloatTensor(batch_y).to(self.device).unsqueeze(1)
                
                predictions = self(states)
                loss = criterion(predictions, targets)
                
                self.optimizer.zero_grad()
                loss.backward() 
                self.optimizer.step()
                
                total_loss += loss.item()
        
        self.is_trained = True
        return self
    
    def prepare_inference_data(self, test):

        # Check if the test set is too small
        if len(test) < 1:
            raise TestSizeErorr("The test set is too small. It must contain at least 1 row.")

        # We only use the features_to_use to train the model.
        test = test[self.features_to_use]

        # We do not need y_test because we can just take the original df to compare the results. And in production we would not have y_test.
        X_test = test.values

        if self.model_CONFIG['normalize']:
            X_test = self.scaler_X.transform(X_test)
        
        return X_test

    def predict(self, test_set):
        # Prepare the data
        X = self.prepare_inference_data(test_set)

        # Predict
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self(X_tensor).cpu().numpy()

        predictions = self.postprocess_predictions(predictions.squeeze())
        return predictions
    
    def postprocess_predictions(self, predictions):

        if self.model_CONFIG['normalize']:
            predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1))

        predictions = output_boundaries(predictions, self.model_CONFIG)
        predictions = predictions.reshape(-1)

        return predictions

    def save_model(self, file_directory='saved_weights'):
        os.makedirs(file_directory, exist_ok=True)
        
        # Save the entire model state, including scalers
        model_state = {
            'model_state_dict': self.state_dict(),
            'model_CONFIG': self.model_CONFIG,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_split': self.train_split.to_dict() if hasattr(self.train_split, 'to_dict') else None,
            'val_split': self.val_split.to_dict() if hasattr(self.val_split, 'to_dict') else None,
            'features_to_use': self.features_to_use,
            'num_features': self.num_features,
            'learning_rate': self.learning_rate,
            'is_trained': self.is_trained
        }
        
        # Save using pickle
        with open(os.path.join(file_directory, 'neural_network_model.pkl'), 'wb') as f:
            pickle.dump(model_state, f)

    def load_model(self, file_directory='saved_weights'):
        file_path = os.path.join(file_directory, 'neural_network_model.pkl')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at {file_path}")

        # Load the entire model state using pickle
        with open(file_path, 'rb') as f:
            model_state = pickle.load(f)
        
        # Update model attributes
        self.load_state_dict(model_state['model_state_dict'])
        self.model_CONFIG = model_state['model_CONFIG']
        self.scaler_X = model_state['scaler_X']
        self.scaler_y = model_state['scaler_y']
        self.features_to_use = model_state['features_to_use']
        self.num_features = model_state['num_features']
        self.learning_rate = model_state['learning_rate']
        
        # Restore train and validation splits if they exist
        if model_state['train_split'] is not None:
            self.train_split = pd.DataFrame.from_dict(model_state['train_split'])
        if model_state['val_split'] is not None:
            self.val_split = pd.DataFrame.from_dict(model_state['val_split'])

        # Update optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        self.is_trained = model_state['is_trained']
        
        # Move model to appropriate device
        self.to(self.device)
        
        return self