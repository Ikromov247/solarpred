"""
Template for implementing a model class
Change implementation details and add new functions based on your needs
"""
from pathlib import Path
import joblib

from .model_config import load_model_config
from core.logging_config import get_logger

class ModelClass:
    def __init__(self,  config, model_parameters):
        self.logger = get_logger(__name__)
        self.config = config
        self.model_parameters = model_parameters
        self.is_trained = False

        self.model = RandomForestRegressor(**model_parameters)

    def save(self, path: str):
        """Save model and all necessary state"""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self.model,
            "config": self.config,
            "model_parameters": self.model_parameters,
            "is_trained": self.is_trained,
        }
        self.logger.debug(
            f"Save model of state {list(state.keys())} at path {model_path}"
        )
        joblib.dump(state, model_path)

    @classmethod
    def load(cls, path: str) -> "ModelClass":
        """Load model and all necessary state"""
        try:
            state = joblib.load(path)
            instance = cls(
                config=state["config"], model_parameters=state["model_parameters"]
            )
            instance.model = state["model"]
            instance.is_trained = state["is_trained"]
            return instance
        
        except FileNotFoundError as e:
            config = load_model_config()
            model_parameters = load_model_parameters()
            return cls(config, model_parameters)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def train(self):
        pass

    def predict(self):
        pass