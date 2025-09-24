from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from sklearn.preprocessing import StandardScaler
import torch.cuda


# Device Detection Utility
def get_device() -> str:
    """Detect available device for model training/inference."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Scaler Factory Functions
def create_standard_scaler() -> StandardScaler:
    """Create a new StandardScaler instance."""
    return StandardScaler()


class BaseMLConfig(BaseModel):
    """Base configuration for all ML models."""
    val_size: int = Field(default=44, ge=1, description="Validation set size")
    separate_val_set: bool = Field(default=False, description="Whether to use separate validation set")
    target_col: str = Field(default="solar_power", min_length=1, description="Target column name")
    features_to_use: List[str] = Field(
        default=[
            'global_tilted_irradiance_instant', 
            'global_tilted_irradiance', 
            'cloud_cover_mid', 
            'cloud_cover_high',
            'uv_index',
            'diffuse_radiation',
            'direct_radiation_instant'
        ],
        min_length=1,
        description="List of features to use for training"
    )
    normalize: bool = Field(default=True, description="Whether to normalize input features")
    deterministic: bool = Field(default=True, description="Whether to use deterministic training")
    device: str = Field(default_factory=get_device, description="Device for model training/inference")
    
    @field_validator('features_to_use')
    @classmethod
    def features_must_not_be_empty_strings(cls, v):
        if any(not feature.strip() for feature in v):
            raise ValueError('Feature names cannot be empty strings')
        return v
    
    def create_scaler(self) -> StandardScaler:
        """Create a new scaler instance."""
        return create_standard_scaler()


class NeuralNetworkConfig(BaseMLConfig):
    """Configuration specific to neural network models."""
    n_epochs: int = Field(default=14, ge=1, le=1000, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=1, le=1024, description="Training batch size")
    learning_rate: float = Field(default=0.002, gt=0, le=1, description="Learning rate for optimizer")
    dropout_rate: float = Field(default=0.1, ge=0, lt=1, description="Dropout rate for regularization")
    


class MLConfigSettings(BaseSettings):
    """Environment-based ML configuration settings."""
    # Neural Network overrides
    nn_epochs: Optional[int] = Field(default=None, alias="ML_NN_EPOCHS")
    nn_batch_size: Optional[int] = Field(default=None, alias="ML_NN_BATCH_SIZE")
    nn_learning_rate: Optional[float] = Field(default=None, alias="ML_NN_LEARNING_RATE")
    nn_dropout_rate: Optional[float] = Field(default=None, alias="ML_NN_DROPOUT_RATE")
    
    # Base ML overrides
    val_size: Optional[int] = Field(default=None, alias="ML_VAL_SIZE")
    normalize: Optional[bool] = Field(default=None, alias="ML_NORMALIZE")
    deterministic: Optional[bool] = Field(default=None, alias="ML_DETERMINISTIC")
    
    model_config = {
        "env_prefix": "ML_",
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra environment variables
    }


# Configuration Factory
class ModelConfigFactory:
    """Factory for creating model configurations with validation."""
    
    _available_models = ["neural_network"]
    _settings = MLConfigSettings()
    
    @classmethod
    def available_models(cls) -> List[str]:
        """Get list of available model types."""
        return cls._available_models.copy()
    
    @classmethod
    def create_neural_network_config(
        cls, 
        **overrides
    ) -> NeuralNetworkConfig:
        """Create neural network configuration with optional overrides."""
        
        # Base configuration
        config_data = {}
        
        # Apply environment variable overrides
        if cls._settings.nn_epochs is not None:
            config_data["n_epochs"] = cls._settings.nn_epochs
        if cls._settings.nn_batch_size is not None:
            config_data["batch_size"] = cls._settings.nn_batch_size
        if cls._settings.nn_learning_rate is not None:
            config_data["learning_rate"] = cls._settings.nn_learning_rate
        if cls._settings.nn_dropout_rate is not None:
            config_data["dropout_rate"] = cls._settings.nn_dropout_rate
        if cls._settings.val_size is not None:
            config_data["val_size"] = cls._settings.val_size
        if cls._settings.normalize is not None:
            config_data["normalize"] = cls._settings.normalize
        if cls._settings.deterministic is not None:
            config_data["deterministic"] = cls._settings.deterministic
        
        # Apply any additional overrides
        config_data.update(overrides)
        
        return NeuralNetworkConfig(**config_data)
    
    @classmethod
    def create_config(
        cls, 
        model_name: str,
        **overrides
    ) -> BaseMLConfig:
        """Create configuration for specified model type."""
        
        if model_name not in cls._available_models:
            raise ValueError(
                f"Model '{model_name}' is not available. "
                f"Available models: {cls._available_models}"
            )
        
        if model_name == "neural_network":
            return cls.create_neural_network_config( **overrides)
        
        raise NotImplementedError(f"Configuration for model '{model_name}' not implemented")


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Legacy function for backward compatibility. Returns dict instead of Pydantic model."""
    config = ModelConfigFactory.create_config(model_name)
    config_dict = config.model_dump()
    config_dict["scaler"] = config.create_scaler()
    
    return config_dict


def available_models() -> List[str]:
    """Get list of available model types."""
    return ModelConfigFactory.available_models()