import os


from solar_pred.core.ai_models.neural_network.model import load_nr_model
from core.config import DEFAULT_MODEL_PATH
from core.logging_config import get_logger

MODELS_AVAILABLE = ["neural_network"]


def available_models():
    return MODELS_AVAILABLE


def initialize_model(chosen_model: str, weights_path: str):
    logger = get_logger(__name__)
    logger.info(f"Initialize model {chosen_model} from {weights_path}")
    # make sure weights path is a directory, not a file
    if os.path.isfile(weights_path):
        weights_path = os.path.dirname(weights_path)

    # make sure chosen model is available
    if chosen_model not in MODELS_AVAILABLE:
        logger.exception("Chosen model is not defined in the API")
        raise ValueError(
            f"Chosen model {chosen_model} is not available. Available models: {MODELS_AVAILABLE}"
        )

    # now create path for model weights based on chosen model's name
    weights_path = os.path.join(
        (weights_path or DEFAULT_MODEL_PATH), chosen_model + ".joblib"
    )

    match chosen_model:
        case "neural_network":
            return load_nr_model(weights_path), weights_path