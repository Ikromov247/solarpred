import os


from core.ai_models.neural_network import load_nr_model
from core.logging_config import get_logger

MODELS_AVAILABLE = ["neural_network"]


def available_models():
    return MODELS_AVAILABLE


def initialize_model(chosen_model: str, weights_dir: str):
    logger = get_logger(__name__)
    logger.info(f"Initialize model {chosen_model} from {weights_dir}")
    # make sure weights path is a directory, not a file
    if os.path.isfile(weights_dir):
        weights_dir = os.path.dirname(weights_dir)

    # make sure chosen model is available
    if chosen_model not in MODELS_AVAILABLE:
        logger.exception("Chosen model is not defined in the API")
        raise ValueError(
            f"Chosen model {chosen_model} is not available. Available models: {MODELS_AVAILABLE}"
        )

    match chosen_model:
        case "neural_network":
            return load_nr_model(weights_dir)