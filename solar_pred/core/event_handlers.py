"""
Methods that are called during API server startup and shutdown.
During startup, you can load the model and start a database connection (depening on your needs)
"""

from typing import Callable
import os

from fastapi import FastAPI
from datetime import datetime

from solar_pred.core.config import config
from solar_pred.core.choose_models import initialize_model
from solar_pred.core.logging_config import setup_logger, get_logger

def _startup_model(app: FastAPI) -> None:
    # load model during startup.
    weights_dir = config.model_dir
    model_instance = initialize_model(chosen_model="neural_network", weights_dir=weights_dir)
    app.state.model = model_instance
    app.state.weights_dir = weights_dir


def _initialize_logger():
    log_file_dir = os.path.join(config.volume_path, "logs")
    log_file_path = os.path.join(log_file_dir, "app.log")
    os.makedirs(log_file_dir, exist_ok=True)
    
    setup_logger(
        log_file_path=log_file_path, 
        log_level=config.log_level,
        console_log_level=config.console_log_level,
        file_log_level=config.file_log_level
    ) # switch to logging.DEBUG when debugging


def _shutdown_model(app: FastAPI) -> None:
    logger = get_logger(__name__)
    try:
        if hasattr(app.state, 'model') and app.state.model is not None:
            backup_path = f"{app.state.weights_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            app.state.model.save_model(backup_path)
            app.state.model.save_model(app.state.weights_dir)
            logger.info("Model saved successfully during shutdown")
        app.state.model = None
    except Exception as e:
        logger.error(f"Failed to save model during shutdown: {str(e)}")


def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        _initialize_logger()
        _startup_model(app)

    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        _shutdown_model(app)

    return shutdown