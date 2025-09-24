"""
Methods that are called during API server startup and shutdown.
During startup, you can load the model and start a database connection (depening on your needs)
"""

from typing import Callable
import os
import logging

from fastapi import FastAPI

from core.config import DEFAULT_MODEL_DIR, VOLUME_PATH
from core.choose_models import initialize_model
from core.logging_config import setup_logger

def _startup_model(app: FastAPI) -> None:
    # load model during startup.
    weights_dir = DEFAULT_MODEL_DIR
    model_instance = initialize_model(chosen_model="neural_network", weights_dir=weights_dir)
    app.state.model = model_instance


def _initialize_logger():
    log_file_path = os.path.join(VOLUME_PATH, "logs" , "app.log")
    os.makedirs(log_file_path, exist_ok=True)
    setup_logger(log_file_path=log_file_path, console_log_level=logging.INFO) # switch to logging.DEBUG when debugging


def _shutdown_model(app: FastAPI) -> None:
    app.state.model = None


def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        _initialize_logger()
        _startup_model(app)

    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        _shutdown_model(app)

    return shutdown