# Solar Power Prediction API

ML pipeline for predicting hourly solar panel power output (in KW) using weather data and panel metadata. Provides REST API endpoints for training models and generating forecasts. The predictions from this project can be integrated into an energy management system to manage solar panel usage. 

The project will run on any system that supports docker. GPU access via CUDA and Nvidia CTK is recommended, but not required. 

## Quick Start

1. Start the API server
```bash
docker compose -f docker-compose.base -f docker-compose.prod.yml up --build
```

Expected setup time: ~10 minutes. The main time sinks are container base image download and pytorch installation


2. Train the model
```bash
python quickstart.py --train
```

Expected response time: ~5 seconds on CPU, <1 second on GPU

3. Generate predictions
```bash
python quickstart.py --predict --days=2
```

Expected response time: <1 second on CPU, <1 second on GPU


## Architecture

- FastAPI/uvicorn - REST API server
- Pydantic - Input/output validation
- Docker - Containerized deployment
- GitHub Actions - CI/CD pipeline
- ML framework - Pytorch
- AI model architecture - 4-layer Neural Network with two hidden layers with dropout. This simple architecture works due to a clean dataset and a simple relationship between input data and the target feature (solar panel output in KW)

The system includes logging and persistent model storage via Docker volumes.

## Technical details

Machine Learning
- PyTorch NN for time series prediction
- Feature engineering with weather data integration (openmeteo api)
- scikit-learn preprocessing pipeline

Backend Engineering
- REST API with FastAPI and async endpoints
- Pydantic data validation and config management

DevOps & Infrastructure
- Docker containerization
- Docker Compose for development and production environments
- GitHub Actions CI/CD pipeline
- Environment-based config management

API Endpoints
- `POST /train` - Train model with historical panel data
- `POST /predict` - Generate solar power forecasts
- `GET /health` - System health monitoring


## Possible improvements

Main improvements:
- mlflow logging to track each request
- Model versioning

Code improvements:
- Google style docstrings on main methods
- Type hints
- Code quality check using commit hooks and tools like ruff

