from pydantic import BaseModel
from typing import Union, Dict, List
import datetime


class TrainingInput(BaseModel):
    inverter_id:str
    plant_id:str
    solar_power:float
    timestamp: Union[float, int] # yyyymmddhhmmss format, or %Y%m%d%H%M%S in strtime format
    # solar panel coordinates. Used to fetch accurate weather
    latitude: float 
    longitude: float
    altitude: float

class PredictionInput(BaseModel):
    inverter_id:str
    plant_id:str
    # solar panel coordinates. Used to fetch accurate weather
    latitude: float 
    longitude: float
    altitude: float
    # number of days to predict solar panel output
    predict_days: int

class PredictionOutput(BaseModel):
    prediction: List[Dict[str, float]]


class HealthCheckOutput(BaseModel):
    status: str
    is_healthy: bool
    timestamp: datetime.datetime = datetime.datetime.now()