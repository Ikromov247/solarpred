from pydantic import BaseModel
from typing import Union, Dict, List, Optional
import datetime


class PanelData(BaseModel):
    inverter_id:str
    plant_id:str
    # solar panel coordinates. Used to fetch accurate weather
    latitude: float 
    longitude: float
    altitude: float
    predict_days: Optional[int]

class PanelOutput(BaseModel):
    timestamp: Union[float, int] # yyyymmddhhmmss format, or %Y%m%d%H%M%S in strtime format
    solar_power:float

class TrainingInput(BaseModel):
    panel_data: PanelData
    panel_output: List[PanelOutput]

class PredictionOutput(BaseModel):
    prediction: List[Dict[str, float]]

class HealthCheckOutput(BaseModel):
    status: str
    is_healthy: bool
    timestamp: datetime.datetime = datetime.datetime.now()