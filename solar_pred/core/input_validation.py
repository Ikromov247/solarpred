from pydantic import BaseModel
from typing import Union, Dict, List, Optional
import datetime


class PanelMetadata(BaseModel):
    inverter_id:str
    plant_id:str
    # solar panel coordinates. Used to fetch accurate weather
    latitude: float 
    longitude: float
    altitude: float
    predict_days: Optional[int] = None

class PanelOutput(BaseModel):
    timestamp: Union[str, int] # yyyymmddhhmmss format, or %Y%m%d%H%M%S in strtime format
    solar_power:float

class TrainingInput(BaseModel):
    panel_metadata: PanelMetadata
    panel_output: List[PanelOutput]

class PredictionOutput(BaseModel):
    prediction: Dict[str, float]

class HealthCheckOutput(BaseModel):
    status: str
    is_healthy: bool
    timestamp: datetime.datetime = datetime.datetime.now()