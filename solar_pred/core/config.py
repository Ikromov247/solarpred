from starlette.config import Config
from starlette.datastructures import Secret

# implement pipeline configurations here


# DEFAULT_MODEL_PATH = config("DEFAULT_MODEL_PATH")
# the model is saved in the container volume 
DEFAULT_MODEL_PATH ="/app/data/model.joblib"
VOLUME_PATH="/app/data"