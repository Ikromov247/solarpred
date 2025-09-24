from starlette.config import Config
from starlette.datastructures import Secret

# implement pipeline configurations here


# the model is saved in the container volume 
DEFAULT_MODEL_DIR ="/app/data/models"
VOLUME_PATH="/app/data"