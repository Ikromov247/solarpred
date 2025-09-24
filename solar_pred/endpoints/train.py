
from fastapi import APIRouter, HTTPException, status
from starlette.requests import Request

from solar_pred.core.input_validation import TrainingInput
from solar_pred.core.preprocessing.processor import DataProcessor
from solar_pred.core.exceptions import ValidationError, DataProcessingError, ModelTrainingError
from solar_pred.core.logging_config import get_logger

router = APIRouter()

@router.post("/train", name="train")
async def train(
        request: Request,
        input_data: TrainingInput)->dict:
    
    logger = get_logger()
    try:
        # load the model from app state
        model = request.app.state.model
        data_processor = DataProcessor()

        train_data = data_processor.preprocess_training_input(input_data)

        # train the model
        model.fit_model(train_data)
        
        return {
                "status": "OK", 
                "status_code": 200
            }
        
    except (ValidationError, DataProcessingError) as e:
        logger.error(f"Validation error in training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input data provided"
        )
    except ModelTrainingError as e:
        logger.error(f"Model training failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model training failed"
        )
    except Exception as e:
        logger.exception("Unexpected error in training endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )