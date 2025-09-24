import traceback

from fastapi import APIRouter, HTTPException, status
from starlette.requests import Request

from solar_pred.core.input_validation import PanelMetadata, PredictionOutput
from solar_pred.core.preprocessing.processor import DataProcessor
from solar_pred.core.exceptions import ValidationError, DataProcessingError, ModelTrainingError
from solar_pred.core.logging_config import get_logger

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput, name="predict")
async def predict(
        request: Request, 
        input_data: PanelMetadata
    )->PredictionOutput:
    logger = get_logger()

    try:
        # load the model from app state
        model = request.app.state.model
        print(model.is_trained)
        processor = DataProcessor()

        inference_data = processor.preprocess_inference_input(input_data)
        # run prediction
        output = model.predict(inference_data)
        
        return PredictionOutput(prediction=output)
    
    except (ValidationError, DataProcessingError) as e:
        logger.error(f"Validation error in prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input data provided"
        )
    except Exception as e:
        logger.exception("Unexpected error in prediction endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )