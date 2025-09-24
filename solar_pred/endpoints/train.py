import traceback
from fastapi import APIRouter, HTTPException, status
from starlette.requests import Request

from solar_pred.core.input_validation import TrainingInput
from solar_pred.core.preprocessing.processor import DataProcessor

router = APIRouter()

@router.post("/train", name="train")
def train(
        request: Request,
        input_data: TrainingInput)->dict:
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
    
    except Exception as e:
        # return error response with details about the error.
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=e.__class__.__name__
        )