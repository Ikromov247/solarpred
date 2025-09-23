import traceback

from fastapi import APIRouter, HTTPException, status
from starlette.requests import Request


from input_validation.models import PredictionInput, PredictionOutput

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput, name="predict")
def predict(
        request: Request, 
        input_data: list[PredictionInput]
    )->PredictionOutput:
    
    try:
        # load the model from app state
        model = request.app.state.model
        
        # run prediction
        output = model.predict(input_data)
        
        return output
    
    except Exception as e:
        # return error response with details about the error.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=e.__class__.__name__
        )