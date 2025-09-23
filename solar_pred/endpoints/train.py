import traceback

from fastapi import APIRouter, HTTPException, status
from starlette.requests import Request

from input_validation.models import TrainingInput

router = APIRouter()

@router.post("/train", name="train")
def train(
        request: Request,
        input_data: list[TrainingInput])->dict:
    try:
        # load the model from app state
        model = request.app.state.model
        
        # train the model
        model.train(input_data)
        
        return {
                "status": "OK", 
                "status_code": 200
            }
    
    except Exception as e:
        # return error response with details about the error.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=e.__class__.__name__
        )