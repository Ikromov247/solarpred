import traceback

from fastapi import APIRouter, HTTPException, status
from starlette.requests import Request

from solar_pred.core.input_validation import PanelMetadata, PredictionOutput
from solar_pred.core.preprocessing.processor import DataProcessor

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput, name="predict")
def predict(
        request: Request, 
        input_data: PanelMetadata
    )->PredictionOutput:
    
    try:
        # load the model from app state
        model = request.app.state.model
        print(model.is_trained)
        processor = DataProcessor()

        inference_data = processor.preprocess_inference_input(input_data)
        # run prediction
        output = model.predict(inference_data)
        print(output)
        return output
    
    except Exception as e:
        # return error response with details about the error.
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=e.__class__.__name__
        )