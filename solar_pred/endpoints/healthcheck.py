from fastapi import APIRouter

from core.input_validation import HealthCheckOutput

router = APIRouter()


@router.get("/healthcheck", response_model=HealthCheckOutput, name="healthcheck")
def get_heathcheck() -> HealthCheckOutput:
    # check API health
    """
    is_healthy = True
    health_status = {
        "database": "healthy",
        "api": "healthy",
        "model": "healthy"
    }
    """
    ## check database connection
    """
    try:
        # Check database connection
        await check_database_connection()
    except Exception as e:
        health_status["database"] = f"unhealthy: {str(e)}"
        is_healthy = False
    """

    ## check model output
    """
    try:
        # Check model status
        await check_model_status()
    except Exception as e:
        health_status["model"] = f"unhealthy: {str(e)}"
        is_healthy = False
    """

    ## construct 
    response = HealthCheckOutput(
                status="healthy", # status = "healthy" if is_healthy else "unhealthy"
                is_healthy=True # is_healthy
                )
    
    # if there is an error in healthcheck, send exception:
    """
    if not is_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.dict() # convert pydantic model to dict
        )
    """
    return response