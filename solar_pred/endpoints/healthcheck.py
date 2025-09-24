from fastapi import APIRouter, Request, HTTPException, status

from solar_pred.core.input_validation import HealthCheckOutput
from solar_pred.core.logging_config import get_logger
from solar_pred.core.healthcheck import check_filesystem_health, check_model_health
router = APIRouter()


@router.get("/healthcheck", response_model=HealthCheckOutput, name="healthcheck")
async def get_healthcheck(request: Request) -> HealthCheckOutput:
    """Simple health check that only verifies critical components."""
    try:
        # Check only critical components
        model_health = await check_model_health(request.app.state)
        fs_health = await check_filesystem_health()
        
        health_checks = {
            "model": model_health,
            "filesystem": fs_health
        }
        
        # Overall health based on critical services only
        is_healthy = all(
            health["status"] == "healthy" 
            for health in health_checks.values()
        )
        
        response = HealthCheckOutput(
            status="healthy" if is_healthy else "unhealthy",
            is_healthy=is_healthy,
            details=health_checks
        )
        
        if not is_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=response.model_dump()
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )