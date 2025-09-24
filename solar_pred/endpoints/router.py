from fastapi import APIRouter

from solar_pred.endpoints import train, predict, healthcheck

api_router = APIRouter()
api_router.include_router(healthcheck.router, tags=["healthcheck"])
api_router.include_router(train.router, tags=["train"])
api_router.include_router(predict.router, tags=["predict"])