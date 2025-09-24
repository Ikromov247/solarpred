import uvicorn
from fastapi import FastAPI

from solar_pred.endpoints.router import api_router
from solar_pred.core.event_handlers import start_app_handler, stop_app_handler
from solar_pred.core.config import config

def get_api_app() -> FastAPI:
    api_app = FastAPI(title="ML API", version="1.0.0", debug=False)
    api_app.include_router(api_router)

    # add event handlers
    api_app.add_event_handler("startup", start_app_handler(api_app))
    api_app.add_event_handler("shutdown", stop_app_handler(api_app))
    return api_app


app = get_api_app()

if __name__=="__main__":
    """Initialize the endpoint"""
    print("Started the pipeline server")
    uvicorn.run("solar_pred.main:app", host="0.0.0.0", port=config.port, reload=config.is_dev)