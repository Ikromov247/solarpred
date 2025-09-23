import os

import uvicorn
from fastapi import FastAPI

from endpoints.router import api_router
from core.event_handlers import start_app_handler, stop_app_handler

PORT = os.environ.get("ENDPOINT_PORT", 8010)


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
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)