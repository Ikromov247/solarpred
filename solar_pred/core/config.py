from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    port: int = 8010
    is_dev: bool = False
    model_dir: str = "/app/data/models"
    volume_path: str = "/app/data"
    log_level: str = "INFO"
    file_log_level: Optional[str] = None
    console_log_level: Optional[str] = None

    @field_validator('port')
    @classmethod
    def is_port_valid(cls, v: int) -> int:
        if not 1024 < v <= 65535:
            raise ValueError("Port must be between 1024 and 65535")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

config = Settings()