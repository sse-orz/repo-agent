import json
from typing import List
from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator
import os


class Settings(BaseSettings):
    """Application settings."""

    model_config = ConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"),
        case_sensitive=True,
        extra="ignore",  # Ignore extra environment variables
    )

    APP_NAME: str = "Repo Agent API"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(origin).strip() for origin in parsed if str(origin).strip()]
                except json.JSONDecodeError:
                    pass
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    # API settings
    API_PREFIX: str = "/api/v1"


settings = Settings()
