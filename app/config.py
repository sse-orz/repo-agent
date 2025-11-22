from typing import List
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",  # Ignore extra environment variables
    )

    APP_NAME: str = "Repo Agent API"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # API settings
    API_PREFIX: str = "/api/v1"


settings = Settings()
