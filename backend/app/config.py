import os
from functools import lru_cache
import logging
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure basic logging for settings startup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CogniDrive.Config")


class Settings(BaseSettings):
    """Application settings for the CogniDrive backend.

    Reads variables from environment or a local .env file.
    Validates directories and database URLs.
    """

    APP_NAME: str = Field(default="CogniDrive", description="Application Name")
    APP_VERSION: str = Field(default="1.0.0", description="Application Version")
    DEBUG: bool = Field(default=False, description="Debug mode flag")

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./cognidrive.db",
        description="SQLite database path",
    )

    # Directories (paths will be created if they do not exist)
    MODEL_DIR: str = Field(default="models_saved", description="Directory to save/load trained ML models")
    DATASET_DIR: str = Field(default="datasets", description="Directory for local training datasets")
    LOG_DIR: str = Field(default="logs", description="Directory for general application logs")

    # Camera & Video Stream settings
    CAMERA_INDEX: int = Field(default=0, description="Local camera/webcam hardware index")
    FRAME_WIDTH: int = Field(default=640, description="Camera frame capture width")
    FRAME_HEIGHT: int = Field(default=480, description="Camera frame capture height")

    # Analytics and History limits
    MAX_HISTORY_SIZE: int = Field(
        default=1800,
        description="Maximum frames of historical driver metrics stored in memory window (e.g. 60 seconds at 30 fps)",
    )

    # ML & Feature configuration
    EMBEDDING_DIMENSION: int = Field(
        default=128,
        description="Dimensionality of driver facial structural embeddings",
    )

    # Configure Pydantic Settings Source
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Ensure that the database is a local SQLite instance for offline compliance."""
        if not v.startswith("sqlite:///"):
            raise ValueError(
                f"CogniDrive requires a local SQLite connection for 100% offline edge compliance. Received: {v}"
            )
        return v

    @field_validator("MODEL_DIR", "DATASET_DIR", "LOG_DIR")
    @classmethod
    def create_directory_if_missing(cls, v: str) -> str:
        """Auto-creates directories if they do not exist locally."""
        try:
            os.makedirs(v, exist_ok=True)
            logger.info(f"Directory verified/created: {v}")
        except Exception as e:
            logger.error(f"Failed to create directory {v}: {e}")
        return v


@lru_cache()
def get_settings() -> Settings:
    """Gets cached global settings instance.

    Uses functools.lru_cache to prevent reading the .env file repeatedly
    across multiple API request cycles.

    Returns:
        Settings: The instantiated and validated Settings object.
    """
    logger.info("Initializing application settings settings from environment/.env")
    return Settings()
