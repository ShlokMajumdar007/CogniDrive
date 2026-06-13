import logging
from typing import Generator, Optional, Any
from fastapi import Request

from app.config import Settings, get_settings

# Configure logger
logger = logging.getLogger("CogniDrive.Dependencies")

# Global variables for singletons (initialized dynamically or in lifespans)
_camera_manager_instance: Optional[Any] = None
_pipeline_manager_instance: Optional[Any] = None


def get_db(request: Request) -> Generator[Any, None, None]:
    """FastAPI dependency to yield database sessions.

    Expects the database sessionmaker or database engine to be stored in the
    FastAPI application state. This ensures standard connection pooling, thread
    safety, and clean cleanup after each request.

    Args:
        request: The current FastAPI Request object.

    Yields:
        Session: A database session from SQLAlchemy.
    """
    # Since we store our sessionmaker in app.state.db_sessionmaker, retrieve it
    sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
    if sessionmaker is None:
        logger.error("Database sessionmaker not found in application state.")
        raise RuntimeError("Database not properly initialized on app start.")

    db = sessionmaker()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database transaction exception, rolling back: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def get_camera_manager() -> Any:
    """Dependency to retrieve the CameraManager singleton.

    Maintains a single CameraManager instance to prevent multiple processes
    attempting to bind to the system camera hardware concurrently.

    Returns:
        CameraManager: The initialized singleton instance.
    """
    global _camera_manager_instance
    if _camera_manager_instance is None:
        logger.info("Initializing CameraManager singleton instance.")
        try:
            from vision.camera_manager import CameraManager
            settings = get_settings()
            _camera_manager_instance = CameraManager(
                camera_index=settings.CAMERA_INDEX,
                width=settings.FRAME_WIDTH,
                height=settings.FRAME_HEIGHT,
            )
        except Exception as e:
            logger.error(f"Critical error initializing CameraManager: {e}")
            raise RuntimeError(f"Camera manager initialization failed: {e}")
    return _camera_manager_instance


def get_pipeline_manager() -> Any:
    """Dependency to retrieve the PipelineManager singleton.

    Orchestrates the lifecycle of frames passing from the camera,
    through vision feature extractions, into ML models, and to DB services.

    Returns:
        PipelineManager: The initialized singleton instance.
    """
    global _pipeline_manager_instance
    if _pipeline_manager_instance is None:
        logger.info("Initializing PipelineManager singleton instance.")
        try:
            from engine.pipeline_manager import PipelineManager
            camera_mgr = get_camera_manager()
            settings = get_settings()
            _pipeline_manager_instance = PipelineManager(
                camera_manager=camera_mgr,
                settings=settings,
            )
        except Exception as e:
            logger.error(f"Critical error initializing PipelineManager: {e}")
            raise RuntimeError(f"Pipeline manager initialization failed: {e}")
    return _pipeline_manager_instance
