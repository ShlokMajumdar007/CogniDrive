import logging
from typing import Generator, Optional, Any
from fastapi import Request

from backend.app.config import Settings, get_settings

# Configure logger
logger = logging.getLogger("CogniDrive.Dependencies")

# Global singleton references — populated lazily on first request / lifespan startup
_camera_manager_instance: Optional[Any] = None
_pipeline_manager_instance: Optional[Any] = None
_live_pipeline_runner_instance: Optional[Any] = None


def get_db(request: Request) -> Generator[Any, None, None]:
    """FastAPI dependency to yield database sessions."""
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

    The camera capture thread is NOT started here; the FastAPI lifespan
    handler calls camera_mgr.start() during startup.
    """
    global _camera_manager_instance
    if _camera_manager_instance is None:
        logger.info("Initializing CameraManager singleton instance.")
        try:
            from backend.vision.camera_manager import CameraManager
            settings = get_settings()
            _camera_manager_instance = CameraManager(
                camera_index=settings.CAMERA_INDEX,
                frame_width=settings.FRAME_WIDTH,
                frame_height=settings.FRAME_HEIGHT,
            )
        except Exception as e:
            logger.error(f"Critical error initializing CameraManager: {e}")
            raise RuntimeError(f"Camera manager initialization failed: {e}")
    return _camera_manager_instance


def get_pipeline_manager() -> Any:
    """Dependency to retrieve the PipelineManager singleton."""
    global _pipeline_manager_instance
    if _pipeline_manager_instance is None:
        logger.info("Initializing PipelineManager singleton instance.")
        try:
            from backend.engine.pipeline_manager import PipelineManager
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


def get_live_pipeline_runner() -> Any:
    """Returns the LivePipelineRunner singleton (may be None if not started)."""
    return _live_pipeline_runner_instance


def _init_live_pipeline_runner(db_session: Any) -> Any:
    """Internal helper called by the FastAPI lifespan to create and start the runner.

    Args:
        db_session: An open SQLAlchemy session for the pipeline's DB writes.

    Returns:
        LivePipelineRunner: The started singleton runner.
    """
    global _live_pipeline_runner_instance
    if _live_pipeline_runner_instance is not None:
        logger.warning("LivePipelineRunner already exists — skipping re-creation.")
        return _live_pipeline_runner_instance

    try:
        from backend.engine.live_pipeline_runner import LivePipelineRunner
        settings = get_settings()
        camera_mgr = get_camera_manager()
        pipeline_mgr = get_pipeline_manager()

        _live_pipeline_runner_instance = LivePipelineRunner.get_instance(
            camera_manager=camera_mgr,
            pipeline_manager=pipeline_mgr,
        )
        _live_pipeline_runner_instance.start(
            driver_id=settings.PIPELINE_DRIVER_ID,
            session_id=settings.PIPELINE_SESSION_ID,
            db_session=db_session,
        )
        logger.info("LivePipelineRunner initialized and started via lifespan.")
    except Exception as e:
        logger.error(f"Failed to start LivePipelineRunner: {e}")
        raise RuntimeError(f"LivePipelineRunner startup failed: {e}")

    return _live_pipeline_runner_instance
