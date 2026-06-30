"""CogniDrive Backend — FastAPI application entry point.

Bootstraps the complete Offline Edge-AI Driver Digital Twin system:

    1. Loads and validates application settings (pydantic-settings / .env).
    2. Initialises the SQLite database in WAL mode and runs DDL schema creation.
    3. Warms up ML models (MobileFaceNet TFLite, XGBoost cognitive model, LightGBM
       risk model, Isolation Forest anomaly engine).
    4. Starts CameraManager automatically (Drishti-style).
    5. Starts LivePipelineRunner — a background thread that continuously reads
       frames and runs ML inference, updating PipelineManager.last_prediction.
    6. Mounts all APIRouters under the canonical ``/api/v1`` prefix.
    7. Configures CORS for the local HMI React / HTML dashboard frontend.
    8. On shutdown: stops the runner, stops the camera, disposes the DB pool.

All operations run 100% offline — no internet, cloud, or external API calls.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import get_settings
from backend.database.session import (
    SessionLocal,
    create_database,
    dispose_engine,
    health_check,
)
from backend.api.routes.auth import router as auth_router
from backend.api.routes.prediction import router as prediction_router
from backend.api.routes.recommendation import router as recommendation_router
from backend.api.routes.dashboard import router as dashboard_router
from backend.app.constants import API_V1_PREFIX

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CogniDrive.Main")


# ---------------------------------------------------------------------------
# ML model warmup helpers (unchanged)
# ---------------------------------------------------------------------------


def _warmup_mobilefacenet() -> None:
    from backend.ml.inference.mobilefacenet_model import MobileFaceNetModel
    model = MobileFaceNetModel.get_instance()
    if model.model_loaded:
        model.warmup()
        logger.info("MobileFaceNet warmup complete.")
    else:
        logger.warning(
            "MobileFaceNet model not loaded — check that mobilefacenet.tflite exists "
            "in backend/ml/models_saved/."
        )


def _warmup_cognitive_model() -> None:
    from backend.ml.inference.cognitive_model import CognitiveModel
    model = CognitiveModel.get_instance()
    logger.info(
        "CognitiveModel warmup: fallback=%s version=%s",
        model.is_fallback,
        model.model_version,
    )


def _warmup_risk_model() -> None:
    from backend.ml.inference.risk_model import RiskModel
    model = RiskModel.get_instance()
    logger.info(
        "RiskModel warmup: fallback=%s version=%s",
        model.is_fallback,
        model.model_version,
    )


def _warmup_anomaly_engine() -> None:
    from backend.ml.anomaly_detection.anomaly_engine import AnomalyEngine
    engine = AnomalyEngine.get_instance()
    logger.info(
        "AnomalyEngine warmup: fallback=%s version=%s",
        engine.is_fallback,
        engine.model_version,
    )


# ---------------------------------------------------------------------------
# FastAPI lifespan — startup + shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage the full application lifecycle.

    Startup order:
        1. Validate settings.
        2. Create DB + tables, store sessionmaker in app.state.
        3. Warm up ML models.
        4. Start CameraManager (opens the physical device).
        5. Start LivePipelineRunner (background ML processing thread).

    Shutdown order:
        1. Stop LivePipelineRunner (signals thread, waits for exit).
        2. Stop CameraManager (releases device).
        3. Dispose DB connection pool.
    """
    settings = get_settings()
    logger.info(
        "Starting %s v%s | DEBUG=%s | PIPELINE_AUTO_START=%s",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.DEBUG,
        settings.PIPELINE_AUTO_START,
    )

    # ── 1. Database ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        create_database()
        app.state.db_sessionmaker = SessionLocal
        db_ok = health_check()
        elapsed = (time.perf_counter() - t0) * 1000
        if db_ok:
            logger.info("Database ready (%.1f ms) — %s", elapsed, settings.DATABASE_URL)
        else:
            logger.error("Database health check failed — metrics may not persist.")
    except Exception as exc:
        logger.critical("Database initialization FAILED: %s", exc)

    # ── 2. ML Model Warmups ───────────────────────────────────────────
    logger.info("Warming up ML models…")
    for warmup_fn in (
        _warmup_mobilefacenet,
        _warmup_cognitive_model,
        _warmup_risk_model,
        _warmup_anomaly_engine,
    ):
        try:
            warmup_fn()
        except Exception as exc:
            logger.warning("Warmup failed for %s: %s", warmup_fn.__name__, exc)

    # ── 3. Camera + Pipeline auto-start (Drishti mode) ────────────────
    if settings.PIPELINE_AUTO_START:
        _start_live_pipeline(app, settings)
    else:
        logger.info(
            "PIPELINE_AUTO_START=False — camera and pipeline will not start automatically. "
            "Use the /prediction/realtime endpoint with a frame payload."
        )

    logger.info("CogniDrive backend ready — all subsystems initialized.")

    yield  # ── Application serving requests ───────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────
    logger.info("Shutting down CogniDrive backend…")

    # Stop live pipeline runner first (it holds a reference to camera + pipeline)
    try:
        from backend.app.dependencies import _live_pipeline_runner_instance
        if _live_pipeline_runner_instance is not None:
            _live_pipeline_runner_instance.stop()
            logger.info("LivePipelineRunner stopped.")
    except Exception as exc:
        logger.error("Error stopping LivePipelineRunner: %s", exc)

    # Stop camera
    try:
        from backend.app.dependencies import _camera_manager_instance
        if _camera_manager_instance is not None:
            _camera_manager_instance.stop()
            logger.info("CameraManager stopped.")
    except Exception:
        pass

    # Dispose DB
    try:
        dispose_engine()
        logger.info("Database connection pool disposed.")
    except Exception as exc:
        logger.error("Error disposing DB engine: %s", exc)

    logger.info("CogniDrive backend shutdown complete.")


def _start_live_pipeline(app: FastAPI, settings: Any) -> None:
    """Helper that starts the camera and the background processing runner.

    Called once during lifespan startup. All failures are caught and logged
    so that the API can still serve requests (degraded mode) even if the
    camera is unavailable.
    """
    from backend.app.dependencies import get_camera_manager, _init_live_pipeline_runner

    # ── Start CameraManager ───────────────────────────────────────────
    camera_started = False
    try:
        camera_mgr = get_camera_manager()
        camera_started = camera_mgr.start()
        if camera_started:
            logger.info(
                "CameraManager started — index=%d, resolution=%dx%d.",
                settings.CAMERA_INDEX,
                settings.FRAME_WIDTH,
                settings.FRAME_HEIGHT,
            )
        else:
            logger.warning(
                "CameraManager could not open camera device %d. "
                "LivePipelineRunner will still start and retry automatically.",
                settings.CAMERA_INDEX,
            )
    except Exception as exc:
        logger.error("CameraManager startup error: %s", exc)

    # ── Start LivePipelineRunner ──────────────────────────────────────
    # Open a long-lived DB session for the pipeline's writes.
    # This session remains open for the lifetime of the runner; the runner
    # itself commits periodically (every 300 frames) so it doesn't hold
    # uncommitted rows indefinitely.
    try:
        pipeline_db_session = SessionLocal()
        app.state.pipeline_db_session = pipeline_db_session  # keep reference for cleanup
        _init_live_pipeline_runner(db_session=pipeline_db_session)
        logger.info(
            "LivePipelineRunner active — driver_id=%d, session_id=%d.",
            settings.PIPELINE_DRIVER_ID,
            settings.PIPELINE_SESSION_ID,
        )
    except Exception as exc:
        logger.error(
            "LivePipelineRunner could not start: %s. "
            "The API will still serve requests; use /prediction/realtime for manual frames.",
            exc,
        )


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="CogniDrive — Offline Edge-AI Driver Digital Twin",
        description=(
            "A 100% offline cognitive risk prediction and driver digital-twin system. "
            "Provides real-time fatigue detection, distraction monitoring, accident risk "
            "scoring, and personalized safety recommendations powered by MobileFaceNet, "
            "XGBoost, and LightGBM — all running on CPU with no internet dependency.\n\n"
            "**Drishti mode**: When PIPELINE_AUTO_START=True (default), the camera opens "
            "automatically on startup and frames are processed continuously in the background."
        ),
        version=settings.APP_VERSION,
        docs_url=f"{API_V1_PREFIX}/docs",
        redoc_url=f"{API_V1_PREFIX}/redoc",
        openapi_url=f"{API_V1_PREFIX}/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ──────────────────────────────────────────────────────
    prefix = API_V1_PREFIX
    app.include_router(auth_router, prefix=prefix)
    app.include_router(prediction_router, prefix=prefix)
    app.include_router(recommendation_router, prefix=prefix)
    app.include_router(dashboard_router, prefix=prefix)

    # ── Root endpoint ─────────────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "service": "CogniDrive",
                "version": settings.APP_VERSION,
                "docs": f"{API_V1_PREFIX}/docs",
            }
        )

    # ── Health endpoint ───────────────────────────────────────────────

    @app.get(f"{prefix}/health", tags=["System"], summary="System health check")
    async def health(request: Request) -> JSONResponse:
        db_healthy = False
        try:
            db_healthy = health_check()
        except Exception:
            pass

        face_model_ok = False
        try:
            from backend.ml.inference.mobilefacenet_model import MobileFaceNetModel
            face_model_ok = MobileFaceNetModel.get_instance().model_loaded
        except Exception:
            pass

        cog_model_ok = False
        try:
            from backend.ml.inference.cognitive_model import CognitiveModel
            cog_model_ok = True
            _ = CognitiveModel.get_instance().is_fallback
        except Exception:
            pass

        risk_model_ok = False
        try:
            from backend.ml.inference.risk_model import RiskModel
            risk_model_ok = True
            _ = RiskModel.get_instance().is_fallback
        except Exception:
            pass

        # Pipeline / camera status
        pipeline_running = False
        camera_running = False
        try:
            from backend.app.dependencies import (
                _live_pipeline_runner_instance,
                _camera_manager_instance,
            )
            if _live_pipeline_runner_instance is not None:
                pipeline_running = _live_pipeline_runner_instance.is_running
            if _camera_manager_instance is not None:
                camera_running = _camera_manager_instance.is_running
        except Exception:
            pass

        all_ok = db_healthy and cog_model_ok and risk_model_ok
        http_status = status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE

        return JSONResponse(
            status_code=http_status,
            content={
                "status": "healthy" if all_ok else "degraded",
                "database": "ok" if db_healthy else "unavailable",
                "face_model": "ok" if face_model_ok else "not_loaded",
                "cognitive_model": "ok" if cog_model_ok else "not_loaded",
                "risk_model": "ok" if risk_model_ok else "not_loaded",
                "camera": "running" if camera_running else "stopped",
                "pipeline": "running" if pipeline_running else "stopped",
            },
        )

    # ── Pipeline status endpoint ──────────────────────────────────────

    @app.get(
        f"{prefix}/pipeline/status",
        tags=["System"],
        summary="Live pipeline status and last prediction snapshot",
    )
    async def pipeline_status(request: Request) -> JSONResponse:
        """Returns the current state of the LivePipelineRunner and the
        most recently processed prediction from PipelineManager.last_prediction."""
        from backend.app.dependencies import (
            _live_pipeline_runner_instance,
            _camera_manager_instance,
            _pipeline_manager_instance,
        )

        runner_running = (
            _live_pipeline_runner_instance is not None
            and _live_pipeline_runner_instance.is_running
        )
        camera_running = (
            _camera_manager_instance is not None
            and _camera_manager_instance.is_running
        )
        last_pred = None
        if _pipeline_manager_instance is not None:
            last_pred = getattr(_pipeline_manager_instance, "last_prediction", None)

        return JSONResponse(
            content={
                "pipeline_running": runner_running,
                "camera_running": camera_running,
                "last_prediction": last_pred,
            }
        )

    # ── Global exception handler ──────────────────────────────────────

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s: %s", request.method, request.url, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected internal server error occurred."},
        )

    return app


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app: FastAPI = create_app()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
        access_log=True,
    )
