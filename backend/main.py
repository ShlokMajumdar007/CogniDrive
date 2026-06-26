"""CogniDrive Backend — FastAPI application entry point.

Bootstraps the complete Offline Edge-AI Driver Digital Twin system:

    1. Loads and validates application settings (pydantic-settings / .env).
    2. Initialises the SQLite database in WAL mode and runs DDL schema creation.
    3. Warms up ML models (MobileFaceNet TFLite, XGBoost cognitive model, LightGBM
       risk model, Isolation Forest anomaly engine) so that first-frame inference
       has no JIT compilation latency spike.
    4. Mounts all APIRouters under the canonical ``/api/v1`` prefix.
    5. Configures CORS for the local HMI React / HTML dashboard frontend.
    6. Registers graceful shutdown handlers to clean up DB connections and camera
       resources.

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

# ---------------------------------------------------------------------------
# Resolve import root — supports both ``python -m backend.main`` (from project
# root) and ``python main.py`` (from inside backend/).
# ---------------------------------------------------------------------------
try:
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
except ImportError:
    from app.config import get_settings  # type: ignore[no-redef]
    from database.session import (  # type: ignore[no-redef]
        SessionLocal,
        create_database,
        dispose_engine,
        health_check,
    )
    from api.routes.auth import router as auth_router  # type: ignore[no-redef]
    from api.routes.prediction import router as prediction_router  # type: ignore[no-redef]
    from api.routes.recommendation import router as recommendation_router  # type: ignore[no-redef]
    from api.routes.dashboard import router as dashboard_router  # type: ignore[no-redef]
    from app.constants import API_V1_PREFIX  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Logging configuration — applied before any module-level loggers are called
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CogniDrive.Main")


# ---------------------------------------------------------------------------
# Internal startup helpers
# ---------------------------------------------------------------------------


def _warmup_mobilefacenet() -> None:
    """Load and warm up the MobileFaceNet TFLite model.

    Runs a dummy inference so that the TFLite interpreter's JIT compilation
    completes before the first real authentication request arrives.
    """
    try:
        from backend.ml.inference.mobilefacenet_model import MobileFaceNetModel  # type: ignore[import]
    except ImportError:
        try:
            from ml.inference.mobilefacenet_model import MobileFaceNetModel  # type: ignore[import,no-redef]
        except ImportError:
            logger.warning("MobileFaceNet warmup skipped — module not found.")
            return

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
    """Load and warm up the XGBoost cognitive load model."""
    try:
        from backend.ml.inference.cognitive_model import CognitiveModel  # type: ignore[import]
    except ImportError:
        try:
            from ml.inference.cognitive_model import CognitiveModel  # type: ignore[import,no-redef]
        except ImportError:
            logger.warning("CognitiveModel warmup skipped — module not found.")
            return

    model = CognitiveModel.get_instance()
    logger.info(
        "CognitiveModel warmup: fallback=%s version=%s",
        model.is_fallback,
        model.model_version,
    )


def _warmup_risk_model() -> None:
    """Load and warm up the LightGBM accident risk model."""
    try:
        from backend.ml.inference.risk_model import RiskModel  # type: ignore[import]
    except ImportError:
        try:
            from ml.inference.risk_model import RiskModel  # type: ignore[import,no-redef]
        except ImportError:
            logger.warning("RiskModel warmup skipped — module not found.")
            return

    model = RiskModel.get_instance()
    logger.info(
        "RiskModel warmup: fallback=%s version=%s",
        model.is_fallback,
        model.model_version,
    )


def _warmup_anomaly_engine() -> None:
    """Load and warm up the Isolation Forest anomaly detection engine."""
    try:
        from backend.ml.anomaly_detection.anomaly_engine import AnomalyEngine  # type: ignore[import]
    except ImportError:
        try:
            from ml.anomaly_detection.anomaly_engine import AnomalyEngine  # type: ignore[import,no-redef]
        except ImportError:
            logger.warning("AnomalyEngine warmup skipped — module not found.")
            return

    engine = AnomalyEngine.get_instance()
    logger.info(
        "AnomalyEngine warmup: fallback=%s version=%s",
        engine.is_fallback,
        engine.model_version,
    )


# ---------------------------------------------------------------------------
# FastAPI lifespan (startup + shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Async context manager managing the full application lifecycle.

    Startup:
        1. Validate settings.
        2. Create SQLite DB + tables.
        3. Store SessionLocal factory in app.state.
        4. Warm up all ML models.

    Shutdown:
        1. Dispose DB connection pool.
        2. Release camera resources (best-effort).
    """
    settings = get_settings()
    logger.info(
        "Starting %s v%s | DEBUG=%s",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.DEBUG,
    )

    # ── Database ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        create_database()
        app.state.db_sessionmaker = SessionLocal
        db_ok = health_check()
        elapsed = (time.perf_counter() - t0) * 1000
        if db_ok:
            logger.info("Database ready (%.1f ms) — %s", elapsed, settings.DATABASE_URL)
        else:
            logger.error("Database health check failed — the system may not persist metrics.")
    except Exception as exc:
        logger.critical("Database initialization FAILED: %s", exc)
        # Do not abort startup — allow health endpoints to report degraded status.

    # ── ML Model Warmups ──────────────────────────────────────────────────
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

    logger.info("CogniDrive backend is ready — all subsystems initialized.")

    yield  # ── Application running ────────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("Shutting down CogniDrive backend…")

    try:
        dispose_engine()
        logger.info("Database connection pool disposed.")
    except Exception as exc:
        logger.error("Error disposing DB engine: %s", exc)

    # Release camera if the dependency was ever loaded
    try:
        from backend.app.dependencies import _camera_manager_instance  # type: ignore[import]
        if _camera_manager_instance is not None:
            _camera_manager_instance.release()
            logger.info("CameraManager released.")
    except Exception:
        pass  # Camera may never have been initialized; silence the error.

    logger.info("CogniDrive backend shutdown complete.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns:
        FastAPI: Fully configured application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="CogniDrive — Offline Edge-AI Driver Digital Twin",
        description=(
            "A 100% offline cognitive risk prediction and driver digital-twin system. "
            "Provides real-time fatigue detection, distraction monitoring, accident risk "
            "scoring, and personalized safety recommendations powered by MobileFaceNet, "
            "XGBoost, and LightGBM — all running on CPU with no internet dependency."
        ),
        version=settings.APP_VERSION,
        docs_url=f"{API_V1_PREFIX}/docs",
        redoc_url=f"{API_V1_PREFIX}/redoc",
        openapi_url=f"{API_V1_PREFIX}/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS — allow local HMI frontend connections ───────────────────────
    # In a hackathon/lab setting the frontend runs on localhost or a local LAN IP.
    # Keep this list restrictive in production deployments.
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

    # ── Routers ───────────────────────────────────────────────────────────
    prefix = API_V1_PREFIX
    app.include_router(auth_router, prefix=prefix)
    app.include_router(prediction_router, prefix=prefix)
    app.include_router(recommendation_router, prefix=prefix)
    app.include_router(dashboard_router, prefix=prefix)

    # ── Root / health endpoints ───────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        """Minimal root endpoint — confirms the server is reachable."""
        return JSONResponse(
            content={
                "service": "CogniDrive",
                "version": settings.APP_VERSION,
                "docs": f"{API_V1_PREFIX}/docs",
            }
        )

    @app.get(f"{prefix}/health", tags=["System"], summary="System health check")
    async def health(request: Request) -> JSONResponse:
        """Returns the operational health of all CogniDrive subsystems.

        Checks database connectivity, ML model load status, and pipeline readiness.
        """
        db_healthy = False
        try:
            db_healthy = health_check()
        except Exception:
            pass

        # MobileFaceNet status
        face_model_ok = False
        try:
            from backend.ml.inference.mobilefacenet_model import MobileFaceNetModel  # type: ignore[import]
            face_model_ok = MobileFaceNetModel.get_instance().model_loaded
        except Exception:
            pass

        # Cognitive / Risk model status (always True when fallback heuristics are active)
        cog_model_ok = False
        try:
            from backend.ml.inference.cognitive_model import CognitiveModel  # type: ignore[import]
            cog_model_ok = True  # fallback heuristics always available
            _ = CognitiveModel.get_instance().is_fallback
        except Exception:
            pass

        risk_model_ok = False
        try:
            from backend.ml.inference.risk_model import RiskModel  # type: ignore[import]
            risk_model_ok = True  # fallback heuristics always available
            _ = RiskModel.get_instance().is_fallback
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
            },
        )

    # ── Global exception handler ──────────────────────────────────────────

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all handler that prevents unhandled exceptions leaking stack traces."""
        logger.exception("Unhandled exception on %s %s: %s", request.method, request.url, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected internal server error occurred."},
        )

    return app


# ---------------------------------------------------------------------------
# Application instance — referenced by uvicorn and test clients
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
