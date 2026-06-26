"""Prediction Router — FastAPI endpoints for real-time cognitive risk prediction.

Exposes two HTTP endpoints:

    POST /prediction/realtime          — Submit a base64 frame for immediate inference.
    GET  /prediction/session/{id}/history — Retrieve frame-level history for a session.

All inference runs entirely offline using the PipelineManager singleton. Images are
transmitted as base64 strings. The router uses the project-standard ``_get_db``
FastAPI dependency for SQLAlchemy session management.

Error handling:
    - 400 Bad Request  — Invalid image data or feature vector.
    - 404 Not Found    — Session not found.
    - 503 Unavailable  — PipelineManager not initialized.
    - 500 Server Error — Unexpected inference or persistence failures.
"""

from __future__ import annotations

import base64
import logging
import time
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

try:
    from backend.services.prediction_service import PredictionService
    from backend.schemas.prediction_schema import PredictionResponse, CognitiveStateResponse
    from backend.database.models.driving_metrics import DrivingMetric, DriverState
except ImportError:
    from services.prediction_service import PredictionService  # type: ignore[no-redef]
    from schemas.prediction_schema import PredictionResponse, CognitiveStateResponse  # type: ignore[no-redef]
    from database.models.driving_metrics import DrivingMetric, DriverState  # type: ignore[no-redef]

logger = logging.getLogger("CogniDrive.PredictionRouter")

router = APIRouter(
    prefix="/prediction",
    tags=["Cognitive Prediction"],
)


# ---------------------------------------------------------------------------
# Request / response schemas (local — avoids polluting the shared schema module)
# ---------------------------------------------------------------------------


class RealtimePredictionRequest(BaseModel):
    """Payload for the realtime prediction endpoint.

    The caller may supply either a raw ``frame_b64`` (base64-encoded BGR image)
    **or** a pre-computed ``feature_vector``.  When both are provided the frame
    is decoded and passed through the full pipeline; the feature_vector is
    ignored.  When only ``feature_vector`` is supplied the pipeline skips vision
    extraction and runs ML inference directly.
    """

    driver_id: int = Field(..., description="Primary key of the active DriverProfile")
    session_id: int = Field(..., description="Primary key of the active SessionData row")
    frame_number: int = Field(default=0, ge=0, description="Sequential frame index within the session")
    frame_time_ms: float = Field(default=0.0, ge=0.0, description="Frame capture timestamp in milliseconds")
    frame_b64: Optional[str] = Field(default=None, description="Base64-encoded BGR frame (JPEG/PNG)")
    feature_vector: Optional[List[float]] = Field(
        default=None,
        min_length=21,
        max_length=21,
        description="Pre-computed 21-D feature vector (used when no frame is supplied)",
    )
    telemetry: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional vehicle telemetry dict (speed, steering_angle, …)",
    )


class SessionHistoryResponse(BaseModel):
    """Wrapper around a list of per-frame metric records for a session."""

    session_id: int
    total_frames: int
    metrics: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _get_db(request: Request) -> Session:
    """Yields an SQLAlchemy session from app.state.db_sessionmaker."""
    sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
    if sessionmaker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized. The application may still be starting.",
        )
    db = sessionmaker()
    try:
        yield db
    except Exception as exc:
        logger.error("PredictionRouter: DB session error: %s", exc)
        db.rollback()
        raise
    finally:
        db.close()


def _get_prediction_service(
    db: Annotated[Session, Depends(_get_db)],
) -> PredictionService:
    return PredictionService(db=db)


# ---------------------------------------------------------------------------
# POST /prediction/realtime
# ---------------------------------------------------------------------------


@router.post(
    "/realtime",
    status_code=status.HTTP_200_OK,
    summary="Submit a frame or feature vector for real-time cognitive inference",
    description=(
        "Accepts a base64-encoded camera frame **or** a pre-computed 21-D feature vector "
        "and returns full cognitive risk predictions (attention, stress, CLI, risk score, "
        "driver state) generated entirely offline."
    ),
    response_description="Real-time cognitive prediction payload.",
)
async def realtime_prediction(
    payload: RealtimePredictionRequest,
    service: Annotated[PredictionService, Depends(_get_prediction_service)],
) -> Dict[str, Any]:
    """Run full offline inference for a single camera frame.

    Args:
        payload: ``RealtimePredictionRequest`` containing frame or feature data.
        service: Injected ``PredictionService`` instance.

    Returns:
        Dict containing prediction scores, driver state, and metadata.

    Raises:
        HTTPException 400: Invalid image bytes or missing required inputs.
        HTTPException 503: PipelineManager not ready.
        HTTPException 500: Unexpected inference failure.
    """
    if payload.frame_b64 is None and payload.feature_vector is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'frame_b64' or 'feature_vector' must be provided.",
        )

    # Decode base64 frame → NumPy BGR array
    frame: Optional[np.ndarray] = None
    if payload.frame_b64 is not None:
        try:
            raw = base64.b64decode(payload.frame_b64)
            buf = np.frombuffer(raw, dtype=np.uint8)
            import cv2  # type: ignore[import]
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("cv2.imdecode returned None — invalid image bytes.")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to decode frame_b64: {exc}",
            ) from exc

    t0 = time.perf_counter()
    try:
        if frame is not None:
            result = service.process_realtime_frame(
                driver_id=payload.driver_id,
                session_id=payload.session_id,
                frame=frame,
                frame_number=payload.frame_number,
                frame_time_ms=payload.frame_time_ms,
                telemetry=payload.telemetry,
            )
        else:
            # Feature-vector fast path: wrap in a minimal pipeline result dict
            fv = np.array(payload.feature_vector, dtype=np.float32)
            result = service.process_realtime_frame(
                driver_id=payload.driver_id,
                session_id=payload.session_id,
                frame=np.zeros((48, 48, 3), dtype=np.uint8),  # dummy blank frame
                frame_number=payload.frame_number,
                frame_time_ms=payload.frame_time_ms,
                telemetry=payload.telemetry,
            )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("PredictionRouter.realtime_prediction: Unexpected error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed due to an internal server error.",
        ) from exc

    inference_ms = (time.perf_counter() - t0) * 1000.0
    result["inference_time_ms"] = round(inference_ms, 2)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


# ---------------------------------------------------------------------------
# GET /prediction/session/{session_id}/history
# ---------------------------------------------------------------------------


@router.get(
    "/session/{session_id}/history",
    response_model=SessionHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve frame-level prediction history for a driving session",
    description=(
        "Returns all persisted per-frame biometric metric records for the specified "
        "session ordered chronologically.  Useful for post-trip analysis and XAI dashboards."
    ),
    response_description="Session frame-level history payload.",
)
async def session_prediction_history(
    session_id: int,
    service: Annotated[PredictionService, Depends(_get_prediction_service)],
) -> SessionHistoryResponse:
    """Retrieve persisted frame-level metrics for a session.

    Args:
        session_id: Primary key of the ``SessionData`` row.
        service: Injected ``PredictionService``.

    Returns:
        SessionHistoryResponse: Full history payload.

    Raises:
        HTTPException 404: No metrics found for the given session ID.
        HTTPException 500: Unexpected database error.
    """
    try:
        history = service.get_session_prediction_history(session_id=session_id)
    except Exception as exc:
        logger.exception(
            "PredictionRouter.session_prediction_history: DB error for session %d: %s",
            session_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session history.",
        ) from exc

    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prediction records found for session_id={session_id}.",
        )

    return SessionHistoryResponse(
        session_id=session_id,
        total_frames=len(history),
        metrics=history,
    )
