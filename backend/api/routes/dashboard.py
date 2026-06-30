"""Dashboard Router — FastAPI endpoints for HMI cockpit dashboard data and live biometrics streaming.

Exposes three endpoints:

    GET  /dashboard/session/{id}/summary        — Aggregated metrics for a session.
    GET  /dashboard/driver/{id}/history         — Historical totals and profile stats.
    GET  /dashboard/metrics/stream              — SSE stream of live biometric events.

All operations run entirely offline.  The SSE endpoint integrates with the
PipelineManager singleton to push sub-second updates to the HMI frontend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Annotated, Any, AsyncGenerator, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.services.dashboard_service import DashboardService
from backend.app.dependencies import get_pipeline_manager

logger = logging.getLogger("CogniDrive.DashboardRouter")

router = APIRouter(
    prefix="/dashboard",
    tags=["HMI Dashboard"],
)

# SSE heartbeat interval in seconds
_SSE_HEARTBEAT_INTERVAL: float = 2.0
# Maximum inactivity duration for an SSE connection (seconds)
_SSE_MAX_IDLE: float = 300.0


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
        logger.error("DashboardRouter: DB session error: %s", exc)
        db.rollback()
        raise
    finally:
        db.close()


def _get_dashboard_service(
    db: Annotated[Session, Depends(_get_db)],
) -> DashboardService:
    return DashboardService(db=db)


# ---------------------------------------------------------------------------
# GET /dashboard/session/{session_id}/summary
# ---------------------------------------------------------------------------


@router.get(
    "/session/{session_id}/summary",
    status_code=status.HTTP_200_OK,
    summary="Retrieve aggregated session statistics for the cockpit dashboard",
    description=(
        "Returns computed session averages (attention, stress, CLI, risk score) "
        "together with event counts (fatigue, distraction, aggression) for the "
        "given session."
    ),
    response_description="Session summary payload.",
)
async def session_summary(
    session_id: int,
    service: DashboardService = Depends(_get_dashboard_service),
) -> Dict[str, Any]:
    try:
        summary = service.get_session_summary(session_id=session_id)
    except Exception as exc:
        logger.exception(
            "DashboardRouter.session_summary: session_id=%d error: %s", session_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session summary.",
        ) from exc

    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session id={session_id} not found.",
        )

    return summary


# ---------------------------------------------------------------------------
# GET /dashboard/driver/{driver_id}/history
# ---------------------------------------------------------------------------


@router.get(
    "/driver/{driver_id}/history",
    status_code=status.HTTP_200_OK,
    summary="Retrieve historical driving analytics for a driver profile",
    description=(
        "Returns lifetime aggregate statistics from the DriverProfile record: "
        "total sessions, total distance, average risk factor, baseline metrics, "
        "and calculated overall risk score."
    ),
    response_description="Driver historical summary payload.",
)
async def driver_history(
    driver_id: int,
    service: DashboardService = Depends(_get_dashboard_service),
) -> Dict[str, Any]:
    try:
        summary = service.get_driver_historical_summary(driver_id=driver_id)
    except Exception as exc:
        logger.exception(
            "DashboardRouter.driver_history: driver_id=%d error: %s", driver_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve driver history.",
        ) from exc

    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Driver id={driver_id} not found.",
        )

    return summary


# ---------------------------------------------------------------------------
# GET /dashboard/metrics/stream  — Server-Sent Events
# ---------------------------------------------------------------------------


async def _biometric_sse_generator(
    driver_id: int,
    request: Request,
) -> AsyncGenerator[str, None]:
    """Async generator that emits SSE events from the PipelineManager's last-frame cache.

    Pushes a JSON payload every 100ms while the HTTP connection is alive.
    Falls back to a heartbeat comment when no new frame data is available.

    The generator reads ``pipeline.last_prediction`` (a plain dict) which is
    updated by PipelineManager.process_frame() after every processed frame.
    It uses the ``frame_number`` key inside that dict to detect new frames and
    avoid re-sending the same prediction twice.
    """
    pipeline = None
    try:
        pipeline = get_pipeline_manager()
    except Exception as exc:
        logger.warning("DashboardRouter SSE: PipelineManager unavailable — %s", exc)

    start_ts = time.monotonic()
    last_frame_number: Optional[int] = None

    while True:
        # Abort if client disconnected
        if await request.is_disconnected():
            logger.debug("DashboardRouter SSE: client disconnected (driver_id=%d).", driver_id)
            break

        # Safety timeout — close stale connections
        if time.monotonic() - start_ts > _SSE_MAX_IDLE:
            yield "event: timeout\ndata: {}\n\n"
            break

        payload: Optional[Dict[str, Any]] = None
        if pipeline is not None:
            try:
                # last_prediction is always a dict or None (set in PipelineManager)
                cached: Optional[Dict[str, Any]] = getattr(pipeline, "last_prediction", None)
                if cached and isinstance(cached, dict):
                    cur_frame = cached.get("frame_number")
                    # Only emit when a genuinely new frame has been processed
                    if cur_frame is not None and cur_frame != last_frame_number:
                        last_frame_number = cur_frame
                        payload = cached
            except Exception as exc:
                logger.debug("DashboardRouter SSE: pipeline read error: %s", exc)

        if payload is not None:
            data = json.dumps(payload, default=str)
            yield f"data: {data}\n\n"
        else:
            # Heartbeat to keep TCP alive
            yield ": heartbeat\n\n"

        await asyncio.sleep(0.1)


@router.get(
    "/metrics/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream live biometric metrics via Server-Sent Events",
    description=(
        "Opens a persistent Server-Sent Events (SSE) connection that pushes "
        "real-time cognitive biometrics (attention, CLI, risk score, driver state) "
        "sourced from the PipelineManager's latest processed frame.  "
        "The stream emits JSON data events at ~10 Hz and heartbeat comments every 2 s. "
        "Connect with EventSource in the HMI browser frontend."
    ),
    response_description="SSE stream of live biometric data events.",
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "SSE stream (text/event-stream)",
        }
    },
)
async def metrics_stream(
    request: Request,
    driver_id: int = Query(..., description="Primary key of the active DriverProfile", ge=1),
) -> StreamingResponse:
    """Open a live SSE stream of biometric metrics from the PipelineManager."""
    generator = _biometric_sse_generator(driver_id=driver_id, request=request)
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
