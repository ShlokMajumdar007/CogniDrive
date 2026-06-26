"""Recommendation Router — FastAPI endpoints for driver safety advisory lifecycle.

Exposes three HTTP endpoints:

    GET /recommendation/active                — Retrieve pending/displayed alerts.
    PUT /recommendation/{id}/acknowledge      — Driver silences an alert.
    PUT /recommendation/{id}/dismiss          — Driver permanently dismisses alert.

All operations run entirely offline against the local SQLite database.

Error handling:
    - 404 Not Found    — Recommendation ID does not exist.
    - 503 Unavailable  — Database sessionmaker not available.
    - 500 Server Error — Unexpected persistence failures.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

try:
    from backend.services.recommendation_service import RecommendationService
    from backend.schemas.recommendation_schema import (
        RecommendationResponse,
        RecommendationBatchResponse,
    )
except ImportError:
    from services.recommendation_service import RecommendationService  # type: ignore[no-redef]
    from schemas.recommendation_schema import (  # type: ignore[no-redef]
        RecommendationResponse,
        RecommendationBatchResponse,
    )

logger = logging.getLogger("CogniDrive.RecommendationRouter")

router = APIRouter(
    prefix="/recommendation",
    tags=["Safety Recommendations"],
)


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
        logger.error("RecommendationRouter: DB session error: %s", exc)
        db.rollback()
        raise
    finally:
        db.close()


def _get_recommendation_service(
    db: Annotated[Session, Depends(_get_db)],
) -> RecommendationService:
    return RecommendationService(db=db)


# ---------------------------------------------------------------------------
# GET /recommendation/active
# ---------------------------------------------------------------------------


@router.get(
    "/active",
    response_model=RecommendationBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve active (pending/displayed) safety recommendations for a driver",
    description=(
        "Returns all unacknowledged, non-expired recommendations for the given driver. "
        "Automatically transitions PENDING items to DISPLAYED and purges expired entries."
    ),
    response_description="Active safety recommendations with priority tallies.",
)
async def get_active_recommendations(
    driver_id: int = Query(..., description="Primary key of the target DriverProfile", ge=1),
    service: RecommendationService = Depends(_get_recommendation_service),
) -> RecommendationBatchResponse:
    """Retrieve all active recommendations for a driver.

    Args:
        driver_id: Primary key of the ``DriverProfile``.
        service: Injected ``RecommendationService``.

    Returns:
        RecommendationBatchResponse: Batch payload with counts.

    Raises:
        HTTPException 500: On unexpected DB failures.
    """
    try:
        recs = service.get_active_recommendations(driver_id=driver_id)
    except Exception as exc:
        logger.exception(
            "RecommendationRouter.get_active_recommendations: driver_id=%d error: %s",
            driver_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recommendations.",
        ) from exc

    # Serialize ORM objects → Pydantic dicts for model_validator processing
    serialized = [r.to_dict() for r in recs]
    return RecommendationBatchResponse(recommendations=serialized)


# ---------------------------------------------------------------------------
# PUT /recommendation/{recommendation_id}/acknowledge
# ---------------------------------------------------------------------------


@router.put(
    "/{recommendation_id}/acknowledge",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Acknowledge a safety recommendation",
    description=(
        "Marks the specified recommendation as ACKNOWLEDGED, sets is_read=True, "
        "and records the acknowledgement timestamp.  Use this when the driver "
        "has seen and responded to the alert."
    ),
    response_description="Updated recommendation record.",
)
async def acknowledge_recommendation(
    recommendation_id: int,
    service: RecommendationService = Depends(_get_recommendation_service),
) -> RecommendationResponse:
    """Acknowledge a safety recommendation by ID.

    Args:
        recommendation_id: Primary key of the ``Recommendation`` to acknowledge.
        service: Injected ``RecommendationService``.

    Returns:
        RecommendationResponse: The updated recommendation.

    Raises:
        HTTPException 404: If the recommendation does not exist.
        HTTPException 500: On unexpected persistence failures.
    """
    try:
        rec = service.acknowledge_recommendation(recommendation_id=recommendation_id)
    except Exception as exc:
        logger.exception(
            "RecommendationRouter.acknowledge_recommendation: id=%d error: %s",
            recommendation_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to acknowledge recommendation.",
        ) from exc

    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recommendation id={recommendation_id} not found.",
        )

    return RecommendationResponse(**rec.to_dict())


# ---------------------------------------------------------------------------
# PUT /recommendation/{recommendation_id}/dismiss
# ---------------------------------------------------------------------------


@router.put(
    "/{recommendation_id}/dismiss",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Dismiss a safety recommendation",
    description=(
        "Marks the specified recommendation as DISMISSED and sets is_read=True. "
        "Dismissed recommendations are excluded from active feeds but remain in "
        "the database for post-trip analytics."
    ),
    response_description="Updated recommendation record after dismissal.",
)
async def dismiss_recommendation(
    recommendation_id: int,
    service: RecommendationService = Depends(_get_recommendation_service),
) -> RecommendationResponse:
    """Dismiss a safety recommendation by ID.

    Args:
        recommendation_id: Primary key of the ``Recommendation`` to dismiss.
        service: Injected ``RecommendationService``.

    Returns:
        RecommendationResponse: The updated recommendation record.

    Raises:
        HTTPException 404: If the recommendation does not exist.
        HTTPException 500: On unexpected persistence failures.
    """
    try:
        rec = service.dismiss_recommendation(recommendation_id=recommendation_id)
    except Exception as exc:
        logger.exception(
            "RecommendationRouter.dismiss_recommendation: id=%d error: %s",
            recommendation_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to dismiss recommendation.",
        ) from exc

    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recommendation id={recommendation_id} not found.",
        )

    return RecommendationResponse(**rec.to_dict())
