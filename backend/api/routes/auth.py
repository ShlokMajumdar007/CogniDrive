"""Auth Router — FastAPI endpoints for MobileFaceNet-based driver authentication.

Exposes four HTTP endpoints:

    POST /auth/enroll    — Enroll a driver using 1–16 face images.
    POST /auth/verify    — Verify a live face against a known driver's enrollment.
    POST /auth/identify  — Identify an unknown face across all enrolled drivers.
    GET  /auth/health   — Operational health of the MobileFaceNet model + DB.

All endpoints operate completely offline.  Images are transmitted as base64
strings in JSON bodies.  The router uses the project-standard ``get_db``
FastAPI dependency for SQLAlchemy session management.

Error handling:
    - 400 Bad Request  — Invalid image data, validation failures.
    - 404 Not Found    — Driver not enrolled or profile not found.
    - 503 Unavailable  — MobileFaceNet model not loaded (system startup race).
    - 500 Server Error — Unexpected inference or database failures.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Annotated, Any, Dict

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from backend.services.face_auth_service import FaceAuthService
from backend.schemas.auth_schema import (
    AuthenticationHealth,
    EnrollRequest,
    EnrollResponse,
    IdentifyRequest,
    IdentifyResponse,
    DriverCandidate,
    VerifyRequest,
    VerifyResponse,
)
from backend.repositories.face_repository import FaceRepository
from backend.ml.inference.mobilefacenet_model import MobileFaceNetModel

logger = logging.getLogger("CogniDrive.AuthRouter")

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/auth",
    tags=["Driver Authentication"],
)

# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _get_db(request: Request) -> Session:
    """Retrieves an active SQLAlchemy session from the FastAPI application state.

    Args:
        request: Current FastAPI ``Request`` object.

    Returns:
        Session: Active database session.

    Raises:
        HTTPException 503: If the database sessionmaker is not available.
    """
    sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
    if sessionmaker is None:
        logger.error("AuthRouter: Database sessionmaker not found in app state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized. The application is still starting up.",
        )
    db = sessionmaker()
    try:
        yield db
    except Exception as exc:
        logger.error("AuthRouter: Database session exception: %s", exc)
        db.rollback()
        raise
    finally:
        db.close()


def _get_face_auth_service(
    db: Annotated[Session, Depends(_get_db)],
) -> FaceAuthService:
    """FastAPI dependency that constructs a per-request ``FaceAuthService``.

    Args:
        db: Injected database session.

    Returns:
        FaceAuthService: Service instance bound to this request's session.
    """
    return FaceAuthService(db=db)


def _assert_model_available() -> None:
    """Raises HTTP 503 if the MobileFaceNet model is not loaded.

    Raises:
        HTTPException 503: If ``MobileFaceNetModel.get_instance().model_loaded`` is False.
    """
    model = MobileFaceNetModel.get_instance()
    if not model.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "MobileFaceNet model is not loaded.  "
                "Check that mobilefacenet.tflite exists at "
                "backend/ml/models_saved/mobilefacenet.tflite and that "
                "tflite_runtime or tensorflow is installed."
            ),
        )


# ---------------------------------------------------------------------------
# POST /auth/enroll
# ---------------------------------------------------------------------------


@router.post(
    "/enroll",
    response_model=EnrollResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a driver using face images",
    description=(
        "Enroll a driver by submitting 1–16 base64-encoded face crop images.  "
        "MobileFaceNet extracts an embedding from each image; the mean L2-normalized "
        "embedding is persisted as the driver's canonical facial identity.  "
        "A quality filter discards low-similarity frames (e.g. mid-blink, motion blur).  "
        "Enrolling again will replace the existing active enrollment."
    ),
    response_description="Enrollment result including enrollment ID and quality metrics.",
)
async def enroll_driver(
    payload: EnrollRequest,
    service: Annotated[FaceAuthService, Depends(_get_face_auth_service)],
) -> EnrollResponse:
    """Enroll a driver using one or more face crop images.

    Args:
        payload: ``EnrollRequest`` containing driver ID and base64 face images.
        service: Injected ``FaceAuthService`` instance.

    Returns:
        EnrollResponse: Enrollment outcome including enrollment_id, quality, and sample count.

    Raises:
        HTTPException 400: On image decode failure.
        HTTPException 404: If the driver profile does not exist.
        HTTPException 503: If the MobileFaceNet model is not loaded.
        HTTPException 500: On unexpected inference or persistence errors.
    """
    _assert_model_available()

    # Decode all base64 images to numpy arrays
    face_crops = []
    for idx, b64_img in enumerate(payload.face_images_b64):
        try:
            arr = FaceAuthService.decode_b64_to_array(b64_img)
            face_crops.append(arr)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image at index {idx} could not be decoded: {exc}",
            ) from exc

    try:
        result = service.enroll_driver(
            driver_id=payload.driver_id,
            face_crops=face_crops,
            override_existing=payload.override_existing,
            quality_threshold=payload.quality_threshold,
        )
    except ValueError as exc:
        # Driver not found
        if "not found" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("AuthRouter.enroll_driver: Unexpected error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Enrollment failed due to an internal server error.",
        ) from exc

    return EnrollResponse(**result)


# ---------------------------------------------------------------------------
# POST /auth/verify
# ---------------------------------------------------------------------------


@router.post(
    "/verify",
    response_model=VerifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify a live face against a known driver's enrollment",
    description=(
        "Authenticate a live camera face crop against a specific driver's stored "
        "MobileFaceNet embedding.  Returns a similarity score and a boolean "
        "verified flag.  Use this endpoint when the driver profile is known in "
        "advance (e.g. the driver selects their profile on the HMI)."
    ),
    response_description="Verification result including similarity score and threshold.",
)
async def verify_driver(
    payload: VerifyRequest,
    service: Annotated[FaceAuthService, Depends(_get_face_auth_service)],
) -> VerifyResponse:
    """Verify a live face against a specific enrolled driver.

    Args:
        payload: ``VerifyRequest`` with driver ID and live face crop.
        service: Injected ``FaceAuthService``.

    Returns:
        VerifyResponse: Verification result.

    Raises:
        HTTPException 400: On image decode failure.
        HTTPException 404: If the driver has no active enrollment.
        HTTPException 503: If the model is not loaded.
        HTTPException 500: On unexpected errors.
    """
    _assert_model_available()

    # Decode image
    try:
        live_crop = FaceAuthService.decode_b64_to_array(payload.face_image_b64)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode face image: {exc}",
        ) from exc

    try:
        result = service.authenticate_driver(
            driver_id=payload.driver_id,
            live_face_crop=live_crop,
            threshold=payload.threshold,
        )
    except ValueError as exc:
        if "no active enrollment" in str(exc).lower() or "not found" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("AuthRouter.verify_driver: Unexpected error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Verification failed due to an internal server error.",
        ) from exc

    return VerifyResponse(
        verified=result["verified"],
        driver_id=result["driver_id"],
        similarity=result["similarity"],
        threshold=result["threshold"],
        inference_ms=result["inference_ms"],
        quality_score=result["quality_score"],
        message=result["message"],
        verified_at=result.get("verified_at"),
    )


# ---------------------------------------------------------------------------
# POST /auth/identify
# ---------------------------------------------------------------------------


@router.post(
    "/identify",
    response_model=IdentifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Identify an unknown face across all enrolled drivers",
    description=(
        "Given a live face crop, compare its MobileFaceNet embedding against all "
        "enrolled drivers and return the best-matching driver (if above the similarity "
        "threshold).  Use this when the driver is not pre-selected — e.g. a driver "
        "simply sits in the vehicle and the system auto-detects their identity.  "
        "Up to ``max_candidates`` ranked matches are returned in the response."
    ),
    response_description="Identification result with ranked driver candidates.",
)
async def identify_driver(
    payload: IdentifyRequest,
    service: Annotated[FaceAuthService, Depends(_get_face_auth_service)],
) -> IdentifyResponse:
    """Identify an unknown driver by scanning all enrolled embeddings.

    Args:
        payload: ``IdentifyRequest`` with live face crop and threshold settings.
        service: Injected ``FaceAuthService``.

    Returns:
        IdentifyResponse: Identification result with best match and candidates.

    Raises:
        HTTPException 400: On image decode failure.
        HTTPException 503: If the model is not loaded.
        HTTPException 500: On unexpected errors.
    """
    _assert_model_available()

    try:
        live_crop = FaceAuthService.decode_b64_to_array(payload.face_image_b64)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode face image: {exc}",
        ) from exc

    try:
        result = service.identify_driver(
            live_face_crop=live_crop,
            threshold=payload.threshold,
            max_candidates=payload.max_candidates,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("AuthRouter.identify_driver: Unexpected error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Identification failed due to an internal server error.",
        ) from exc

    # Build typed candidates
    candidates = [
        DriverCandidate(
            driver_id=c["driver_id"],
            driver_name=c["driver_name"],
            similarity=c["similarity"],
            above_threshold=c["above_threshold"],
        )
        for c in result.get("candidates", [])
    ]

    best_match_schema: Any = None
    if result.get("best_match"):
        bm = result["best_match"]
        best_match_schema = DriverCandidate(
            driver_id=bm["driver_id"],
            driver_name=bm["driver_name"],
            similarity=bm["similarity"],
            above_threshold=bm["above_threshold"],
        )

    return IdentifyResponse(
        identified=result["identified"],
        best_match=best_match_schema,
        candidates=candidates,
        threshold=result["threshold"],
        inference_ms=result["inference_ms"],
        message=result["message"],
        identified_at=result.get("identified_at"),
    )


# ---------------------------------------------------------------------------
# GET /auth/health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=AuthenticationHealth,
    status_code=status.HTTP_200_OK,
    summary="MobileFaceNet model and enrollment database health check",
    description=(
        "Returns the operational status of the MobileFaceNet TFLite model "
        "and the face enrollment database.  Use this endpoint to monitor "
        "system readiness before starting a session.  "
        "A ``model_loaded: false`` response means the system is not ready "
        "for authentication."
    ),
    response_description="Authentication subsystem health information.",
)
async def auth_health(
    request: Request,
) -> AuthenticationHealth:
    """Returns health status for the authentication subsystem.

    Queries the MobileFaceNet singleton for operational status and the
    database for enrollment statistics.  Does not require authentication.

    Args:
        request: Current FastAPI ``Request`` (used to access app state).

    Returns:
        AuthenticationHealth: Full health payload.
    """
    model = MobileFaceNetModel.get_instance()
    model_info = model.get_model_info()

    # Default enrollment stats (if DB unavailable)
    enrollment_stats: Dict[str, int] = {
        "total_enrollments": 0,
        "active_enrollments": 0,
        "enrolled_drivers": 0,
    }

    # Try to get enrollment counts from the DB
    sessionmaker = getattr(request.app.state, "db_sessionmaker", None)
    if sessionmaker is not None:
        try:
            db = sessionmaker()
            try:
                repo = FaceRepository(db)
                enrollment_stats = repo.get_all_enrollment_stats()
            finally:
                db.close()
        except Exception as exc:
            logger.warning(
                "AuthRouter.auth_health: Failed to query enrollment stats: %s", exc
            )

    return AuthenticationHealth(
        model_loaded=model_info["model_loaded"],
        embedding_dimension=model_info["embedding_dimension"],
        threshold=model_info["verification_threshold"],
        tflite_backend=model_info["tflite_backend"],
        warmup_done=model_info["warmup_done"],
        average_inference_ms=model_info["average_inference_ms"],
        total_enrollments=enrollment_stats["total_enrollments"],
        active_enrollments=enrollment_stats["active_enrollments"],
        enrolled_drivers=enrollment_stats["enrolled_drivers"],
        model_path=model_info["model_path"],
    )


# ---------------------------------------------------------------------------
# POST /auth/enroll/upload  (multipart file upload alternative)
# ---------------------------------------------------------------------------


@router.post(
    "/enroll/upload",
    response_model=EnrollResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a driver using multipart file upload (alternative to base64)",
    description=(
        "Alternative enrollment endpoint for clients that prefer multipart file uploads "
        "over base64 JSON.  Accepts exactly one face image file per request.  "
        "For multi-image calibration, submit multiple requests or use ``POST /auth/enroll``."
    ),
    response_description="Single-sample enrollment result.",
)
async def enroll_driver_upload(
    driver_id: Annotated[int, Form(description="Primary key of the DriverProfile to enroll.")],
    face_image: Annotated[UploadFile, File(description="Face crop image file (JPEG or PNG).")],
    service: Annotated[FaceAuthService, Depends(_get_face_auth_service)],
) -> EnrollResponse:
    """Enroll a driver using a multipart file upload.

    Accepts a single image per request.  Internally converts the uploaded file
    to a NumPy RGB array and delegates to ``FaceAuthService.enroll_driver()``.

    Args:
        driver_id: Primary key of the ``DriverProfile`` being enrolled.
        face_image: Uploaded face image file (JPEG or PNG).
        service: Injected ``FaceAuthService``.

    Returns:
        EnrollResponse: Single-sample enrollment result.

    Raises:
        HTTPException 400: On invalid image data.
        HTTPException 404: If the driver profile does not exist.
        HTTPException 503: If the model is not loaded.
        HTTPException 500: On unexpected errors.
    """
    _assert_model_available()

    # Read file bytes and decode to numpy
    try:
        raw_bytes = await face_image.read()
        import base64 as b64lib
        b64_str = b64lib.b64encode(raw_bytes).decode("utf-8")
        face_crop = FaceAuthService.decode_b64_to_array(b64_str)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode uploaded image: {exc}",
        ) from exc

    try:
        result = service.enroll_driver(
            driver_id=driver_id,
            face_crops=[face_crop],
            override_existing=True,
        )
    except ValueError as exc:
        if "not found" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("AuthRouter.enroll_driver_upload: Unexpected error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Enrollment upload failed due to an internal server error.",
        ) from exc

    return EnrollResponse(**result)
