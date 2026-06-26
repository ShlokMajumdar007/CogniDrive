"""Auth Schemas — Pydantic v2 request/response models for face authentication API.

Covers all data contracts for the ``/auth`` router:

    POST /auth/enroll    → EnrollRequest  / EnrollResponse
    POST /auth/verify    → VerifyRequest  / VerifyResponse
    POST /auth/identify  → IdentifyRequest / IdentifyResponse
    GET  /auth/health   → AuthenticationHealth

Image transport:
    Images are base64-encoded strings (``data:image/jpeg;base64,...`` prefix is
    stripped automatically).  For multipart file uploads the route handler decodes
    the ``UploadFile`` and converts it to the same base64 flow before calling the
    service.

Embedding transport:
    Embeddings are serialised as ``List[float]`` in JSON.  The Pydantic validators
    enforce dimension, finiteness, and L2-norm constraints.
"""

from __future__ import annotations

import base64
import binascii
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _decode_base64_image(value: str) -> bytes:
    """Strips the optional data URI prefix and base64-decodes an image string.

    Args:
        value: Raw base64 string, optionally prefixed with
            ``data:image/<type>;base64,``.

    Returns:
        bytes: Raw image bytes (JPEG, PNG, etc.).

    Raises:
        ValueError: If the string is not valid base64.
    """
    if "," in value:
        value = value.split(",", 1)[1]
    try:
        return base64.b64decode(value)
    except (binascii.Error, Exception) as exc:
        raise ValueError(f"Invalid base64 image data: {exc}") from exc


def _validate_embedding_vector(v: List[float], dim: Optional[int] = None) -> List[float]:
    """Validates an embedding vector for dimension and value integrity.

    Args:
        v: Raw float list.
        dim: Expected dimension (128 or 512).  If ``None``, skips dimension check.

    Returns:
        List[float]: The validated vector.

    Raises:
        ValueError: On empty vector, wrong dimension, or non-finite values.
    """
    if not v:
        raise ValueError("Embedding vector cannot be empty.")
    if dim is not None and len(v) != dim:
        raise ValueError(
            f"Expected embedding of dimension {dim}, got {len(v)}."
        )
    arr = np.array(v, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Embedding vector contains NaN or Inf values.")
    return v


# ---------------------------------------------------------------------------
# Enroll
# ---------------------------------------------------------------------------


class EnrollRequest(BaseModel):
    """Request payload for ``POST /auth/enroll``.

    The client submits one or more base64-encoded face crops.  The service will
    extract MobileFaceNet embeddings from each, average them, and store the
    result as the driver's canonical face embedding.

    Attributes:
        driver_id: Primary key of the ``DriverProfile`` being enrolled.
        face_images_b64: One to 16 base64-encoded face crops (JPEG/PNG).
            Each image should be a tightly-cropped, frontal face with minimal
            background, captured from the cabin camera.
        override_existing: If True, any existing active enrollment is replaced.
            Defaults to True.
        quality_threshold: Minimum mean cosine inter-sample similarity required
            for enrollment acceptance.  Defaults to 0.60.
    """

    driver_id: int = Field(
        ...,
        gt=0,
        description="Primary key of the DriverProfile to enroll.",
    )
    face_images_b64: List[str] = Field(
        ...,
        min_length=1,
        max_length=16,
        description=(
            "List of 1–16 base64-encoded face crop images (JPEG or PNG). "
            "Each image must be a frontal, cropped face aligned for MobileFaceNet."
        ),
    )
    override_existing: bool = Field(
        default=True,
        description="Replace the driver's existing active enrollment.",
    )
    quality_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum mean inter-sample similarity required for enrollment acceptance.",
    )

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("face_images_b64", mode="before")
    @classmethod
    def validate_images(cls, v: List[str]) -> List[str]:
        """Validates that each image string can be base64-decoded.

        Args:
            v: List of raw base64 strings.

        Returns:
            List[str]: The validated list.

        Raises:
            ValueError: If any image fails base64 decoding.
        """
        if not v:
            raise ValueError("At least one face image is required for enrollment.")
        for idx, img in enumerate(v):
            try:
                _decode_base64_image(img)
            except ValueError as exc:
                raise ValueError(f"Image at index {idx} is invalid: {exc}") from exc
        return v


class EnrollResponse(BaseModel):
    """Response from ``POST /auth/enroll``.

    Attributes:
        success: True if enrollment completed successfully.
        driver_id: Primary key of the enrolled ``DriverProfile``.
        enrollment_id: Primary key of the new ``FaceEnrollment`` record.
        embedding_dimension: Dimensionality of the stored embedding.
        samples_processed: Number of face crops that passed quality checks.
        quality_score: Overall calibration quality in [0.0, 1.0].
        message: Human-readable status message.
        enrolled_at: UTC timestamp of the enrollment event.
    """

    success: bool
    driver_id: int
    enrollment_id: Optional[int] = Field(default=None)
    embedding_dimension: int = Field(default=0)
    samples_processed: int = Field(default=0)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = Field(default="")
    enrolled_at: Optional[datetime] = Field(default=None)

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


class VerifyRequest(BaseModel):
    """Request payload for ``POST /auth/verify``.

    Verifies whether a live face crop matches a specific driver's enrolled
    embedding.  Use this when the driver ID is known (e.g. the driver manually
    selects their profile before sitting down).

    Attributes:
        driver_id: Primary key of the ``DriverProfile`` to verify against.
        face_image_b64: Single base64-encoded face crop from the live camera.
        threshold: Optional per-request threshold override.  If omitted, the
            service default (0.65) is used.
    """

    driver_id: int = Field(
        ...,
        gt=0,
        description="Primary key of the DriverProfile to verify.",
    )
    face_image_b64: str = Field(
        ...,
        description="Base64-encoded live face crop image (JPEG or PNG).",
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Per-request cosine similarity threshold override. "
            "If None, the service default (0.65) is applied."
        ),
    )

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("face_image_b64", mode="before")
    @classmethod
    def validate_image(cls, v: str) -> str:
        """Validates that the image string is decodable base64.

        Args:
            v: Raw base64 string.

        Returns:
            str: The validated string.
        """
        _decode_base64_image(v)  # Raises ValueError on failure
        return v


class VerifyResponse(BaseModel):
    """Response from ``POST /auth/verify``.

    Attributes:
        verified: True if the live face matches the enrolled driver.
        driver_id: Primary key of the queried ``DriverProfile``.
        similarity: Cosine similarity in [-1.0, 1.0].
        threshold: The threshold used for the verification decision.
        inference_ms: Wall-clock time for embedding extraction in milliseconds.
        quality_score: Embedding quality estimate of the enrolled reference.
        message: Human-readable authentication result message.
        verified_at: UTC timestamp of the verification event.
    """

    verified: bool
    driver_id: int
    similarity: float = Field(ge=-1.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    inference_ms: float = Field(default=0.0, ge=0.0)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = Field(default="")
    verified_at: Optional[datetime] = Field(default=None)

    model_config = ConfigDict(from_attributes=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialises the response to a JSON-safe dictionary.

        Returns:
            Dict[str, Any]: Plain dictionary representation.
        """
        return self.model_dump()


# ---------------------------------------------------------------------------
# Identify
# ---------------------------------------------------------------------------


class IdentifyRequest(BaseModel):
    """Request payload for ``POST /auth/identify``.

    Identifies an unknown face by comparing its embedding against all enrolled
    drivers.  Used when the driver profile is not known in advance.

    Attributes:
        face_image_b64: Single base64-encoded face crop from the live camera.
        threshold: Minimum cosine similarity for a positive identification.
            Defaults to 0.65.
        max_candidates: Maximum number of ranked candidates to return in the
            response (even if they fall below threshold).  Defaults to 3.
    """

    face_image_b64: str = Field(
        ...,
        description="Base64-encoded live face crop for driver identification.",
    )
    threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for positive identification.",
    )
    max_candidates: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of ranked driver candidates to return.",
    )

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("face_image_b64", mode="before")
    @classmethod
    def validate_image(cls, v: str) -> str:
        """Validates that the image string is decodable base64.

        Args:
            v: Raw base64 string.

        Returns:
            str: The validated string.
        """
        _decode_base64_image(v)
        return v


class DriverCandidate(BaseModel):
    """A single ranked identification candidate.

    Attributes:
        driver_id: Primary key of the ``DriverProfile``.
        driver_name: Display name of the driver.
        similarity: Cosine similarity score.
        above_threshold: True if this candidate's similarity meets the threshold.
    """

    driver_id: int
    driver_name: str
    similarity: float = Field(ge=-1.0, le=1.0)
    above_threshold: bool

    model_config = ConfigDict(from_attributes=True)


class IdentifyResponse(BaseModel):
    """Response from ``POST /auth/identify``.

    Attributes:
        identified: True if at least one driver was matched above the threshold.
        best_match: The highest-scoring driver candidate, or ``None``.
        candidates: Ranked list of up to ``max_candidates`` candidates.
        threshold: The threshold applied for this identification.
        inference_ms: Wall-clock embedding extraction latency.
        message: Human-readable result message.
        identified_at: UTC timestamp of the identification event.
    """

    identified: bool
    best_match: Optional[DriverCandidate] = Field(default=None)
    candidates: List[DriverCandidate] = Field(default_factory=list)
    threshold: float = Field(ge=0.0, le=1.0)
    inference_ms: float = Field(default=0.0, ge=0.0)
    message: str = Field(default="")
    identified_at: Optional[datetime] = Field(default=None)

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class AuthenticationHealth(BaseModel):
    """Response from ``GET /auth/health``.

    Provides operational status of the MobileFaceNet model and the enrollment
    database.

    Attributes:
        model_loaded: True if the TFLite interpreter is successfully initialised.
        embedding_dimension: Output dimension of the loaded model (e.g. 128, 512).
        threshold: Default cosine similarity threshold for verification.
        tflite_backend: Active TFLite backend name (``tflite_runtime`` or
            ``tensorflow``).
        warmup_done: True if the model has completed its warm-up pass.
        average_inference_ms: Running average inference latency in milliseconds.
        total_enrollments: Total number of ``FaceEnrollment`` rows in the DB.
        active_enrollments: Number of currently active enrollment rows.
        enrolled_drivers: Number of distinct drivers with an active enrollment.
        model_path: Filesystem path of the loaded ``.tflite`` file.
    """

    model_loaded: bool
    embedding_dimension: int = Field(default=0)
    threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    tflite_backend: str = Field(default="unknown")
    warmup_done: bool = Field(default=False)
    average_inference_ms: float = Field(default=0.0, ge=0.0)
    total_enrollments: int = Field(default=0, ge=0)
    active_enrollments: int = Field(default=0, ge=0)
    enrolled_drivers: int = Field(default=0, ge=0)
    model_path: str = Field(default="")

    model_config = ConfigDict(from_attributes=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialises the health response to a JSON-safe dictionary.

        Returns:
            Dict[str, Any]: Plain dictionary representation.
        """
        return self.model_dump()


# ---------------------------------------------------------------------------
# Embedding-level schemas (for internal service use)
# ---------------------------------------------------------------------------


class EmbeddingPayload(BaseModel):
    """Internal schema for passing a raw embedding vector between service layers.

    Not exposed directly on the API surface.

    Attributes:
        driver_id: Owning driver primary key.
        embedding_vector: L2-normalized float list.
        embedding_dimension: Length of ``embedding_vector``.
        quality_score: Embedding quality in [0.0, 1.0].
        samples_count: Number of samples averaged to produce this embedding.
    """

    driver_id: int = Field(..., gt=0)
    embedding_vector: List[float] = Field(...)
    embedding_dimension: int = Field(..., gt=0)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    samples_count: int = Field(default=1, ge=1)

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="after")
    def validate_vector_integrity(self) -> "EmbeddingPayload":
        """Ensures embedding_vector length matches embedding_dimension.

        Returns:
            EmbeddingPayload: The validated model.

        Raises:
            ValueError: On mismatch, empty vector, or non-finite values.
        """
        _validate_embedding_vector(self.embedding_vector, dim=self.embedding_dimension)
        return self
