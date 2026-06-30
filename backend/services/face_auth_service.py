"""FaceAuthService — Driver Authentication Orchestration using MobileFaceNet.

This service is the single point of authority for all facial authentication
decisions in CogniDrive.  It sits between the FastAPI route layer (which
handles HTTP) and the repository/model layers (which handle persistence and
inference), coordinating the full authentication pipeline:

    Camera Frame
        ↓  (face detection + cropping done upstream, e.g. by MediaPipe)
    RGB Face Crop (np.ndarray)
        ↓  FaceAuthService.authenticate_driver()
    MobileFaceNetModel.extract_embedding()
        ↓
    FaceRepository.get_embedding_by_driver()
        ↓
    MobileFaceNetModel.verify_faces()
        ↓
    FaceVerificationResult
        ↓  authenticated → load DriverProfile → initialize Digital Twin

Enrollment flow::

    FaceAuthService.enroll_driver(driver_id, face_crops)
        ↓
    MobileFaceNetModel.extract_embeddings_batch()
        ↓  (quality filter + mean average)
    normalized_mean_embedding
        ↓
    FaceRepository.create_enrollment()
        ↓
    DB: FaceEnrollment + DriverEmbedding

Identification (unknown driver) flow::

    FaceAuthService.identify_driver(live_face_crop)
        ↓
    MobileFaceNetModel.extract_embedding()
        ↓
    FaceRepository.find_closest_driver()
        ↓
    Best-match driver_id if similarity ≥ threshold

All operations are offline and CPU-only.

Note on image formats:
    All public methods accept ``np.ndarray`` images in **RGB** channel order
    (H×W×3, uint8).  The ``FaceAuthService.decode_b64_to_array()`` static helper
    converts base64 API inputs to this format.
"""

from __future__ import annotations

import base64
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from backend.ml.inference.mobilefacenet_model import (
    MobileFaceNetModel,
    FaceVerificationResult,
)
from backend.repositories.face_repository import FaceRepository
from backend.database.models.driver_profile import DriverProfile
from backend.database.models.face_enrollment import FaceEnrollment
from backend.database.models.embeddings import DriverEmbedding

# Optional cv2 for image decode
try:
    import cv2  # type: ignore[import]
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

logger = logging.getLogger("CogniDrive.FaceAuthService")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default cosine similarity threshold for a positive face match.
DEFAULT_THRESHOLD: float = 0.65

#: Minimum quality score (inter-sample similarity) for enrollment acceptance.
MIN_ENROLL_QUALITY: float = 0.50

#: Minimum number of face samples to compute a reliable mean embedding.
MIN_ENROLL_SAMPLES: int = 1


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class FaceAuthService:
    """Orchestrates MobileFaceNet-based driver enrollment and authentication.

    Instantiate once per request via the FastAPI ``Depends`` pattern, injecting
    a fresh SQLAlchemy session.  The ``MobileFaceNetModel`` singleton is
    retrieved internally and shared across all requests.

    Attributes:
        _db: The active SQLAlchemy session for this request lifecycle.
        _repo: FaceRepository instance wrapping the session.
        _model: Shared MobileFaceNetModel singleton.
        _threshold: Default cosine similarity verification threshold.
    """

    def __init__(
        self,
        db: Session,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        """Initialises the service with a database session and configuration.

        Args:
            db: Active SQLAlchemy ``Session`` injected by the FastAPI dependency.
            threshold: Default cosine similarity cut-off.  Defaults to 0.65.
        """
        self._db: Session = db
        self._repo: FaceRepository = FaceRepository(db)
        self._model: MobileFaceNetModel = MobileFaceNetModel.get_instance()
        self._threshold: float = threshold

    # --------------------------------------------------------------------------
    # Enrollment
    # --------------------------------------------------------------------------

    def enroll_driver(
        self,
        driver_id: int,
        face_crops: List[np.ndarray],
        *,
        override_existing: bool = True,
        quality_threshold: float = MIN_ENROLL_QUALITY,
    ) -> Dict[str, Any]:
        """Enrolls a driver by extracting and averaging MobileFaceNet embeddings.

        Enrollment pipeline:
            1. Validate inputs.
            2. Extract one embedding per face crop via MobileFaceNet.
            3. Filter crops whose pairwise cosine similarity to the mean is below
               ``quality_threshold`` (removes outliers, e.g. mid-blink frames).
            4. Compute the mean of accepted embeddings and L2-normalize.
            5. Persist via ``FaceRepository.create_enrollment()``.
            6. Optionally link the embedding back to the ``DriverProfile``.

        Args:
            driver_id: Primary key of the ``DriverProfile`` to enroll.
            face_crops: List of 1–16 RGB face crop arrays (H×W×3, uint8).
            override_existing: Replace any existing active enrollment.
                Defaults to True.
            quality_threshold: Minimum mean inter-sample similarity required.
                Crops below this threshold are discarded.  Defaults to 0.50.

        Returns:
            Dict[str, Any]: Enrollment result with keys:
                - ``success`` (bool)
                - ``driver_id`` (int)
                - ``enrollment_id`` (Optional[int])
                - ``embedding_dimension`` (int)
                - ``samples_processed`` (int)
                - ``quality_score`` (float)
                - ``message`` (str)
                - ``enrolled_at`` (Optional[datetime])

        Raises:
            RuntimeError: If the MobileFaceNet model is not loaded.
            ValueError: If ``face_crops`` is empty or ``driver_id`` is invalid.
        """
        self._model._assert_model_loaded()

        if not face_crops:
            raise ValueError(
                "FaceAuthService.enroll_driver: face_crops list cannot be empty."
            )

        logger.info(
            "FaceAuthService.enroll_driver: Starting enrollment for driver_id=%d "
            "with %d face crop(s).",
            driver_id,
            len(face_crops),
        )

        # Step 1: Extract embeddings for all crops
        t0 = time.perf_counter()
        raw_embeddings = self._model.extract_embeddings_batch(face_crops)
        extraction_ms = (time.perf_counter() - t0) * 1000.0

        logger.debug(
            "FaceAuthService.enroll_driver: Extracted %d embedding(s) in %.1f ms.",
            len(raw_embeddings),
            extraction_ms,
        )

        # Step 2: Compute provisional mean for quality filtering
        stacked = np.stack(raw_embeddings, axis=0)  # (N, D)
        provisional_mean = stacked.mean(axis=0)

        # Step 3: Filter by cosine similarity to provisional mean
        accepted_embeddings: List[np.ndarray] = []
        for idx, emb in enumerate(raw_embeddings):
            sim = self._model.cosine_similarity(emb, provisional_mean)
            if sim >= quality_threshold or len(raw_embeddings) == 1:
                accepted_embeddings.append(emb)
                logger.debug(
                    "FaceAuthService.enroll_driver: Crop %d accepted (sim=%.4f).",
                    idx,
                    sim,
                )
            else:
                logger.debug(
                    "FaceAuthService.enroll_driver: Crop %d rejected (sim=%.4f < %.2f).",
                    idx,
                    sim,
                    quality_threshold,
                )

        if not accepted_embeddings:
            logger.warning(
                "FaceAuthService.enroll_driver: All crops rejected by quality filter "
                "for driver_id=%d.",
                driver_id,
            )
            return {
                "success": False,
                "driver_id": driver_id,
                "enrollment_id": None,
                "embedding_dimension": self._model.get_embedding_dimension(),
                "samples_processed": 0,
                "quality_score": 0.0,
                "message": (
                    f"Enrollment failed: all {len(face_crops)} face crop(s) were rejected "
                    f"by the quality filter (threshold={quality_threshold:.2f}). "
                    "Ensure the face is well-lit, frontal, and not occluded."
                ),
                "enrolled_at": None,
            }

        # Step 4: Final mean embedding + L2 normalization
        accepted_stack = np.stack(accepted_embeddings, axis=0)
        mean_embedding = accepted_stack.mean(axis=0).astype(np.float32)
        mean_embedding = self._model.normalize_embedding(mean_embedding)

        # Step 5: Compute quality score (mean pairwise similarity to mean)
        quality_scores = [
            self._model.cosine_similarity(e, mean_embedding) for e in accepted_embeddings
        ]
        quality_score = float(np.mean(quality_scores)) if quality_scores else 0.0
        samples_processed = len(accepted_embeddings)

        # Step 6: Persist enrollment
        enrollment = self._repo.create_enrollment(
            driver_id=driver_id,
            embedding_vector=mean_embedding.tolist(),
            samples_count=samples_processed,
            quality_score=quality_score,
        )

        # Step 7: Update DriverProfile.embedding_id backref
        driver = self._db.get(DriverProfile, driver_id)
        if driver is not None and enrollment.embedding_id is not None:
            driver.embedding_id = enrollment.embedding_id

        self._db.commit()

        enrolled_at = datetime.now(timezone.utc)
        logger.info(
            "FaceAuthService.enroll_driver: ✓ Enrolled driver_id=%d "
            "(enrollment_id=%d, samples=%d, quality=%.3f, dim=%d).",
            driver_id,
            enrollment.id,
            samples_processed,
            quality_score,
            self._model.get_embedding_dimension(),
        )

        return {
            "success": True,
            "driver_id": driver_id,
            "enrollment_id": enrollment.id,
            "embedding_dimension": self._model.get_embedding_dimension(),
            "samples_processed": samples_processed,
            "quality_score": round(quality_score, 4),
            "message": (
                f"Driver enrolled successfully using {samples_processed} face sample(s) "
                f"(quality={quality_score:.2f})."
            ),
            "enrolled_at": enrolled_at,
        }

    # --------------------------------------------------------------------------
    # Authentication (driver ID known)
    # --------------------------------------------------------------------------

    def authenticate_driver(
        self,
        driver_id: int,
        live_face_crop: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Authenticates a live face crop against a known driver's enrollment.

        Use this method when the driver has been pre-selected (e.g. the driver
        taps their profile on the HMI before sitting down).

        Pipeline:
            1. Load the driver's active enrollment embedding.
            2. Extract MobileFaceNet embedding from the live face crop.
            3. Compute cosine similarity.
            4. Return verification result.
            5. If verified, mark the enrollment as verified and load the
               DriverProfile for Digital Twin initialization.

        Args:
            driver_id: Primary key of the ``DriverProfile`` to authenticate.
            live_face_crop: RGB face crop from the live camera (H×W×3, uint8).
            threshold: Per-request similarity threshold override.  If ``None``,
                ``self._threshold`` (default 0.65) is used.

        Returns:
            Dict[str, Any]: Authentication result with keys:
                - ``verified`` (bool)
                - ``driver_id`` (int)
                - ``similarity`` (float)
                - ``threshold`` (float)
                - ``inference_ms`` (float)
                - ``quality_score`` (float)
                - ``message`` (str)
                - ``verified_at`` (Optional[datetime])
                - ``driver_profile`` (Optional[Dict]) — populated when verified

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If the driver is not enrolled.
        """
        self._model._assert_model_loaded()
        effective_threshold = threshold if threshold is not None else self._threshold

        # Load enrolled embedding
        enrolled_emb = self._repo.get_embedding_by_driver(driver_id)
        if enrolled_emb is None:
            raise ValueError(
                f"FaceAuthService.authenticate_driver: Driver {driver_id} has no active "
                "enrollment.  Call /auth/enroll first."
            )

        enrolled_vector = enrolled_emb.to_numpy()

        # Extract live embedding
        t0 = time.perf_counter()
        live_embedding = self._model.extract_embedding(live_face_crop)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # Verify
        result: FaceVerificationResult = self._model.verify_faces(
            enrolled_vector, live_embedding, threshold=effective_threshold
        )

        verified_at: Optional[datetime] = None
        driver_profile_dict: Optional[Dict[str, Any]] = None

        if result.verified:
            verified_at = datetime.now(timezone.utc)

            # Mark enrollment as verified
            enrollment = self._repo.get_active_enrollment(driver_id)
            if enrollment is not None:
                enrollment.mark_verified()

            # Load driver profile for Digital Twin initialization
            driver_profile_dict = self._load_driver_profile_dict(driver_id)
            self._db.commit()

            logger.info(
                "FaceAuthService.authenticate_driver: ✓ Driver %d AUTHENTICATED "
                "(similarity=%.4f, threshold=%.2f, inference=%.1f ms).",
                driver_id,
                result.similarity,
                effective_threshold,
                inference_ms,
            )
        else:
            logger.warning(
                "FaceAuthService.authenticate_driver: ✗ Driver %d REJECTED "
                "(similarity=%.4f, threshold=%.2f, inference=%.1f ms).",
                driver_id,
                result.similarity,
                effective_threshold,
                inference_ms,
            )

        return {
            "verified": result.verified,
            "driver_id": driver_id,
            "similarity": round(result.similarity, 6),
            "threshold": effective_threshold,
            "inference_ms": round(inference_ms, 3),
            "quality_score": enrolled_emb.embedding_quality_score,
            "message": (
                f"Driver {'authenticated' if result.verified else 'rejected'} "
                f"(similarity={result.similarity:.4f})."
            ),
            "verified_at": verified_at,
            "driver_profile": driver_profile_dict,
        }

    # --------------------------------------------------------------------------
    # Verification (embedding-to-embedding, no DB lookup)
    # --------------------------------------------------------------------------

    def verify_driver(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
        threshold: Optional[float] = None,
    ) -> FaceVerificationResult:
        """Performs a direct embedding-to-embedding verification.

        Does not touch the database.  Useful for testing, calibration pipelines,
        and internal service calls where embeddings are already available in memory.

        Args:
            embedding_a: First L2-normalized face embedding, shape ``(D,)``.
            embedding_b: Second L2-normalized face embedding, shape ``(D,)``.
            threshold: Similarity threshold override.  Defaults to ``self._threshold``.

        Returns:
            FaceVerificationResult: Verification result dataclass.
        """
        effective_threshold = threshold if threshold is not None else self._threshold
        return self._model.verify_faces(embedding_a, embedding_b, threshold=effective_threshold)

    # --------------------------------------------------------------------------
    # Identification (unknown driver)
    # --------------------------------------------------------------------------

    def identify_driver(
        self,
        live_face_crop: np.ndarray,
        threshold: Optional[float] = None,
        max_candidates: int = 3,
    ) -> Dict[str, Any]:
        """Identifies an unknown face by scanning all enrolled drivers.

        Extracts a MobileFaceNet embedding from the live crop, then runs a
        cosine similarity scan against every active enrollment in the database.
        Returns the top-N candidates ranked by similarity.

        Args:
            live_face_crop: RGB face crop from the live camera (H×W×3, uint8).
            threshold: Minimum similarity for a positive identification.
                Defaults to ``self._threshold``.
            max_candidates: Maximum number of ranked candidates to return.
                Defaults to 3.

        Returns:
            Dict[str, Any]: Identification result with keys:
                - ``identified`` (bool)
                - ``best_match`` (Optional[Dict]) — best driver if above threshold
                - ``candidates`` (List[Dict]) — top-N ranked candidates
                - ``threshold`` (float)
                - ``inference_ms`` (float)
                - ``message`` (str)
                - ``identified_at`` (Optional[datetime])

        Raises:
            RuntimeError: If the model is not loaded.
        """
        self._model._assert_model_loaded()
        effective_threshold = threshold if threshold is not None else self._threshold

        # Extract embedding from live crop
        t0 = time.perf_counter()
        live_embedding = self._model.extract_embedding(live_face_crop)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # Find all ranked candidates
        candidates = self._find_ranked_candidates(
            live_embedding, threshold=effective_threshold, max_candidates=max_candidates
        )

        identified = bool(candidates and candidates[0]["above_threshold"])
        best_match: Optional[Dict[str, Any]] = candidates[0] if identified else None

        if identified and best_match:
            # Mark the winning driver's enrollment as verified
            winning_driver_id = best_match["driver_id"]
            enrollment = self._repo.get_active_enrollment(winning_driver_id)
            if enrollment:
                enrollment.mark_verified()
                self._db.commit()

            best_match["driver_profile"] = self._load_driver_profile_dict(winning_driver_id)

            logger.info(
                "FaceAuthService.identify_driver: ✓ Identified driver_id=%d "
                "(similarity=%.4f, inference=%.1f ms).",
                winning_driver_id,
                best_match["similarity"],
                inference_ms,
            )
        else:
            logger.info(
                "FaceAuthService.identify_driver: ✗ No driver identified above "
                "threshold=%.2f (inference=%.1f ms).",
                effective_threshold,
                inference_ms,
            )

        return {
            "identified": identified,
            "best_match": best_match,
            "candidates": candidates,
            "threshold": effective_threshold,
            "inference_ms": round(inference_ms, 3),
            "message": (
                f"Driver identified: {best_match['driver_name']}"
                if identified and best_match
                else "No driver matched the face above the similarity threshold."
            ),
            "identified_at": datetime.now(timezone.utc) if identified else None,
        }

    def _find_ranked_candidates(
        self,
        query_embedding: np.ndarray,
        threshold: float,
        max_candidates: int,
    ) -> List[Dict[str, Any]]:
        """Scans all active enrollments and returns ranked similarity candidates.

        Args:
            query_embedding: L2-normalized live face embedding.
            threshold: Similarity threshold for ``above_threshold`` flag.
            max_candidates: Maximum results to return.

        Returns:
            List[Dict[str, Any]]: Ranked list (descending similarity) of candidates.
                Each dict has: driver_id, driver_name, similarity, above_threshold.
        """
        from sqlalchemy import func

        active_enrollments = (
            self._db.query(FaceEnrollment)
            .filter(
                FaceEnrollment.is_active.is_(True),
                FaceEnrollment.embedding_id.isnot(None),
            )
            .all()
        )

        if not active_enrollments:
            return []

        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-8:
            return []

        scored: List[Tuple[int, str, float]] = []

        for enrollment in active_enrollments:
            emb_record = self._db.get(DriverEmbedding, enrollment.embedding_id)
            if emb_record is None:
                continue

            stored = emb_record.to_numpy().ravel()
            s_norm = float(np.linalg.norm(stored))
            if s_norm < 1e-8:
                continue

            similarity = float(np.clip(np.dot(q, stored) / (q_norm * s_norm), -1.0, 1.0))

            driver = self._db.get(DriverProfile, enrollment.driver_id)
            driver_name = driver.name if driver else f"Driver #{enrollment.driver_id}"
            scored.append((enrollment.driver_id, driver_name, similarity))

        # Sort descending by similarity
        scored.sort(key=lambda x: x[2], reverse=True)

        return [
            {
                "driver_id": driver_id,
                "driver_name": driver_name,
                "similarity": round(similarity, 6),
                "above_threshold": similarity >= threshold,
            }
            for driver_id, driver_name, similarity in scored[:max_candidates]
        ]

    # --------------------------------------------------------------------------
    # Embedding management
    # --------------------------------------------------------------------------

    def update_driver_embedding(
        self,
        driver_id: int,
        new_face_crop: np.ndarray,
    ) -> Dict[str, Any]:
        """Incrementally updates a driver's stored embedding with a new face crop.

        Uses the weighted incremental update formula::

            updated = normalize((old_embedding * n + new_embedding) / (n + 1))

        This allows the embedding to drift slowly toward the driver's current
        appearance (e.g. seasonal changes, haircuts) without requiring a full
        re-enrollment.

        Args:
            driver_id: Primary key of the ``DriverProfile``.
            new_face_crop: RGB face crop from the live camera (H×W×3, uint8).

        Returns:
            Dict[str, Any]: Update result with keys:
                - ``success`` (bool)
                - ``driver_id`` (int)
                - ``new_samples_count`` (int)
                - ``quality_score`` (float)
                - ``message`` (str)

        Raises:
            RuntimeError: If the model is not loaded.
        """
        self._model._assert_model_loaded()

        new_embedding = self._model.extract_embedding(new_face_crop)

        enrolled_emb = self._repo.get_embedding_by_driver(driver_id)
        if enrolled_emb is None:
            logger.warning(
                "FaceAuthService.update_driver_embedding: driver_id=%d has no enrollment. "
                "Use enroll_driver() first.",
                driver_id,
            )
            return {
                "success": False,
                "driver_id": driver_id,
                "new_samples_count": 0,
                "quality_score": 0.0,
                "message": "No active enrollment found. Call /auth/enroll first.",
            }

        new_n = enrolled_emb.calibration_samples + 1
        similarity = self._model.cosine_similarity(enrolled_emb.to_numpy(), new_embedding)

        updated_emb = self._repo.update_embedding(
            driver_id=driver_id,
            new_vector=new_embedding,
            quality_score=float(np.clip(similarity, 0.0, 1.0)),
            samples_count=new_n,
        )

        self._db.commit()

        logger.info(
            "FaceAuthService.update_driver_embedding: driver_id=%d updated "
            "(samples=%d, similarity=%.4f).",
            driver_id,
            new_n,
            similarity,
        )

        return {
            "success": True,
            "driver_id": driver_id,
            "new_samples_count": new_n,
            "quality_score": float(np.clip(similarity, 0.0, 1.0)),
            "message": (
                f"Embedding updated incrementally (samples={new_n}, similarity={similarity:.4f})."
            ),
        }

    def find_closest_driver(
        self,
        query_embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Optional[Tuple[int, float]]:
        """Returns the driver ID and similarity of the best-matching enrolled driver.

        Delegates to ``FaceRepository.find_closest_driver()``.

        Args:
            query_embedding: L2-normalized query embedding, shape ``(D,)``.
            threshold: Similarity threshold.  Defaults to ``self._threshold``.

        Returns:
            Optional[Tuple[int, float]]: ``(driver_id, similarity)`` or ``None``
                if no driver exceeds the threshold.
        """
        effective_threshold = threshold if threshold is not None else self._threshold
        return self._repo.find_closest_driver(
            query_embedding, threshold=effective_threshold
        )

    # --------------------------------------------------------------------------
    # Digital Twin initialization
    # --------------------------------------------------------------------------

    def load_driver_digital_twin(self, driver_id: int) -> Dict[str, Any]:
        """Loads all data required to initialize a driver's Digital Twin session.

        Retrieves the ``DriverProfile``, active ``FaceEnrollment``, and stored
        ``DriverEmbedding`` and packages them into a single dictionary for
        consumption by the Digital Twin initialization pipeline.

        Args:
            driver_id: Primary key of the ``DriverProfile``.

        Returns:
            Dict[str, Any]: Digital Twin initialization payload with keys:
                - ``driver_profile`` (Dict) — full DriverProfile.to_dict()
                - ``enrollment`` (Optional[Dict]) — active FaceEnrollment.to_dict()
                - ``embedding_dimension`` (int)
                - ``personalized_threshold`` (float) — per-driver verification threshold
                    (currently derived from enrollment quality; future: ML-adapted)
                - ``model_info`` (Dict) — MobileFaceNet operational status

        Raises:
            ValueError: If no ``DriverProfile`` with ``driver_id`` exists.
        """
        driver = self._db.get(DriverProfile, driver_id)
        if driver is None:
            raise ValueError(
                f"FaceAuthService.load_driver_digital_twin: DriverProfile {driver_id} not found."
            )

        enrollment = self._repo.get_active_enrollment(driver_id)
        enrollment_dict = enrollment.to_dict() if enrollment else None

        # Derive personalized threshold: higher quality → lower threshold (tighter match)
        # Base: 0.65. Range: [0.55, 0.75] scaled by quality_score.
        if enrollment is not None:
            q = enrollment.quality_score
            personalized_threshold = round(
                0.75 - (q * 0.20),  # 0.75 at quality=0.0, 0.55 at quality=1.0
                4,
            )
            personalized_threshold = float(np.clip(personalized_threshold, 0.50, 0.80))
        else:
            personalized_threshold = self._threshold

        logger.info(
            "FaceAuthService.load_driver_digital_twin: Loaded Digital Twin data for "
            "driver_id=%d (personalized_threshold=%.3f).",
            driver_id,
            personalized_threshold,
        )

        return {
            "driver_profile": driver.to_dict(),
            "enrollment": enrollment_dict,
            "embedding_dimension": self._model.get_embedding_dimension(),
            "personalized_threshold": personalized_threshold,
            "model_info": self._model.get_model_info(),
        }

    # --------------------------------------------------------------------------
    # Static helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def decode_b64_to_array(b64_string: str) -> np.ndarray:
        """Decodes a base64-encoded image string to a NumPy RGB array.

        Handles optional ``data:image/<type>;base64,`` URI prefixes.
        Decodes the image using OpenCV (preferred) or NumPy/PIL fallback.

        Args:
            b64_string: Base64-encoded JPEG or PNG image string.

        Returns:
            np.ndarray: RGB image array of shape (H, W, 3), dtype uint8.

        Raises:
            ValueError: If the string cannot be decoded or the image is invalid.
        """
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]

        try:
            raw_bytes = base64.b64decode(b64_string)
        except Exception as exc:
            raise ValueError(f"Invalid base64 image string: {exc}") from exc

        if _CV2_AVAILABLE:
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(
                    "FaceAuthService.decode_b64_to_array: cv2.imdecode returned None. "
                    "Ensure the image is a valid JPEG or PNG."
                )
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # PIL fallback
        try:
            import io
            from PIL import Image as PILImage  # type: ignore[import]
            pil_img = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")
            return np.array(pil_img, dtype=np.uint8)
        except ImportError:
            raise RuntimeError(
                "FaceAuthService.decode_b64_to_array: Neither cv2 nor PIL is available. "
                "Install opencv-python or Pillow."
            )
        except Exception as exc:
            raise ValueError(
                f"FaceAuthService.decode_b64_to_array: Failed to decode image: {exc}"
            ) from exc

    # --------------------------------------------------------------------------
    # Private helpers
    # --------------------------------------------------------------------------

    def _load_driver_profile_dict(self, driver_id: int) -> Optional[Dict[str, Any]]:
        """Loads a DriverProfile and returns its dictionary representation.

        Args:
            driver_id: Primary key of the DriverProfile.

        Returns:
            Optional[Dict[str, Any]]: DriverProfile.to_dict() or None if not found.
        """
        driver = self._db.get(DriverProfile, driver_id)
        if driver is None:
            return None
        return driver.to_dict()
