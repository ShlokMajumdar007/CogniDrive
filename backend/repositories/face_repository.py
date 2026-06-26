"""FaceRepository — Data Access Layer for Face Enrollment and Embedding Operations.

Provides a clean, typed repository interface over the ``FaceEnrollment`` and
``DriverEmbedding`` database tables.  All database operations are encapsulated
here so that the service layer (``FaceAuthService``) has zero direct SQLAlchemy
query logic.

Design:
    - SQLAlchemy 2.0 session-based queries.
    - All methods are synchronous (SQLite, single-node deployment).
    - Explicit transaction management: callers are responsible for ``db.commit()``
      after mutating operations, which keeps transaction boundaries clean.
    - ``find_closest_driver()`` performs an in-Python cosine similarity scan over
      all active embeddings.  This is acceptable for a fleet of ≤1000 drivers on
      edge hardware; for larger corpora a vector index (FAISS / pgvector) should
      be substituted.

Usage::

    repo = FaceRepository(db)
    enrollment = repo.create_enrollment(driver_id=3, embedding_vector=[...], ...)
    db.commit()

    active = repo.get_active_enrollment(driver_id=3)
    match = repo.find_closest_driver(live_embedding, threshold=0.65)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

# Fallback imports
try:
    from backend.database.models.face_enrollment import FaceEnrollment
    from backend.database.models.embeddings import DriverEmbedding
    from backend.database.models.driver_profile import DriverProfile
except ImportError:
    from database.models.face_enrollment import FaceEnrollment
    from database.models.embeddings import DriverEmbedding
    from database.models.driver_profile import DriverProfile

logger = logging.getLogger("CogniDrive.FaceRepository")


class FaceRepository:
    """Data access layer for facial enrollment and embedding persistence.

    All mutating operations (create / update / delete) are performed within
    the caller-managed SQLAlchemy session.  The caller is responsible for
    committing or rolling back the transaction.

    Attributes:
        _db: The active SQLAlchemy ``Session`` instance.
    """

    def __init__(self, db: Session) -> None:
        """Initialises the repository with a database session.

        Args:
            db: An active SQLAlchemy ``Session``.  Typically injected by the
                FastAPI ``get_db`` dependency.
        """
        self._db: Session = db

    # --------------------------------------------------------------------------
    # Enrollment CRUD
    # --------------------------------------------------------------------------

    def create_enrollment(
        self,
        driver_id: int,
        embedding_vector: List[float],
        samples_count: int = 1,
        quality_score: float = 0.0,
    ) -> FaceEnrollment:
        """Creates a new face enrollment record and its associated DriverEmbedding.

        Deactivates any existing active enrollment for the same driver before
        persisting the new one.  A new ``DriverEmbedding`` row is created and
        linked to the ``FaceEnrollment``.

        Args:
            driver_id: Primary key of the ``DriverProfile`` being enrolled.
            embedding_vector: L2-normalized float list produced by MobileFaceNet.
            samples_count: Number of face crops averaged to produce the embedding.
            quality_score: Calibration quality estimate in [0.0, 1.0].

        Returns:
            FaceEnrollment: The newly created and flushed enrollment record.
                **The caller must call ``db.commit()`` to persist the transaction.**

        Raises:
            ValueError: If ``driver_id`` does not exist in ``driver_profiles``.
        """
        # Verify the driver exists
        driver = self._db.get(DriverProfile, driver_id)
        if driver is None:
            raise ValueError(
                f"FaceRepository.create_enrollment: DriverProfile with id={driver_id} not found."
            )

        # Deactivate any existing active enrollment
        existing = self.get_active_enrollment(driver_id)
        if existing is not None:
            logger.info(
                "FaceRepository: Deactivating previous enrollment id=%d for driver_id=%d.",
                existing.id,
                driver_id,
            )
            existing.deactivate()

        # Create the embedding record
        emb = DriverEmbedding(driver_id=driver_id)
        emb.from_numpy(np.array(embedding_vector, dtype=np.float32))
        emb.embedding_quality_score = max(0.0, min(1.0, quality_score))
        emb.calibration_samples = samples_count
        emb.last_updated = datetime.now(timezone.utc)
        self._db.add(emb)
        self._db.flush()  # Obtain emb.id before linking

        # Create the enrollment record
        enrollment = FaceEnrollment(
            driver_id=driver_id,
            embedding_id=emb.id,
            quality_score=quality_score,
            samples_count=samples_count,
            is_active=True,
        )
        self._db.add(enrollment)
        self._db.flush()  # Obtain enrollment.id

        logger.info(
            "FaceRepository: Created enrollment id=%d for driver_id=%d "
            "(samples=%d, quality=%.3f).",
            enrollment.id,
            driver_id,
            samples_count,
            quality_score,
        )
        return enrollment

    def get_driver_enrollments(
        self,
        driver_id: int,
        active_only: bool = False,
    ) -> List[FaceEnrollment]:
        """Retrieves all enrollment records for a driver, optionally filtered to active only.

        Args:
            driver_id: Primary key of the ``DriverProfile``.
            active_only: If True, returns only records where ``is_active=True``.
                Defaults to False (returns all historical records).

        Returns:
            List[FaceEnrollment]: Ordered by ``created_at`` descending (most recent first).
        """
        query = (
            self._db.query(FaceEnrollment)
            .filter(FaceEnrollment.driver_id == driver_id)
        )
        if active_only:
            query = query.filter(FaceEnrollment.is_active.is_(True))

        results = query.order_by(FaceEnrollment.created_at.desc()).all()
        logger.debug(
            "FaceRepository.get_driver_enrollments: driver_id=%d, active_only=%s → %d record(s).",
            driver_id,
            active_only,
            len(results),
        )
        return results

    def get_active_enrollment(self, driver_id: int) -> Optional[FaceEnrollment]:
        """Retrieves the single active enrollment for a driver.

        Args:
            driver_id: Primary key of the ``DriverProfile``.

        Returns:
            Optional[FaceEnrollment]: The active ``FaceEnrollment`` record,
                or ``None`` if the driver has not been enrolled.
        """
        enrollment = (
            self._db.query(FaceEnrollment)
            .filter(
                FaceEnrollment.driver_id == driver_id,
                FaceEnrollment.is_active.is_(True),
            )
            .first()
        )
        return enrollment

    def delete_enrollment(self, enrollment_id: int) -> bool:
        """Permanently deletes a face enrollment and its linked embedding.

        Args:
            enrollment_id: Primary key of the ``FaceEnrollment`` to delete.

        Returns:
            bool: True if the record was found and deleted, False if not found.
        """
        enrollment = self._db.get(FaceEnrollment, enrollment_id)
        if enrollment is None:
            logger.warning(
                "FaceRepository.delete_enrollment: enrollment_id=%d not found.",
                enrollment_id,
            )
            return False

        # Delete linked embedding if present
        if enrollment.embedding_id is not None:
            emb = self._db.get(DriverEmbedding, enrollment.embedding_id)
            if emb is not None:
                self._db.delete(emb)

        self._db.delete(enrollment)
        logger.info(
            "FaceRepository.delete_enrollment: Deleted enrollment id=%d for driver_id=%d.",
            enrollment_id,
            enrollment.driver_id,
        )
        return True

    # --------------------------------------------------------------------------
    # Embedding operations
    # --------------------------------------------------------------------------

    def update_embedding(
        self,
        driver_id: int,
        new_vector: np.ndarray,
        quality_score: float,
        samples_count: int,
    ) -> Optional[DriverEmbedding]:
        """Replaces or incrementally updates the embedding for a driver's active enrollment.

        Performs an **incremental weighted average** update (not a full replace)::

            updated = (old_vector * old_n + new_vector) / (old_n + 1)

        The result is then L2-normalized before being persisted.

        Args:
            driver_id: Primary key of the ``DriverProfile``.
            new_vector: New L2-normalized float32 embedding from MobileFaceNet.
            quality_score: Updated quality estimate in [0.0, 1.0].
            samples_count: Total number of samples (including this update).

        Returns:
            Optional[DriverEmbedding]: The updated ``DriverEmbedding``, or ``None``
                if no active enrollment exists for the driver.
        """
        enrollment = self.get_active_enrollment(driver_id)
        if enrollment is None:
            logger.warning(
                "FaceRepository.update_embedding: No active enrollment for driver_id=%d. "
                "Call create_enrollment first.",
                driver_id,
            )
            return None

        emb = (
            self._db.get(DriverEmbedding, enrollment.embedding_id)
            if enrollment.embedding_id
            else None
        )

        if emb is None:
            # First time: create a fresh embedding record
            logger.info(
                "FaceRepository.update_embedding: No existing embedding — creating fresh record."
            )
            emb = DriverEmbedding(driver_id=driver_id)
            emb.from_numpy(new_vector)
            emb.embedding_quality_score = quality_score
            emb.calibration_samples = samples_count
            self._db.add(emb)
            self._db.flush()
            enrollment.embedding_id = emb.id
        else:
            old_vector = emb.to_numpy()
            old_n = max(emb.calibration_samples, 1)

            # Weighted incremental average
            updated_vector = (old_vector * old_n + new_vector) / (old_n + 1)

            # Re-normalize to unit sphere
            norm = float(np.linalg.norm(updated_vector))
            if norm > 1e-8:
                updated_vector = (updated_vector / norm).astype(np.float32)

            emb.update_embedding(updated_vector, quality_score)
            emb.calibration_samples = samples_count

        # Sync quality on enrollment record as well
        enrollment.update_quality(quality_score, samples_count)

        logger.info(
            "FaceRepository.update_embedding: Updated embedding for driver_id=%d "
            "(samples=%d, quality=%.3f).",
            driver_id,
            samples_count,
            quality_score,
        )
        return emb

    def get_embedding_by_driver(self, driver_id: int) -> Optional[DriverEmbedding]:
        """Retrieves the ``DriverEmbedding`` linked to a driver's active enrollment.

        Args:
            driver_id: Primary key of the ``DriverProfile``.

        Returns:
            Optional[DriverEmbedding]: The linked embedding record, or ``None``
                if the driver has no active enrollment with an embedding.
        """
        enrollment = self.get_active_enrollment(driver_id)
        if enrollment is None or enrollment.embedding_id is None:
            return None
        return self._db.get(DriverEmbedding, enrollment.embedding_id)

    # --------------------------------------------------------------------------
    # Identification: nearest-neighbor search
    # --------------------------------------------------------------------------

    def find_closest_driver(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.65,
    ) -> Optional[Tuple[int, float]]:
        """Identifies the best-matching driver by scanning all active embeddings.

        Performs a linear scan of every active ``DriverEmbedding``, computing
        cosine similarity against ``query_embedding``.  Returns the driver with
        the highest similarity that also exceeds ``threshold``.

        Complexity: O(N × D) where N = number of enrolled drivers and D = embedding
        dimension.  For a single-vehicle edge deployment (N ≤ ~500) this is
        well within real-time constraints.

        Args:
            query_embedding: L2-normalized face embedding from the live camera frame.
            threshold: Minimum cosine similarity for a positive match.
                Defaults to 0.65.

        Returns:
            Optional[Tuple[int, float]]: A tuple of ``(driver_id, similarity)``
                for the best match, or ``None`` if no driver exceeds the threshold.
        """
        # Fetch all active enrollments with an embedding
        active_enrollments = (
            self._db.query(FaceEnrollment)
            .filter(
                FaceEnrollment.is_active.is_(True),
                FaceEnrollment.embedding_id.isnot(None),
            )
            .all()
        )

        if not active_enrollments:
            logger.info(
                "FaceRepository.find_closest_driver: No active enrollments in database."
            )
            return None

        best_driver_id: Optional[int] = None
        best_similarity: float = -1.0

        q = np.asarray(query_embedding, dtype=np.float32).ravel()
        q_norm = float(np.linalg.norm(q))

        if q_norm < 1e-8:
            logger.warning(
                "FaceRepository.find_closest_driver: Query embedding has near-zero norm."
            )
            return None

        for enrollment in active_enrollments:
            emb_record = self._db.get(DriverEmbedding, enrollment.embedding_id)
            if emb_record is None:
                continue

            stored = emb_record.to_numpy().ravel()
            s_norm = float(np.linalg.norm(stored))
            if s_norm < 1e-8:
                continue

            similarity = float(np.clip(np.dot(q, stored) / (q_norm * s_norm), -1.0, 1.0))

            if similarity > best_similarity:
                best_similarity = similarity
                best_driver_id = enrollment.driver_id

        if best_driver_id is not None and best_similarity >= threshold:
            logger.info(
                "FaceRepository.find_closest_driver: Best match driver_id=%d "
                "(similarity=%.4f, threshold=%.2f) ✓",
                best_driver_id,
                best_similarity,
                threshold,
            )
            return best_driver_id, best_similarity

        logger.info(
            "FaceRepository.find_closest_driver: No match above threshold=%.2f "
            "(best=%.4f).",
            threshold,
            best_similarity,
        )
        return None

    def get_all_enrollment_stats(self) -> Dict[str, int]:
        """Returns aggregate counts for health and monitoring endpoints.

        Returns:
            Dict[str, int]: Dictionary with keys:
                - ``total_enrollments``: All enrollment rows.
                - ``active_enrollments``: Rows with ``is_active=True``.
                - ``enrolled_drivers``: Distinct driver IDs with an active enrollment.
        """
        from sqlalchemy import func

        total = self._db.query(func.count(FaceEnrollment.id)).scalar() or 0
        active = (
            self._db.query(func.count(FaceEnrollment.id))
            .filter(FaceEnrollment.is_active.is_(True))
            .scalar()
            or 0
        )
        distinct_drivers = (
            self._db.query(func.count(func.distinct(FaceEnrollment.driver_id)))
            .filter(FaceEnrollment.is_active.is_(True))
            .scalar()
            or 0
        )

        return {
            "total_enrollments": int(total),
            "active_enrollments": int(active),
            "enrolled_drivers": int(distinct_drivers),
        }
