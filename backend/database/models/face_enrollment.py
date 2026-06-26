"""FaceEnrollment Database Model — MobileFaceNet Enrollment Records.

Tracks every facial enrollment session for a driver.  A single driver may have
multiple enrollment records (e.g. captured under different lighting conditions),
but only one active record (``is_active=True``) is used during live
authentication.

Relationship to other models:

    DriverProfile (1) ──< FaceEnrollment (N)
                                │
                                └──> DriverEmbedding (1)

The active enrollment record is the authoritative source of truth for:
    - The driver's stored face embedding (via ``DriverEmbedding``).
    - The quality score of that embedding.
    - When the driver was last successfully verified.
    - How many calibration samples were used to build the embedding.

Lifecycle::

    enroll_driver()
        → FaceEnrollment(is_active=True)
        → DriverEmbedding(embedding_vector=[...])

    authenticate_driver() → success
        → enrollment.mark_verified()

    enroll_driver() again (re-enrollment)
        → old enrollment.deactivate()
        → new FaceEnrollment(is_active=True)
"""

from __future__ import annotations

import uuid as uuid_pkg
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

# Fallback imports to support both `python -m backend.xxx` and `python backend/xxx` run styles
try:
    from backend.database.base import Base, TimestampMixin
except ImportError:
    from database.base import Base, TimestampMixin

if TYPE_CHECKING:
    try:
        from backend.database.models.driver_profile import DriverProfile
        from backend.database.models.embeddings import DriverEmbedding
    except ImportError:
        from database.models.driver_profile import DriverProfile
        from database.models.embeddings import DriverEmbedding


class FaceEnrollment(Base, TimestampMixin):
    """Persistent record of a single facial enrollment event for a driver.

    Each row represents one complete enrollment run — a collection of face
    crops whose mean MobileFaceNet embedding is stored in the linked
    ``DriverEmbedding`` record.

    Only one ``FaceEnrollment`` per driver should have ``is_active=True`` at
    any time.  The service layer enforces this invariant by calling
    ``deactivate()`` on previous records before creating a new one.

    Attributes:
        id: Auto-incrementing integer primary key.
        uuid: Unique UUID4 string for external API identification.
        driver_id: Foreign key to the owning ``DriverProfile``.
        embedding_id: Foreign key to the associated ``DriverEmbedding``.
        quality_score: Calibration quality in [0.0, 1.0].
            Higher values indicate more samples, better lighting, and higher
            inter-frame embedding stability.
        samples_count: Number of face crops averaged into the stored embedding.
        is_active: True if this enrollment is the current production record.
        last_verified_at: UTC timestamp of the most recent successful
            authentication against this enrollment.
        driver: SQLAlchemy relationship to the parent ``DriverProfile``.
        embedding: SQLAlchemy relationship to the ``DriverEmbedding`` record.
    """

    __tablename__ = "face_enrollments"

    # --------------------------------------------------------------------------
    # Columns
    # --------------------------------------------------------------------------

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    uuid: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        index=True,
        default=lambda: str(uuid_pkg.uuid4()),
        doc="UUID4 string for external resource identification",
    )

    driver_id: Mapped[int] = mapped_column(
        ForeignKey(
            "driver_profiles.id",
            name="fk_face_enrollments_driver_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
        doc="Foreign key to the owning DriverProfile",
    )

    embedding_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(
            "driver_embeddings.id",
            name="fk_face_enrollments_embedding_id",
            ondelete="SET NULL",
        ),
        nullable=True,
        doc="Foreign key to the associated DriverEmbedding; NULL until embedding is committed",
    )

    quality_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0.0",
        doc="Overall calibration quality score in [0.0, 1.0]",
    )

    samples_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        doc="Number of face crop samples averaged into the stored embedding",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("1"),
        doc="True if this enrollment is the active production record for the driver",
    )

    last_verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        doc="UTC timestamp of the last successful live authentication against this enrollment",
    )

    # --------------------------------------------------------------------------
    # Relationships
    # --------------------------------------------------------------------------

    driver: Mapped["DriverProfile"] = relationship(
        "DriverProfile",
        foreign_keys=[driver_id],
        lazy="select",
    )

    embedding: Mapped[Optional["DriverEmbedding"]] = relationship(
        "DriverEmbedding",
        foreign_keys=[embedding_id],
        lazy="select",
    )

    # --------------------------------------------------------------------------
    # Composite indexes
    # --------------------------------------------------------------------------

    __table_args__ = (
        Index("ix_face_enrollments_driver_id", "driver_id"),
        Index("ix_face_enrollments_driver_id_is_active", "driver_id", "is_active"),
        Index("ix_face_enrollments_driver_id_created_at", "driver_id", "created_at"),
    )

    # --------------------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------------------

    def mark_verified(self) -> None:
        """Records a successful live authentication against this enrollment.

        Sets ``last_verified_at`` to the current UTC time.  Call this every time
        the driver successfully passes cosine similarity verification during a
        session start.

        Example::

            if result.verified:
                enrollment.mark_verified()
                db.commit()
        """
        self.last_verified_at = datetime.now(timezone.utc)

    def deactivate(self) -> None:
        """Marks this enrollment as inactive.

        Should be called on the previous active enrollment before a new
        enrollment record is created for the same driver.  Preserves the
        historical record in the database for audit purposes.

        Example::

            old_enrollment = repo.get_active_enrollment(driver_id)
            if old_enrollment:
                old_enrollment.deactivate()
            new_enrollment = FaceEnrollment(driver_id=driver_id, is_active=True)
            db.add(new_enrollment)
            db.commit()
        """
        self.is_active = False

    def update_quality(self, quality_score: float, samples_count: int) -> None:
        """Updates the calibration quality metrics.

        Args:
            quality_score: New quality score in [0.0, 1.0].
            samples_count: Total number of samples used to compute the embedding.

        Raises:
            ValueError: If quality_score is outside [0.0, 1.0] or samples_count
                is negative.
        """
        if not (0.0 <= quality_score <= 1.0):
            raise ValueError(
                f"quality_score must be in [0.0, 1.0]. Got: {quality_score}"
            )
        if samples_count < 0:
            raise ValueError(
                f"samples_count must be non-negative. Got: {samples_count}"
            )
        self.quality_score = quality_score
        self.samples_count = samples_count

    def to_dict(self) -> Dict[str, Any]:
        """Serialises the enrollment record to a JSON-safe dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation with all scalar fields.
                The ``embedding_vector`` is deliberately excluded to keep
                API response sizes manageable; use the dedicated embedding
                endpoint to retrieve the full vector.
        """
        return {
            "id": self.id,
            "uuid": self.uuid,
            "driver_id": self.driver_id,
            "embedding_id": self.embedding_id,
            "quality_score": self.quality_score,
            "samples_count": self.samples_count,
            "is_active": self.is_active,
            "last_verified_at": (
                self.last_verified_at.isoformat() if self.last_verified_at else None
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"FaceEnrollment("
            f"id={self.id}, "
            f"driver_id={self.driver_id}, "
            f"is_active={self.is_active}, "
            f"quality={self.quality_score:.2f}, "
            f"samples={self.samples_count})"
        )
