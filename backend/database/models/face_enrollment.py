"""FaceEnrollment Database Model — MobileFaceNet Enrollment Records."""

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

from backend.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.database.models.driver_profile import DriverProfile
    from backend.database.models.embeddings import DriverEmbedding


class FaceEnrollment(Base, TimestampMixin):
    """Persistent record of a single facial enrollment event for a driver."""

    __tablename__ = "face_enrollments"

    # --------------------------------------------------------------------------
    # Columns
    # --------------------------------------------------------------------------

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # unique=True is kept; index=True removed — declared in __table_args__
    uuid: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        default=lambda: str(uuid_pkg.uuid4()),
        doc="UUID4 string for external resource identification",
    )

    # index=True removed — declared in __table_args__
    driver_id: Mapped[int] = mapped_column(
        ForeignKey(
            "driver_profiles.id",
            name="fk_face_enrollments_driver_id",
            ondelete="CASCADE",
        ),
        nullable=False,
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
    )

    samples_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("1"),
    )

    last_verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
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
    # Composite indexes — single authoritative source
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
        """Records a successful live authentication against this enrollment."""
        self.last_verified_at = datetime.now(timezone.utc)

    def deactivate(self) -> None:
        """Marks this enrollment as inactive."""
        self.is_active = False

    def update_quality(self, quality_score: float, samples_count: int) -> None:
        """Updates the calibration quality metrics."""
        if not (0.0 <= quality_score <= 1.0):
            raise ValueError(f"quality_score must be in [0.0, 1.0]. Got: {quality_score}")
        if samples_count < 0:
            raise ValueError(f"samples_count must be non-negative. Got: {samples_count}")
        self.quality_score = quality_score
        self.samples_count = samples_count

    def to_dict(self) -> Dict[str, Any]:
        """Serialises the enrollment record to a JSON-safe dictionary."""
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
