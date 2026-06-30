from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from backend.database.models.driver_profile import DriverProfile

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base


class DriverEmbedding(Base):
    """Stores the persistent high-dimensional Driver Digital Twin embedding vectors."""

    __tablename__ = "driver_embeddings"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Back-reference linking to the driver profile
    # unique=True is kept; index=True removed — declared in __table_args__
    driver_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("driver_profiles.id", name="fk_driver_embeddings_driver_id", ondelete="CASCADE"),
        unique=True,
        nullable=True,
    )

    # Embedding payload details
    embedding_vector: Mapped[List[float]] = mapped_column(JSON, nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False, default=32, server_default="32")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    calibration_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")

    # Metrics
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    embedding_quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0.0")

    driver: Mapped["DriverProfile"] = relationship(
        "DriverProfile",
        back_populates="embedding",
        foreign_keys="DriverProfile.embedding_id",
    )

    # Table arguments: single authoritative index declarations
    __table_args__ = (
        Index("ix_driver_embeddings_driver_id", "driver_id"),
        Index("ix_driver_embeddings_last_updated", "last_updated"),
        Index("ix_driver_embeddings_version_quality", "version", "embedding_quality_score"),
    )

    def to_numpy(self) -> np.ndarray:
        """Converts the serialized JSON embedding vector into a NumPy array."""
        return np.array(self.embedding_vector, dtype=np.float32)

    def from_numpy(self, vector: np.ndarray) -> None:
        """Encodes a NumPy array vector into the JSON-compatible list format."""
        self.embedding_vector = vector.tolist()
        self.embedding_dimension = len(self.embedding_vector)

    def update_embedding(self, vector: np.ndarray, quality_score: float) -> None:
        """Updates the stored embedding coordinates and records updating timestamp/metrics."""
        self.from_numpy(vector)
        self.embedding_quality_score = max(0.0, min(1.0, quality_score))
        self.last_updated = datetime.now(timezone.utc)

    def increment_calibration_samples(self) -> None:
        """Increments the tracked calibration samples count."""
        self.calibration_samples += 1

    def compute_quality_score(
        self,
        calibration_completeness: float,
        session_diversity: float,
        embedding_stability: float,
    ) -> float:
        """Computes the overall calibration embedding quality score."""
        quality = (
            0.40 * max(0.0, min(1.0, calibration_completeness))
            + 0.30 * max(0.0, min(1.0, session_diversity))
            + 0.30 * max(0.0, min(1.0, embedding_stability))
        )
        self.embedding_quality_score = max(0.0, min(1.0, quality))
        return self.embedding_quality_score

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the embedding record database parameters to a dictionary."""
        return {
            "id": self.id,
            "driver_id": self.driver_id,
            "embedding_dimension": self.embedding_dimension,
            "version": self.version,
            "calibration_samples": self.calibration_samples,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "embedding_quality_score": self.embedding_quality_score,
            "embedding_vector": self.embedding_vector,
        }
