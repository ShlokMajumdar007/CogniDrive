from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import Any, Dict, Optional, TYPE_CHECKING

from sqlalchemy import Boolean, CheckConstraint, DateTime, Enum as SAEnum, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base, TimestampMixin, UUIDMixin, SoftDeleteMixin

if TYPE_CHECKING:
    from backend.database.models.driver_profile import DriverProfile


class RecommendationType(str, PyEnum):
    """Categorized varieties of recommendations."""
    FATIGUE = "FATIGUE"
    STRESS = "STRESS"
    AGGRESSION = "AGGRESSIVE"
    DISTRACTION = "DISTRACTION"
    HIGH_RISK = "HIGH_RISK"
    BEHAVIORAL = "BEHAVIORAL"
    CALIBRATION = "CALIBRATION"
    ATTENTION = "ATTENTION"
    HEALTH = "HEALTH"
    GENERAL = "GENERAL"


class PriorityLevel(str, PyEnum):
    """Danger evaluation severity of alerts."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecommendationStatus(str, PyEnum):
    """Lifecycle phase states of recommendations."""
    PENDING = "PENDING"
    DISPLAYED = "DISPLAYED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    DISMISSED = "DISMISSED"
    EXPIRED = "EXPIRED"


class Recommendation(Base, TimestampMixin, UUIDMixin, SoftDeleteMixin):
    """Stores actionable guidance instructions and safety alerts generated entirely offline."""

    __tablename__ = "recommendations"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key — index=True removed; declared in __table_args__
    driver_id: Mapped[int] = mapped_column(
        ForeignKey("driver_profiles.id", ondelete="CASCADE", name="fk_recommendations_driver_id"),
        nullable=False,
    )

    # Enum details — index=True removed; declared in __table_args__
    recommendation_type: Mapped[RecommendationType] = mapped_column(
        SAEnum(RecommendationType),
        nullable=False,
    )
    priority: Mapped[PriorityLevel] = mapped_column(
        SAEnum(PriorityLevel),
        nullable=False,
    )
    status: Mapped[RecommendationStatus] = mapped_column(
        SAEnum(RecommendationStatus),
        nullable=False,
        default=RecommendationStatus.PENDING,
        server_default="PENDING",
    )

    # Core payloads
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    message: Mapped[str] = mapped_column(String(1000), nullable=False)
    explanation: Mapped[str] = mapped_column(String(2000), nullable=False)
    recommended_action: Mapped[str] = mapped_column(String(1000), nullable=False)

    # Evaluation metrics
    risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0.0")
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0.0")

    # Diagnostic trigger indicators
    trigger_metric: Mapped[str] = mapped_column(String(100), nullable=False)
    trigger_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    baseline_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Flags and lifecycles
    is_personalized: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="1")
    is_read: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="0")
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    driver: Mapped["DriverProfile"] = relationship(
        "DriverProfile",
        back_populates="recommendations",
        foreign_keys=[driver_id],
    )

    # Constraints and Indexes — single authoritative source
    __table_args__ = (
        Index("ix_recommendations_driver_id", "driver_id"),
        Index("ix_recommendations_status", "status"),
        Index("ix_recommendations_priority", "priority"),
        Index("ix_recommendations_type", "recommendation_type"),
        Index("ix_recommendations_driver_status", "driver_id", "status"),
        Index("ix_recommendations_driver_priority", "driver_id", "priority"),
        Index("ix_recommendations_driver_type", "driver_id", "recommendation_type"),
        Index("ix_recommendations_driver_created", "driver_id", "created_at"),
        CheckConstraint("risk_score >= 0.0 AND risk_score <= 1.0", name="chk_recommendation_risk"),
        CheckConstraint("confidence_score >= 0.0 AND confidence_score <= 1.0", name="chk_recommendation_confidence"),
    )

    def mark_as_displayed(self) -> None:
        if self.status == RecommendationStatus.PENDING:
            self.status = RecommendationStatus.DISPLAYED

    def mark_as_read(self) -> None:
        self.is_read = True

    def acknowledge(self) -> None:
        self.status = RecommendationStatus.ACKNOWLEDGED
        self.is_read = True
        self.acknowledged_at = datetime.now(timezone.utc)

    def dismiss(self) -> None:
        self.status = RecommendationStatus.DISMISSED
        self.is_read = True

    def expire(self) -> None:
        self.status = RecommendationStatus.EXPIRED

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        now = datetime.now(timezone.utc)
        return now > self.expires_at

    def is_critical(self) -> bool:
        return self.priority in (PriorityLevel.HIGH, PriorityLevel.CRITICAL)

    def generate_summary(self) -> str:
        severity_label = self.priority.value
        if self.trigger_value is not None and self.baseline_value is not None and self.baseline_value > 0.0:
            deviation = ((self.trigger_value - self.baseline_value) / self.baseline_value) * 100.0
            deviation_sign = "+" if deviation >= 0.0 else ""
            summary_info = f"{self.trigger_metric} changed by {deviation_sign}{deviation:.1f}%"
        else:
            summary_info = self.title
        return f"[{severity_label}] {summary_info}"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes recommendation properties to a JSON compatible dictionary."""
        return {
            "id": self.id,
            "uuid": self.uuid,
            "driver_id": self.driver_id,
            "recommendation_type": self.recommendation_type.value if self.recommendation_type else None,
            "priority": self.priority.value if self.priority else None,
            "status": self.status.value if self.status else None,
            "title": self.title,
            "message": self.message,
            "explanation": self.explanation,
            "recommended_action": self.recommended_action,
            "risk_score": self.risk_score,
            "confidence_score": self.confidence_score,
            "trigger_metric": self.trigger_metric,
            "trigger_value": self.trigger_value,
            "baseline_value": self.baseline_value,
            "is_personalized": self.is_personalized,
            "is_read": self.is_read,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
