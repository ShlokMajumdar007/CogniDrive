from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np

from sqlalchemy import CheckConstraint, DateTime, Enum as SAEnum, Float, ForeignKey, Index, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from backend.database.models.session_data import SessionData


class DriverState(str, PyEnum):
    """Categorized status classification of a driver for a specific frame."""
    NORMAL = "NORMAL"
    DISTRACTED = "DISTRACTED"
    FATIGUED = "FATIGUED"
    OVERLOADED = "OVERLOADED"
    HIGH_RISK = "HIGH_RISK"


class DrivingMetric(Base, TimestampMixin, UUIDMixin):
    """Stores high-frequency time-series records representing frame-by-frame biometrics.

    Enables deep logging and diagnostic checks for continuous training and model prediction
    refinements. Designed for append-only performance at ~30 FPS.
    """

    __tablename__ = "driving_metrics"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key referencing parent session
    # index=True removed here; index is declared in __table_args__ below
    session_id: Mapped[int] = mapped_column(
        ForeignKey("driver_sessions.id", ondelete="CASCADE", name="fk_driving_metrics_session_id"),
        nullable=False,
    )

    # Frame timing
    metric_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    frame_time_ms: Mapped[float] = mapped_column(Float, nullable=False)

    # Computer Vision biometrics
    ear: Mapped[float] = mapped_column(Float, nullable=False)
    mar: Mapped[float] = mapped_column(Float, nullable=False)
    perclos: Mapped[float] = mapped_column(Float, nullable=False)
    blink_rate: Mapped[float] = mapped_column(Float, nullable=False)
    head_pitch: Mapped[float] = mapped_column(Float, nullable=False)
    head_yaw: Mapped[float] = mapped_column(Float, nullable=False)
    head_roll: Mapped[float] = mapped_column(Float, nullable=False)
    gaze_x: Mapped[float] = mapped_column(Float, nullable=False)
    gaze_y: Mapped[float] = mapped_column(Float, nullable=False)
    yawning_probability: Mapped[float] = mapped_column(Float, nullable=False)

    # Vehicle telemetry features
    speed: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    acceleration: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    steering_angle: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    brake_pressure: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    lane_offset: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    indicator_state: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Cognitive estimation features
    attention_score: Mapped[float] = mapped_column(Float, nullable=False)
    cli: Mapped[float] = mapped_column(Float, nullable=False)
    stress_score: Mapped[float] = mapped_column(Float, nullable=False)
    fatigue_probability: Mapped[float] = mapped_column(Float, nullable=False)
    distraction_probability: Mapped[float] = mapped_column(Float, nullable=False)
    aggression_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Driver State — index=True removed; declared in __table_args__
    driver_state: Mapped[DriverState] = mapped_column(
        SAEnum(DriverState),
        nullable=False,
    )

    # Relationships
    session: Mapped["SessionData"] = relationship(
        "SessionData",
        back_populates="metrics",
        foreign_keys=[session_id],
    )

    # Constraints and Indexes — single authoritative source for all indexes
    __table_args__ = (
        Index("ix_driving_metrics_session_id", "session_id"),
        Index("ix_driving_metrics_timestamp", "metric_timestamp"),
        Index("ix_driving_metrics_driver_state", "driver_state"),
        Index("ix_driving_metrics_session_timestamp", "session_id", "metric_timestamp"),
        Index("ix_driving_metrics_session_frame", "session_id", "frame_number"),
        Index("ix_driving_metrics_session_risk", "session_id", "risk_score"),
        CheckConstraint("ear >= 0", name="chk_metric_ear"),
        CheckConstraint("mar >= 0", name="chk_metric_mar"),
        CheckConstraint("perclos >= 0", name="chk_metric_perclos"),
        CheckConstraint("attention_score >= 0 AND attention_score <= 100", name="chk_metric_attention"),
        CheckConstraint("cli >= 0 AND cli <= 100", name="chk_metric_cli"),
        CheckConstraint("risk_score >= 0.0 AND risk_score <= 1.0", name="chk_metric_risk"),
        CheckConstraint("fatigue_probability >= 0.0 AND fatigue_probability <= 1.0", name="chk_metric_fatigue"),
        CheckConstraint("distraction_probability >= 0.0 AND distraction_probability <= 1.0", name="chk_metric_distraction"),
    )

    def is_high_risk(self) -> bool:
        """Determines if current metrics match unsafe state criteria."""
        return self.risk_score > 0.8 or self.driver_state == DriverState.HIGH_RISK

    def is_fatigued(self) -> bool:
        """Determines if driver matches drowsiness or sleep criteria."""
        return self.fatigue_probability > 0.6 or self.driver_state == DriverState.FATIGUED

    def is_distracted(self) -> bool:
        """Determines if eyes have wandered off screen targets."""
        return self.distraction_probability > 0.6 or self.driver_state == DriverState.DISTRACTED

    def is_overloaded(self) -> bool:
        """Determines if cognitive load metrics exceed comfortable capacity limits."""
        return self.cli > 70.0 or self.driver_state == DriverState.OVERLOADED

    def compute_alert_level(self) -> str:
        """Evaluates numerical danger levels to trigger matching auditory alerts."""
        score = self.risk_score
        if score < 0.30:
            return "LOW"
        elif score < 0.60:
            return "MEDIUM"
        elif score < 0.80:
            return "HIGH"
        else:
            return "CRITICAL"

    def to_feature_vector(self) -> List[float]:
        """Flattens primary metrics parameters into an ordered vector for ML processors."""
        return [
            self.ear,
            self.mar,
            self.perclos,
            self.blink_rate,
            self.head_pitch,
            self.head_yaw,
            self.head_roll,
            self.gaze_x,
            self.gaze_y,
            self.speed,
            self.steering_angle,
            self.attention_score,
            self.cli,
            self.stress_score,
            self.fatigue_probability,
            self.distraction_probability,
            self.aggression_score,
            self.risk_score,
        ]

    def to_numpy(self) -> np.ndarray:
        """Converts driver feature coordinates to a float32 NumPy array."""
        return np.array(self.to_feature_vector(), dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes metric records to a JSON compatible dictionary."""
        return {
            "id": self.id,
            "uuid": self.uuid,
            "session_id": self.session_id,
            "metric_timestamp": self.metric_timestamp.isoformat() if self.metric_timestamp else None,
            "frame_number": self.frame_number,
            "frame_time_ms": self.frame_time_ms,
            "ear": self.ear,
            "mar": self.mar,
            "perclos": self.perclos,
            "blink_rate": self.blink_rate,
            "head_pitch": self.head_pitch,
            "head_yaw": self.head_yaw,
            "head_roll": self.head_roll,
            "gaze_x": self.gaze_x,
            "gaze_y": self.gaze_y,
            "yawning_probability": self.yawning_probability,
            "speed": self.speed,
            "acceleration": self.acceleration,
            "steering_angle": self.steering_angle,
            "brake_pressure": self.brake_pressure,
            "lane_offset": self.lane_offset,
            "indicator_state": self.indicator_state,
            "attention_score": self.attention_score,
            "cli": self.cli,
            "stress_score": self.stress_score,
            "fatigue_probability": self.fatigue_probability,
            "distraction_probability": self.distraction_probability,
            "aggression_score": self.aggression_score,
            "risk_score": self.risk_score,
            "driver_state": self.driver_state.value if self.driver_state else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
