from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import uuid

from sqlalchemy import CheckConstraint, DateTime, Enum as SAEnum, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base, TimestampMixin, UUIDMixin, SoftDeleteMixin

if TYPE_CHECKING:
    from backend.database.models.driver_profile import DriverProfile
    from backend.database.models.driving_metrics import DrivingMetric


class SessionStatus(str, PyEnum):
    """Execution status of a driving trip session."""
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"


class SessionData(Base, TimestampMixin, UUIDMixin, SoftDeleteMixin):
    """Represents a single complete driving session (trip)."""

    __tablename__ = "driver_sessions"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Identifiers — unique=True is kept; index=True removed (declared in __table_args__)
    session_uuid: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
    )
    # index=True removed — declared in __table_args__
    driver_id: Mapped[int] = mapped_column(
        ForeignKey("driver_profiles.id", ondelete="CASCADE", name="fk_driver_sessions_driver_id"),
        nullable=False,
    )

    # Session Times & Metrics
    session_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    session_end: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    duration_minutes: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0.0",
    )
    distance_km: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0.0",
    )

    # Driving Statistics
    avg_speed: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0.0")
    max_speed: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0.0")
    avg_attention_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0.0")
    avg_cli: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0.0")
    avg_risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0.0")
    avg_stress_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0.0")

    # Event Counters
    fatigue_events: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    distraction_events: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    aggression_events: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    total_blinks: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    total_yawns: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    sudden_braking_events: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    lane_departure_events: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")

    # Status — index=True removed; declared in __table_args__
    status: Mapped[SessionStatus] = mapped_column(
        SAEnum(SessionStatus),
        nullable=False,
        default=SessionStatus.ACTIVE,
    )

    # Relationships
    driver: Mapped["DriverProfile"] = relationship(
        "DriverProfile",
        back_populates="sessions",
        foreign_keys=[driver_id],
    )

    metrics: Mapped[List["DrivingMetric"]] = relationship(
        "DrivingMetric",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select",
    )

    # Table constraints and indexes — single authoritative source
    __table_args__ = (
        Index("ix_driver_sessions_driver_id", "driver_id"),
        Index("ix_driver_sessions_session_start", "session_start"),
        Index("ix_driver_sessions_status", "status"),
        Index("ix_driver_sessions_driver_start", "driver_id", "session_start"),
        Index("ix_driver_sessions_driver_status", "driver_id", "status"),
        CheckConstraint("duration_minutes >= 0", name="chk_session_duration"),
        CheckConstraint("distance_km >= 0", name="chk_session_distance"),
        CheckConstraint("avg_speed >= 0", name="chk_session_avg_speed"),
        CheckConstraint("max_speed >= 0", name="chk_session_max_speed"),
    )

    def start_session(self) -> None:
        """Initializes values for beginning an active session."""
        self.session_start = datetime.now(timezone.utc)
        self.status = SessionStatus.ACTIVE

    def close_session(self) -> None:
        """Finalizes the active session statistics and records end timestamp."""
        self.session_end = datetime.now(timezone.utc)
        self.status = SessionStatus.COMPLETED
        self.calculate_duration()

    def pause_session(self) -> None:
        """Temporarily pauses tracking updates."""
        self.status = SessionStatus.PAUSED
        self.calculate_duration()

    def resume_session(self) -> None:
        """Resumes active session status tracking."""
        self.status = SessionStatus.ACTIVE

    def calculate_duration(self) -> float:
        """Computes the active duration of the trip in minutes."""
        end = self.session_end or datetime.now(timezone.utc)
        delta = end - self.session_start
        self.duration_minutes = max(0.0, delta.total_seconds() / 60.0)
        return self.duration_minutes

    def calculate_average_speed(self, current_total_distance: float) -> float:
        """Calculates average speed based on distance and duration."""
        self.distance_km = max(0.0, current_total_distance)
        duration = self.calculate_duration()
        if duration > 0:
            self.avg_speed = (self.distance_km / duration) * 60.0
        else:
            self.avg_speed = 0.0
        return self.avg_speed

    def calculate_event_rate(self, event_count: int) -> float:
        """Computes event frequency per minute based on current duration."""
        duration = self.duration_minutes or self.calculate_duration()
        if duration > 0:
            return event_count / duration
        return 0.0

    def calculate_session_risk(self) -> float:
        """Computes overall trip session risk level."""
        duration = self.duration_minutes or self.calculate_duration()
        if duration > 0.0:
            distraction_frequency = min((self.distraction_events / duration) / 0.33, 1.0)
            fatigue_frequency = min((self.fatigue_events / duration) / 0.33, 1.0)
        else:
            distraction_frequency = 0.0
            fatigue_frequency = 0.0

        cli = (self.avg_cli / 100.0) if self.avg_cli > 1.0 else self.avg_cli
        stress = (self.avg_stress_score / 100.0) if self.avg_stress_score > 1.0 else self.avg_stress_score
        risk_score = (self.avg_risk_score / 100.0) if self.avg_risk_score > 1.0 else self.avg_risk_score

        cli = max(0.0, min(1.0, cli))
        stress = max(0.0, min(1.0, stress))
        risk_score = max(0.0, min(1.0, risk_score))

        risk = (
            0.30 * cli
            + 0.25 * stress
            + 0.25 * risk_score
            + 0.15 * distraction_frequency
            + 0.10 * fatigue_frequency
        )
        return max(0.0, min(1.0, risk))

    def increment_fatigue_event(self) -> None:
        self.fatigue_events += 1

    def increment_distraction_event(self) -> None:
        self.distraction_events += 1

    def increment_aggression_event(self) -> None:
        self.aggression_events += 1

    def increment_lane_departure(self) -> None:
        self.lane_departure_events += 1

    def increment_sudden_braking(self) -> None:
        self.sudden_braking_events += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serializes session summary data to a JSON compatible dictionary."""
        return {
            "id": self.id,
            "session_uuid": self.session_uuid,
            "driver_id": self.driver_id,
            "session_start": self.session_start.isoformat() if self.session_start else None,
            "session_end": self.session_end.isoformat() if self.session_end else None,
            "duration_minutes": self.duration_minutes,
            "distance_km": self.distance_km,
            "avg_speed": self.avg_speed,
            "max_speed": self.max_speed,
            "avg_attention_score": self.avg_attention_score,
            "avg_cli": self.avg_cli,
            "avg_risk_score": self.avg_risk_score,
            "avg_stress_score": self.avg_stress_score,
            "fatigue_events": self.fatigue_events,
            "distraction_events": self.distraction_events,
            "aggression_events": self.aggression_events,
            "total_blinks": self.total_blinks,
            "total_yawns": self.total_yawns,
            "sudden_braking_events": self.sudden_braking_events,
            "lane_departure_events": self.lane_departure_events,
            "status": self.status.value if self.status else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
