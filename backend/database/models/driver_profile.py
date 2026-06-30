from enum import Enum as PyEnum
from typing import Any, Dict, List, Optional

from sqlalchemy import CheckConstraint, Enum as SAEnum, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base, TimestampMixin, UUIDMixin, SoftDeleteMixin


class DrivingStyle(str, PyEnum):
    """Supported styles of driving behavior."""
    DEFENSIVE = "DEFENSIVE"
    NORMAL = "NORMAL"
    AGGRESSIVE = "AGGRESSIVE"
    UNKNOWN = "UNKNOWN"


class StressSensitivity(str, PyEnum):
    """Driver's sensitivity levels under driving load."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class RiskTolerance(str, PyEnum):
    """Driver's structural levels of risk tolerance."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class DriverProfile(Base, TimestampMixin, UUIDMixin, SoftDeleteMixin):
    """Central persistence model mapping a driver's personalized Digital Twin.

    Maintains biometric thresholds, historical behavioral averages, driving characteristics,
    and reference embedding relations.
    """

    __tablename__ = "driver_profiles"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Personal Information
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    experience_years: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    preferred_driving_time: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)

    # Digital Twin Features
    driving_style: Mapped[DrivingStyle] = mapped_column(
        SAEnum(DrivingStyle),
        nullable=False,
        default=DrivingStyle.UNKNOWN,
        server_default="UNKNOWN",
        index=True,
    )
    aggression_tendency: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0.0")
    stress_sensitivity: Mapped[StressSensitivity] = mapped_column(
        SAEnum(StressSensitivity),
        nullable=False,
        default=StressSensitivity.UNKNOWN,
        server_default="UNKNOWN",
    )
    reaction_time: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Driver base reaction time in milliseconds",
    )
    fatigue_tolerance: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Driver fatigue tolerance duration in minutes",
    )
    risk_tolerance: Mapped[RiskTolerance] = mapped_column(
        SAEnum(RiskTolerance),
        nullable=False,
        default=RiskTolerance.UNKNOWN,
        server_default="UNKNOWN",
    )

    # Behavioral Baselines
    avg_blink_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_ear: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_mar: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_attention_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_cli: Mapped[Optional[float]] = mapped_column(Float, nullable=True, doc="Average Cognitive Load Index")
    avg_risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True, doc="Average Accident Risk Score")
    avg_stress_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True, doc="Average stress metric score")

    # Profile Statistics
    total_sessions: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    total_drive_minutes: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0.0")
    fatigue_events: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    distraction_events: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    aggression_events: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")

    # Embedding Relation (digital twin face/mesh baseline mapping)
    embedding_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("driver_embeddings.id", name="fk_driver_profiles_embedding_id"),
        unique=True,
        nullable=True,
    )

    # Relationships
    sessions: Mapped[List["SessionData"]] = relationship(
        "SessionData",
        back_populates="driver",
        cascade="all, delete-orphan",
    )

    embedding: Mapped[Optional["DriverEmbedding"]] = relationship(
        "DriverEmbedding",
        back_populates="driver",
        uselist=False,
        foreign_keys=[embedding_id],
    )

    recommendations: Mapped[List["Recommendation"]] = relationship(
        "Recommendation",
        back_populates="driver",
        cascade="all, delete-orphan",
    )

    # Table arguments: Composite indexes and domain check constraints
    __table_args__ = (
        Index("ix_driver_profiles_name_driving_style", "name", "driving_style"),
        CheckConstraint("age >= 0", name="chk_driver_profile_age"),
        CheckConstraint("experience_years >= 0", name="chk_driver_profile_experience"),
        CheckConstraint("reaction_time >= 0", name="chk_driver_profile_reaction_time"),
        CheckConstraint("fatigue_tolerance >= 0", name="chk_driver_profile_fatigue_tolerance"),
    )

    def update_behavior_metrics(
        self,
        avg_blink_rate: float,
        avg_ear: float,
        avg_mar: float,
        avg_attention_score: float,
        avg_cli: float,
        avg_risk_score: float,
        avg_stress_score: float,
    ) -> None:
        """Updates baseline behavior statistics dynamically based on calibration or session results."""
        self.avg_blink_rate = avg_blink_rate
        self.avg_ear = avg_ear
        self.avg_mar = avg_mar
        self.avg_attention_score = avg_attention_score
        self.avg_cli = avg_cli
        self.avg_risk_score = avg_risk_score
        self.avg_stress_score = avg_stress_score

    def increment_session_count(self, drive_minutes: float) -> None:
        """Increments driver's accumulated driving session counts and tracking duration."""
        self.total_sessions += 1
        if drive_minutes > 0.0:
            self.total_drive_minutes += drive_minutes

    def record_fatigue_event(self) -> None:
        """Records an incident where drowsiness/sleep patterns were detected."""
        self.fatigue_events += 1

    def record_distraction_event(self) -> None:
        """Records an incident where driver attentiveness wandered off road targets."""
        self.distraction_events += 1

    def record_aggression_event(self) -> None:
        """Records an incident where aggressive posture or vehicle handling tendencies arose."""
        self.aggression_events += 1

    def calculate_driver_risk_factor(self) -> float:
        """Evaluates driver risk using behavior parameters.

        Formula:
            risk = 0.35 * aggression_tendency
                 + 0.25 * normalized_stress
                 + 0.20 * avg_risk_score
                 + 0.20 * distraction_frequency

        Returns:
            float: Cumulative risk value normalized between 0.0 and 1.0.
        """
        # Normalize stress score. If it is configured 0-100, scale it to 0-1.
        raw_stress = self.avg_stress_score or 0.0
        normalized_stress = (raw_stress / 100.0) if raw_stress > 1.0 else raw_stress
        normalized_stress = max(0.0, min(1.0, normalized_stress))

        # Distraction frequency defined as distraction events per minute, normalized
        if self.total_drive_minutes > 0:
            distraction_rate = self.distraction_events / self.total_drive_minutes
        else:
            distraction_rate = 0.0
        # Normalizing distraction rate: 1 event per 5 mins (0.2 rate) as baseline high risk cap.
        distraction_frequency = min(distraction_rate / 0.2, 1.0)

        avg_risk = self.avg_risk_score or 0.0
        avg_risk = max(0.0, min(1.0, avg_risk))

        aggression = self.aggression_tendency or 0.0

        risk = (
            0.35 * aggression
            + 0.25 * normalized_stress
            + 0.20 * avg_risk
            + 0.20 * distraction_frequency
        )
        return max(0.0, min(1.0, risk))

    def reset_statistics(self) -> None:
        """Resets all metrics accumulation tallies and averages back to zero/null baselines."""
        self.avg_blink_rate = None
        self.avg_ear = None
        self.avg_mar = None
        self.avg_attention_score = None
        self.avg_cli = None
        self.avg_risk_score = None
        self.avg_stress_score = None
        self.total_sessions = 0
        self.total_drive_minutes = 0.0
        self.fatigue_events = 0
        self.distraction_events = 0
        self.aggression_events = 0

    def to_dict(self) -> Dict[str, Any]:
        """Converts model parameters to a dictionary serialization."""
        return {
            "id": self.id,
            "uuid": self.uuid,
            "name": self.name,
            "age": self.age,
            "experience_years": self.experience_years,
            "preferred_driving_time": self.preferred_driving_time,
            "driving_style": self.driving_style.value if self.driving_style else None,
            "aggression_tendency": self.aggression_tendency,
            "stress_sensitivity": self.stress_sensitivity.value if self.stress_sensitivity else None,
            "reaction_time": self.reaction_time,
            "fatigue_tolerance": self.fatigue_tolerance,
            "risk_tolerance": self.risk_tolerance.value if self.risk_tolerance else None,
            "avg_blink_rate": self.avg_blink_rate,
            "avg_ear": self.avg_ear,
            "avg_mar": self.avg_mar,
            "avg_attention_score": self.avg_attention_score,
            "avg_cli": self.avg_cli,
            "avg_risk_score": self.avg_risk_score,
            "avg_stress_score": self.avg_stress_score,
            "total_sessions": self.total_sessions,
            "total_drive_minutes": self.total_drive_minutes,
            "fatigue_events": self.fatigue_events,
            "distraction_events": self.distraction_events,
            "aggression_events": self.aggression_events,
            "embedding_id": self.embedding_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
