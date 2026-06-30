from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, computed_field, ConfigDict

from backend.database.models.session_data import SessionStatus


class SessionBase(BaseModel):
    """Base schema properties shared across driving sessions."""

    driver_id: int = Field(..., description="Foreign key linking to driver profile")
    session_start: datetime = Field(..., description="Timestamp representing the start of the trip")
    distance_km: float = Field(default=0.0, description="Total distance accumulated in kilometers")
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Active status identifier of the trip")

    @field_validator("distance_km")
    @classmethod
    def validate_distance(cls, v: float) -> float:
        """Enforces non-negative distance entries."""
        if v < 0.0:
            raise ValueError("Distance must be a non-negative number.")
        return v

    @field_validator("session_start")
    @classmethod
    def validate_session_start(cls, v: datetime) -> datetime:
        """Enforces that a session cannot start in the future, providing a 5-minute tolerance window."""
        now_utc = datetime.now(timezone.utc)
        val_utc = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        if val_utc > now_utc + timedelta(minutes=5):
            raise ValueError("Session start timestamp cannot exceed the current time by more than 5 minutes.")
        return v


class SessionCreate(SessionBase):
    """Schema for validating session initiation payloads."""
    pass


class SessionUpdate(BaseModel):
    """Schema for validating session modifications during closures or pauses."""

    session_end: Optional[datetime] = Field(default=None, description="Optional timestamp for session end")
    distance_km: Optional[float] = Field(default=None, ge=0.0)
    avg_speed: Optional[float] = Field(default=None, ge=0.0)
    max_speed: Optional[float] = Field(default=None, ge=0.0)
    avg_attention_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    avg_cli: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    avg_risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    avg_stress_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)

    fatigue_events: Optional[int] = Field(default=None, ge=0)
    distraction_events: Optional[int] = Field(default=None, ge=0)
    aggression_events: Optional[int] = Field(default=None, ge=0)
    total_blinks: Optional[int] = Field(default=None, ge=0)
    total_yawns: Optional[int] = Field(default=None, ge=0)
    sudden_braking_events: Optional[int] = Field(default=None, ge=0)
    lane_departure_events: Optional[int] = Field(default=None, ge=0)

    status: Optional[SessionStatus] = Field(default=None)

    @model_validator(mode="after")
    def validate_speed_metrics(self) -> "SessionUpdate":
        """Ensures that max speed remains equal or superior to average speed if both are supplied."""
        if self.avg_speed is not None and self.max_speed is not None:
            if self.max_speed < self.avg_speed:
                raise ValueError("Maximum speed cannot be less than average speed.")
        return self


class SessionSummary(BaseModel):
    """Simplified trip data summary for quick dashboard listings."""

    session_uuid: str
    duration_minutes: float
    distance_km: float
    avg_speed: float
    avg_attention_score: float
    avg_cli: float
    avg_risk_score: float
    fatigue_events: int
    distraction_events: int
    aggression_events: int

    model_config = ConfigDict(from_attributes=True)


class SessionStatistics(BaseModel):
    """Calculated occurrence frequency rates and density properties for session analysis."""

    fatigue_rate: float = Field(..., description="Fatigue events count per minute")
    distraction_rate: float = Field(..., description="Distraction events count per minute")
    aggression_rate: float = Field(..., description="Aggressive events count per minute")
    blink_rate_per_minute: float = Field(..., description="Calculated eye blinks per minute")
    yawns_per_minute: float = Field(..., description="Calculated yawns per minute")
    risk_density: float = Field(..., description="Accident risk score weighted against trigger rates")
    average_event_rate: float = Field(..., description="Sum of all primary alert events per minute")

    model_config = ConfigDict(from_attributes=True)


class SessionResponse(BaseModel):
    """Detailed response schema returned by session query APIs."""

    id: int
    session_uuid: str
    driver_id: int
    session_start: datetime
    session_end: Optional[datetime] = None
    duration_minutes: float
    distance_km: float
    avg_speed: float
    max_speed: float
    avg_attention_score: float
    avg_cli: float
    avg_risk_score: float
    avg_stress_score: float

    fatigue_events: int
    distraction_events: int
    aggression_events: int
    total_blinks: int
    total_yawns: int
    sudden_braking_events: int
    lane_departure_events: int

    status: SessionStatus

    model_config = ConfigDict(from_attributes=True)

    @computed_field
    @property
    def session_duration(self) -> float:
        """Returns the calculated active duration minutes."""
        return self.duration_minutes

    @computed_field
    @property
    def fatigue_rate(self) -> float:
        """Calculates fatigue triggers per minute."""
        return self.fatigue_events / max(self.duration_minutes, 1.0)

    @computed_field
    @property
    def distraction_rate(self) -> float:
        """Calculates distraction triggers per minute."""
        return self.distraction_events / max(self.duration_minutes, 1.0)

    @computed_field
    @property
    def aggression_rate(self) -> float:
        """Calculates aggression triggers per minute."""
        return self.aggression_events / max(self.duration_minutes, 1.0)

    @computed_field
    @property
    def blink_rate_per_minute(self) -> float:
        """Calculates driver eye blinks per minute."""
        return self.total_blinks / max(self.duration_minutes, 1.0)

    @computed_field
    @property
    def yawns_per_minute(self) -> float:
        """Calculates driver yawn actions per minute."""
        return self.total_yawns / max(self.duration_minutes, 1.0)

    @computed_field
    @property
    def risk_density(self) -> float:
        """Calculates composite risk density weighted against driver behaviors."""
        total_rate = self.fatigue_rate + self.distraction_rate + self.aggression_rate
        return self.avg_risk_score * total_rate

    @computed_field
    @property
    def summary(self) -> SessionSummary:
        """Extracts and formats summary fields into a simplified nested block."""
        return SessionSummary(
            session_uuid=self.session_uuid,
            duration_minutes=self.duration_minutes,
            distance_km=self.distance_km,
            avg_speed=self.avg_speed,
            avg_attention_score=self.avg_attention_score,
            avg_cli=self.avg_cli,
            avg_risk_score=self.avg_risk_score,
            fatigue_events=self.fatigue_events,
            distraction_events=self.distraction_events,
            aggression_events=self.aggression_events,
        )

    @computed_field
    @property
    def statistics(self) -> SessionStatistics:
        """Assembles dynamically derived rate and density values into a stats block."""
        dur = max(self.duration_minutes, 1.0)
        f_rate = self.fatigue_events / dur
        d_rate = self.distraction_events / dur
        a_rate = self.aggression_events / dur
        b_rate = self.total_blinks / dur
        y_rate = self.total_yawns / dur
        r_density = self.avg_risk_score * (f_rate + d_rate + a_rate)
        total_events = self.fatigue_events + self.distraction_events + self.aggression_events
        avg_evt_rate = total_events / dur

        return SessionStatistics(
            fatigue_rate=f_rate,
            distraction_rate=d_rate,
            aggression_rate=a_rate,
            blink_rate_per_minute=b_rate,
            yawns_per_minute=y_rate,
            risk_density=r_density,
            average_event_rate=avg_evt_rate,
        )

    def is_active(self) -> bool:
        """Verifies if the session status maps to ACTIVE."""
        return self.status == SessionStatus.ACTIVE

    def is_completed(self) -> bool:
        """Verifies if the session status maps to COMPLETED."""
        return self.status == SessionStatus.COMPLETED

    def is_high_risk(self) -> bool:
        """Determines if the session bounds violate safe limits.

        Checks average risk scores or event frequency thresholds.
        """
        dur = max(self.duration_minutes, 1.0)
        f_rate = self.fatigue_events / dur
        d_rate = self.distraction_events / dur

        return (
            self.avg_risk_score > 0.70
            or f_rate > 0.50
            or d_rate > 0.50
        )

    def session_risk(self) -> float:
        """Evaluates overall trip risk level.

        Formula:
            risk = 0.30 * avg_cli + 0.25 * avg_stress_score + 0.25 * avg_risk_score
                 + 0.15 * distraction_frequency + 0.10 * fatigue_frequency
        """
        dur = max(self.duration_minutes, 1.0)
        # Normalize percentage metrics (0-100 to 0-1)
        cli = (self.avg_cli / 100.0) if self.avg_cli > 1.0 else self.avg_cli
        stress = (self.avg_stress_score / 100.0) if self.avg_stress_score > 1.0 else self.avg_stress_score
        risk_score = (self.avg_risk_score / 100.0) if self.avg_risk_score > 1.0 else self.avg_risk_score

        # Scale rates relative to extreme hazard frequency definitions (1 trigger per 3 minutes)
        d_freq = min((self.distraction_events / dur) / 0.33, 1.0)
        f_freq = min((self.fatigue_events / dur) / 0.33, 1.0)

        risk = 0.30 * cli + 0.25 * stress + 0.25 * risk_score + 0.15 * d_freq + 0.10 * f_freq
        return float(max(0.0, min(1.0, risk)))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the response schema to a standard Python dictionary."""
        return self.model_dump()
