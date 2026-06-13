from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Fallback imports to support different run paths
try:
    from backend.database.models.driver_profile import DrivingStyle, StressSensitivity, RiskTolerance
except ImportError:
    from database.models.driver_profile import DrivingStyle, StressSensitivity, RiskTolerance


class DriverBase(BaseModel):
    """Base schema containing shared properties for a driver profile."""

    name: str = Field(..., min_length=1, max_length=100, description="Driver's full name")
    age: Optional[int] = Field(default=None, description="Driver's age")
    experience_years: int = Field(default=0, description="Years of driving experience")
    preferred_driving_time: Optional[str] = Field(default=None, max_length=30, description="Preferred driving period")
    driving_style: DrivingStyle = Field(default=DrivingStyle.UNKNOWN, description="Long-term driving style")
    stress_sensitivity: StressSensitivity = Field(default=StressSensitivity.UNKNOWN, description="Sensitivity level to stress")
    reaction_time: Optional[float] = Field(default=None, description="Reaction time in milliseconds")
    fatigue_tolerance: Optional[float] = Field(default=None, description="Fatigue tolerance in minutes")
    risk_tolerance: RiskTolerance = Field(default=RiskTolerance.UNKNOWN, description="Driver's level of risk tolerance")

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: Optional[int]) -> Optional[int]:
        """Validates that age is positive."""
        if v is not None and v < 0:
            raise ValueError("Age must be a non-negative integer.")
        return v

    @field_validator("experience_years")
    @classmethod
    def validate_experience_years(cls, v: int) -> int:
        """Validates that experience years are positive."""
        if v < 0:
            raise ValueError("Experience years must be a non-negative integer.")
        return v

    @field_validator("reaction_time")
    @classmethod
    def validate_reaction_time(cls, v: Optional[float]) -> Optional[float]:
        """Validates that reaction time is positive."""
        if v is not None and v < 0.0:
            raise ValueError("Reaction time must be a non-negative number.")
        return v

    @field_validator("fatigue_tolerance")
    @classmethod
    def validate_fatigue_tolerance(cls, v: Optional[float]) -> Optional[float]:
        """Validates that fatigue tolerance is positive."""
        if v is not None and v < 0.0:
            raise ValueError("Fatigue tolerance must be a non-negative number.")
        return v


class DriverCreate(DriverBase):
    """Schema for validating driver profile creation payloads."""
    pass


class DriverUpdate(BaseModel):
    """Schema for validating driver profile update payloads.

    All fields are optional to support partial updates.
    """

    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    age: Optional[int] = Field(default=None)
    experience_years: Optional[int] = Field(default=None)
    preferred_driving_time: Optional[str] = Field(default=None, max_length=30)
    driving_style: Optional[DrivingStyle] = Field(default=None)
    stress_sensitivity: Optional[StressSensitivity] = Field(default=None)
    reaction_time: Optional[float] = Field(default=None)
    fatigue_tolerance: Optional[float] = Field(default=None)
    risk_tolerance: Optional[RiskTolerance] = Field(default=None)
    aggression_tendency: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("age", "experience_years", "reaction_time", "fatigue_tolerance")
    @classmethod
    def validate_positive_fields(cls, v: Any) -> Any:
        """Ensures updated values remain positive."""
        if v is not None and v < 0:
            raise ValueError("Updated field value must be non-negative.")
        return v


class DriverStatistics(BaseModel):
    """Accumulated performance counts for a driver's career history."""

    total_sessions: int = Field(default=0, ge=0)
    total_drive_minutes: float = Field(default=0.0, ge=0.0)
    fatigue_events: int = Field(default=0, ge=0)
    distraction_events: int = Field(default=0, ge=0)
    aggression_events: int = Field(default=0, ge=0)

    model_config = ConfigDict(from_attributes=True)


class DriverAverages(BaseModel):
    """Biometric and behavioral baseline metrics for the driver's profile."""

    avg_blink_rate: Optional[float] = Field(default=None, ge=0.0)
    avg_ear: Optional[float] = Field(default=None, ge=0.0)
    avg_mar: Optional[float] = Field(default=None, ge=0.0)
    avg_attention_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    avg_cli: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    avg_risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    avg_stress_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)

    model_config = ConfigDict(from_attributes=True)


class DriverResponse(DriverBase):
    """Schema returned by APIs representing a driver's details and Digital Twin statistics."""

    id: int
    uuid: str
    aggression_tendency: float
    embedding_id: Optional[int]
    created_at: datetime
    updated_at: datetime

    # Nested structures constructed by validator
    totals: DriverStatistics
    averages: DriverAverages
    risk_factor: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="before")
    @classmethod
    def construct_nested_statistics(cls, data: Any) -> Any:
        """Transforms a flat database model or dict into nested statistics response blocks."""
        if isinstance(data, dict):
            # Resolve dictionary structures
            if "totals" not in data:
                data["totals"] = {
                    "total_sessions": data.get("total_sessions", 0),
                    "total_drive_minutes": data.get("total_drive_minutes", 0.0),
                    "fatigue_events": data.get("fatigue_events", 0),
                    "distraction_events": data.get("distraction_events", 0),
                    "aggression_events": data.get("aggression_events", 0),
                }
            if "averages" not in data:
                data["averages"] = {
                    "avg_blink_rate": data.get("avg_blink_rate"),
                    "avg_ear": data.get("avg_ear"),
                    "avg_mar": data.get("avg_mar"),
                    "avg_attention_score": data.get("avg_attention_score"),
                    "avg_cli": data.get("avg_cli"),
                    "avg_risk_score": data.get("avg_risk_score"),
                    "avg_stress_score": data.get("avg_stress_score"),
                }
            if "risk_factor" not in data:
                # Fallback mock calculation for dict inputs
                data["risk_factor"] = 0.0
        else:
            # Resolve SQLAlchemy database instances
            if not hasattr(data, "totals"):
                data.totals = DriverStatistics.model_validate(data)
            if not hasattr(data, "averages"):
                data.averages = DriverAverages.model_validate(data)
            if not hasattr(data, "risk_factor"):
                # Call driver model's risk factor logic if present
                if hasattr(data, "calculate_driver_risk_factor"):
                    data.risk_factor = data.calculate_driver_risk_factor()
                else:
                    data.risk_factor = 0.0

        return data
