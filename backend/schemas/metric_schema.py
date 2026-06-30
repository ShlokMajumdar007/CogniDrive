from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, computed_field, ConfigDict

from backend.database.models.driving_metrics import DriverState


class MetricStatistics(BaseModel):
    """Statistical summary values calculated over a metric feature vector."""

    feature_mean: float = Field(..., description="Mean of the features")
    feature_std: float = Field(..., description="Standard deviation of the features")
    feature_min: float = Field(..., description="Minimum value in the feature set")
    feature_max: float = Field(..., description="Maximum value in the feature set")
    feature_norm: float = Field(..., description="L2 norm of the feature vector")

    model_config = ConfigDict(from_attributes=True)


class MetricFeatureVector(BaseModel):
    """Flattens feature coordinates and metadata for downstream ML model pipes."""

    features: List[float] = Field(..., description="Ordered float values of dimensions")
    dimension: int = Field(default=21, description="Size of the feature set")
    statistics: MetricStatistics = Field(..., description="Calculated statistics of the features")

    model_config = ConfigDict(from_attributes=True)


class MetricBase(BaseModel):
    """Base schema holding common variables for a single frame-level metric."""

    session_id: int = Field(..., description="Primary key of parent session")
    metric_timestamp: datetime = Field(..., description="Timestamp of when the frame metrics were recorded")
    frame_number: int = Field(..., ge=0, description="Chronological index of target frame")
    frame_time_ms: float = Field(..., ge=0.0, description="Elapsed time offset in milliseconds")

    # Vision features
    ear: float = Field(..., ge=0.0, description="Eye Aspect Ratio")
    mar: float = Field(..., ge=0.0, description="Mouth Aspect Ratio")
    perclos: float = Field(..., ge=0.0, le=1.0, description="Percentage of eye closure")
    blink_rate: float = Field(..., ge=0.0, description="Blinks per minute")
    head_pitch: float = Field(..., description="Head pitch angle in degrees")
    head_yaw: float = Field(..., description="Head yaw angle in degrees")
    head_roll: float = Field(..., description="Head roll angle in degrees")
    gaze_x: float = Field(..., description="Gaze coordinate X coordinate offset")
    gaze_y: float = Field(..., description="Gaze coordinate Y coordinate offset")
    yawning_probability: float = Field(..., ge=0.0, le=1.0, description="Probability value of yawning state")

    # Vehicle features
    speed: float = Field(default=0.0, ge=0.0, description="Current speed in km/h")
    acceleration: float = Field(default=0.0, description="Current acceleration rate in m/s^2")
    steering_angle: float = Field(default=0.0, description="Steering angle in degrees")
    brake_pressure: float = Field(default=0.0, ge=0.0, description="Braking pressure applied")
    lane_offset: float = Field(default=0.0, description="Offset from lane center in meters")
    indicator_state: int = Field(default=0, description="Vehicle indicator status (0=None, 1=Left, 2=Right)")

    # Cognitive features
    attention_score: float = Field(..., ge=0.0, le=100.0, description="Driver focus score (0-100)")
    cli: float = Field(..., ge=0.0, le=100.0, description="Cognitive Load Index (0-100)")
    stress_score: float = Field(..., ge=0.0, le=100.0, description="Estimated driver stress score (0-100)")
    fatigue_probability: float = Field(..., ge=0.0, le=1.0, description="Fatigue state probability (0-1)")
    distraction_probability: float = Field(..., ge=0.0, le=1.0, description="Distraction state probability (0-1)")
    aggression_score: float = Field(..., ge=0.0, le=1.0, description="Aggressiveness score (0-1)")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Accident risk score (0-1)")

    driver_state: DriverState = Field(..., description="Inferred state classification of the driver")

    @model_validator(mode="before")
    @classmethod
    def validate_finite_floats(cls, data: Any) -> Any:
        """Enforces that all floating-point entries are finite (no NaN or Inf)."""
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, float) and not np.isfinite(v):
                    raise ValueError(f"Value for field '{k}' must be finite. Received: {v}")
        return data

    def to_feature_vector(self) -> List[float]:
        """Flattens variables into an ordered list matching ML pipeline specs.

        Returns:
            List[float]: Ordered float values (dimension = 21).
        """
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
            self.acceleration,
            self.steering_angle,
            self.brake_pressure,
            self.lane_offset,
            self.attention_score,
            self.cli,
            self.stress_score,
            self.fatigue_probability,
            self.distraction_probability,
            self.aggression_score,
            self.risk_score,
        ]

    def to_numpy(self) -> np.ndarray:
        """Converts feature coordinates to a float32 NumPy array.

        Returns:
            np.ndarray: Vector array matching model configurations.
        """
        return np.array(self.to_feature_vector(), dtype=np.float32)

    def normalize(self) -> List[float]:
        """Applies L2 normalization on the flattened feature vector.

        Returns:
            List[float]: L2-normalized vector list coordinates.
        """
        arr = self.to_numpy()
        # Ensure finite array
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        norm = np.linalg.norm(arr)

        if norm == 0.0:
            return [0.0] * len(arr)

        return (arr / norm).tolist()

    def to_dict(self) -> Dict[str, Any]:
        """Converts model schema properties to a dictionary."""
        return self.model_dump()


class MetricCreate(MetricBase):
    """Schema for validating metric creation uploads."""
    pass


class MetricBatchCreate(BaseModel):
    """Schema for validating batch uploads of driving metrics.

    Used by vision pipelines for high-frequency bulk insertion transactions.
    """

    metrics: List[MetricCreate] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="A list containing between 1 and 1000 frame metric records",
    )


class MetricResponse(MetricBase):
    """Schema returned by APIs representing a frame metric record."""

    id: int

    model_config = ConfigDict(from_attributes=True)

    @computed_field
    @property
    def is_high_risk(self) -> bool:
        """Determines if current metrics violate risk limits (>0.80)."""
        return self.risk_score > 0.80 or self.driver_state == DriverState.HIGH_RISK

    @computed_field
    @property
    def is_fatigued(self) -> bool:
        """Determines if driver matches fatigue state criteria (>0.60)."""
        return self.fatigue_probability > 0.60 or self.driver_state == DriverState.FATIGUED

    @computed_field
    @property
    def is_distracted(self) -> bool:
        """Determines if driver eyes are off road targets (>0.60)."""
        return self.distraction_probability > 0.60 or self.driver_state == DriverState.DISTRACTED

    @computed_field
    @property
    def alert_level(self) -> str:
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
