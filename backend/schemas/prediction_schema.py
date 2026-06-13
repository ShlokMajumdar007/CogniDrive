from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, computed_field, field_validator, model_validator

# Fallback imports to support different run paths
try:
    from backend.database.models.driving_metrics import DriverState
except ImportError:
    from database.models.driving_metrics import DriverState


class PredictionRequest(BaseModel):
    """Schema representing an incoming request payload to evaluate real-time driver predictions."""

    driver_id: int = Field(..., description="Primary key of driver profile")
    session_id: Optional[int] = Field(default=None, description="Primary key of driving session if active")
    timestamp: datetime = Field(..., description="Timestamp of metric capture")
    feature_vector: List[float] = Field(..., description="Flat 21-dimensional coordinates list")
    model_version: str = Field(default="1.0.0", description="Cognitive prediction model version")
    personalized: bool = Field(default=True, description="True if using driver personalized thresholds")

    @field_validator("feature_vector")
    @classmethod
    def validate_features(cls, v: List[float]) -> List[float]:
        """Validates that input features array conforms to shape limits and contains finite values."""
        if not v:
            raise ValueError("Feature vector cannot be empty.")
        if len(v) != 21:
            raise ValueError(f"Feature vector length must be exactly 21. Received: {len(v)}")

        for idx, val in enumerate(v):
            if not np.isfinite(val):
                raise ValueError(
                    f"Invalid vector coordinate detected at index {idx}: {val}. "
                    "All coordinates must be finite (no NaN or Infinity allowed)."
                )
        return v

    def to_numpy(self) -> np.ndarray:
        """Converts the request feature list into a NumPy float32 array.

        Returns:
            np.ndarray: Vector array matching model configurations.
        """
        return np.array(self.feature_vector, dtype=np.float32)

    def to_feature_vector(self) -> List[float]:
        """Returns the raw float feature list.

        Returns:
            List[float]: The coordinate values list.
        """
        return self.feature_vector

    def to_dict(self) -> Dict[str, Any]:
        """Converts schema properties to a dictionary."""
        return self.model_dump()


class PredictionFeatureVectorStats(BaseModel):
    """Statistical summary values calculated over a prediction feature vector."""

    mean: float = Field(..., description="Mean value of features")
    std: float = Field(..., description="Standard deviation of features")
    minimum: float = Field(..., description="Minimum value in feature set")
    maximum: float = Field(..., description="Maximum value in feature set")
    l2_norm: float = Field(..., description="L2 norm of the feature vector")


class PredictionFeatureVector(BaseModel):
    """Flattens feature coordinates alongside computed statistical attributes."""

    features: List[float] = Field(..., description="Ordered float coordinate values")
    dimension: int = Field(default=21, description="Size of the feature set")
    statistics: PredictionFeatureVectorStats = Field(..., description="Calculated statistics of features")

    model_config = ConfigDict(from_attributes=True)


class PredictionExplanation(BaseModel):
    """Explainable AI (XAI) prediction metrics and SHAP values details."""

    top_features: List[str] = Field(..., description="Feature names with peak contributions")
    feature_importances: Dict[str, float] = Field(..., description="SHAP feature importance weightings")
    triggered_thresholds: Dict[str, float] = Field(..., description="Metrics that crossed danger limits")
    baseline_differences: Dict[str, float] = Field(..., description="Metric deviation offsets from driver baseline")
    explanation: str = Field(..., description="Readable diagnostic explainability narrative text")
    recommended_action: Optional[str] = Field(default=None, description="Suggested action response")


class PredictionResponse(BaseModel):
    """Response schema returned by the prediction engine containing driver status scores."""

    timestamp: datetime
    driver_id: int
    session_id: Optional[int] = None
    attention_score: float = Field(..., ge=0.0, le=100.0)
    cli: float = Field(..., ge=0.0, le=100.0, description="Cognitive Load Index")
    stress_score: float = Field(..., ge=0.0, le=100.0)
    fatigue_probability: float = Field(..., ge=0.0, le=1.0)
    distraction_probability: float = Field(..., ge=0.0, le=1.0)
    aggression_score: float = Field(..., ge=0.0, le=1.0)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    driver_state: DriverState
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    inference_time_ms: float
    personalized: bool
    explanation: PredictionExplanation

    model_config = ConfigDict(from_attributes=True)

    @computed_field
    @property
    def is_high_risk(self) -> bool:
        """Indicates if prediction risk score exceeds limits (>0.80)."""
        return self.risk_score > 0.80 or self.driver_state == DriverState.HIGH_RISK

    @computed_field
    @property
    def is_fatigued(self) -> bool:
        """Indicates if driver shows signs of drowsiness (>0.60)."""
        return self.fatigue_probability > 0.60 or self.driver_state == DriverState.FATIGUED

    @computed_field
    @property
    def is_distracted(self) -> bool:
        """Indicates if driver gaze is off screen targets (>0.60)."""
        return self.distraction_probability > 0.60 or self.driver_state == DriverState.DISTRACTED

    @computed_field
    @property
    def is_overloaded(self) -> bool:
        """Indicates if driver CLI exceeds comfortable bounds (>70.0)."""
        return self.cli > 70.0 or self.driver_state == DriverState.OVERLOADED

    @computed_field
    @property
    def risk_level(self) -> str:
        """Evaluates danger levels to categorise alert priority bounds.

        Rules:
            risk < 0.30         -> LOW
            0.30 <= risk < 0.60 -> MEDIUM
            0.60 <= risk < 0.80 -> HIGH
            risk >= 0.80        -> CRITICAL
        """
        score = self.risk_score
        if score < 0.30:
            return "LOW"
        elif score < 0.60:
            return "MEDIUM"
        elif score < 0.80:
            return "HIGH"
        else:
            return "CRITICAL"

    def generate_explanation(self) -> str:
        """Generates a text narrative explaining trigger causes."""
        if self.is_high_risk:
            return "Accident risk score is critically high due to combined fatigue and distraction cues."
        elif self.is_fatigued:
            return f"Fatigue probability of {self.fatigue_probability:.2f} has exceeded baseline thresholds."
        elif self.is_distracted:
            return f"Gaze distraction probability of {self.distraction_probability:.2f} is significantly higher than baseline."
        elif self.is_overloaded:
            return f"Cognitive Load Index (CLI) of {self.cli:.1f} has exceeded normal stress limits."
        return "Driver biometrics and alert indicators correspond to normal state."

    def generate_recommended_action(self) -> str:
        """Provides human-readable recommended safety actions based on classification.

        Rules:
            FATIGUED    -> "Take a 15-minute break."
            DISTRACTED  -> "Refocus on the road."
            OVERLOADED  -> "Reduce distractions and slow down."
            HIGH_RISK   -> "Stop driving immediately and rest."
        """
        state = self.driver_state
        if state == DriverState.HIGH_RISK or self.risk_score >= 0.80:
            return "Stop driving immediately and rest."
        elif state == DriverState.FATIGUED:
            return "Take a 15-minute break."
        elif state == DriverState.DISTRACTED:
            return "Refocus on the road."
        elif state == DriverState.OVERLOADED:
            return "Reduce distractions and slow down."
        return "Maintain current attentiveness and focus."

    def to_dict(self) -> Dict[str, Any]:
        """Converts response schema properties to a dictionary."""
        return self.model_dump()


class BatchPredictionRequest(BaseModel):
    """Schema representing lists of metric vectors sent for batch prediction evaluations."""

    driver_id: int = Field(..., description="Primary key of target driver profile")
    predictions: List[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of metrics prediction request objects",
    )

    @field_validator("predictions")
    @classmethod
    def validate_predictions_length(cls, v: List[PredictionRequest]) -> List[PredictionRequest]:
        """Enforces that batch processing size parameters are respected."""
        if not (1 <= len(v) <= 1000):
            raise ValueError("Predictions batch size must contain between 1 and 1000 request items.")
        return v


class BatchPredictionResponse(BaseModel):
    """Schema representing aggregates and predictions compiled from a batch query request."""

    predictions: List[PredictionResponse] = Field(..., description="Collection of prediction responses")
    total_predictions: int = Field(default=0, description="Total predictions count")
    average_risk: float = Field(default=0.0, description="Average risk score computed over batch")
    maximum_risk: float = Field(default=0.0, description="Peak risk score computed over batch")
    high_risk_count: int = Field(default=0, description="Number of items identified as high risk (>0.80)")

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="before")
    @classmethod
    def compute_aggregates(cls, data: Any) -> Any:
        """Automatically computes batch summary averages and peak metrics."""
        if isinstance(data, dict):
            recs = data.get("predictions", [])
            data["total_predictions"] = len(recs)

            risks = []
            high_risk_cnt = 0
            for r in recs:
                if isinstance(r, dict):
                    risk = r.get("risk_score", 0.0)
                else:
                    risk = getattr(r, "risk_score", 0.0)
                risks.append(risk)
                if risk >= 0.80:
                    high_risk_cnt += 1

            data["average_risk"] = float(np.mean(risks)) if risks else 0.0
            data["maximum_risk"] = float(np.max(risks)) if risks else 0.0
            data["high_risk_count"] = high_risk_cnt
        return data


class CognitiveStateResponse(BaseModel):
    """Simplified contract returned to the dashboard detailing real-time driver state warnings."""

    state: DriverState = Field(..., description="Inferred state classification of driver")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence level")
    risk_level: str = Field(..., description="Priority risk level (LOW, MEDIUM, HIGH, CRITICAL)")
    recommended_action: Optional[str] = Field(default=None, description="Suggested action response warning text")
    explanation: str = Field(..., description="Readable diagnostic explanation narrative")
