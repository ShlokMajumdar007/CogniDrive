from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from backend.database.models.recommendations import (
    RecommendationType,
    PriorityLevel,
    RecommendationStatus,
)


class RecommendationCreate(BaseModel):
    """Schema for validating recommendation alert creation payloads."""

    driver_id: int = Field(..., description="Primary key of associated driver profile")
    recommendation_type: RecommendationType = Field(..., description="Categorized alert type classification")
    priority: PriorityLevel = Field(..., description="Severity level evaluation")
    title: str = Field(..., min_length=1, max_length=200, description="Short alert header text")
    message: str = Field(..., min_length=1, max_length=1000, description="Main alert warning statement")
    explanation: Optional[str] = Field(default=None, description="Detailed explanatory text")
    recommended_action: Optional[str] = Field(default=None, max_length=1000, description="Suggested action response")

    risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Safety threat scale (0.0 to 1.0)")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Prediction confidence score")

    trigger_metric: Optional[str] = Field(default=None, description="Biometric index name that triggered the alert")
    trigger_value: Optional[float] = Field(default=None, description="Sensor value measured")
    baseline_value: Optional[float] = Field(default=None, description="Driver baseline value during calibration")
    is_personalized: bool = Field(default=True, description="True if customized for driver baseline thresholds")
    expires_at: Optional[datetime] = Field(default=None, description="Timestamp after which alert is obsolete")

    @field_validator("title")
    @classmethod
    def validate_title_length(cls, v: str) -> str:
        """Enforces title character bounds."""
        if not (1 <= len(v) <= 200):
            raise ValueError("Title length must be between 1 and 200 characters.")
        return v

    @field_validator("message")
    @classmethod
    def validate_message_length(cls, v: str) -> str:
        """Enforces message character bounds."""
        if not (1 <= len(v) <= 1000):
            raise ValueError("Message length must be between 1 and 1000 characters.")
        return v

    @field_validator("recommended_action")
    @classmethod
    def validate_action_length(cls, v: Optional[str]) -> Optional[str]:
        """Enforces recommended action character bounds."""
        if v is not None and len(v) > 1000:
            raise ValueError("Recommended action length must not exceed 1000 characters.")
        return v


class RecommendationUpdate(BaseModel):
    """Schema for validating modifications to recommendation lifecycle states."""

    status: Optional[RecommendationStatus] = Field(default=None)
    is_read: Optional[bool] = Field(default=None)
    acknowledged_at: Optional[datetime] = Field(default=None)


class RecommendationExplanation(BaseModel):
    """Explainable AI (XAI) output schema providing human-readable biometrics logic."""

    metric_name: str = Field(..., description="Target sensor metric name")
    current_value: Optional[float] = Field(..., description="Measured sensor reading during the event")
    baseline_value: Optional[float] = Field(..., description="Driver's personalized baseline reference")
    difference_percentage: Optional[float] = Field(..., description="Deviation calculation offset from baseline")
    explanation: Optional[str] = Field(..., description="Detailed diagnostic explainability narrative text")
    recommended_action: Optional[str] = Field(..., description="Guidance steps the driver should execute")


class RecommendationSummary(BaseModel):
    """Compact summary of a recommendation alert suitable for quick feeds."""

    id: int
    title: str
    priority: PriorityLevel
    recommendation_type: RecommendationType
    status: RecommendationStatus
    is_personalized: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RecommendationResponse(BaseModel):
    """Explainable AI (XAI) recommendation details returned by query endpoints."""

    id: int
    uuid: str
    driver_id: int
    recommendation_type: RecommendationType
    priority: PriorityLevel
    status: RecommendationStatus
    title: str
    message: str
    explanation: Optional[str]
    recommended_action: Optional[str]
    risk_score: float
    confidence_score: float
    trigger_metric: Optional[str]
    trigger_value: Optional[float]
    baseline_value: Optional[float]
    is_personalized: bool
    is_read: bool
    created_at: datetime
    expires_at: Optional[datetime]
    acknowledged_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)

    @computed_field
    @property
    def is_critical(self) -> bool:
        """Indicates if recommendation is highly urgent (HIGH or CRITICAL)."""
        return self.priority in (PriorityLevel.HIGH, PriorityLevel.CRITICAL)

    @computed_field
    @property
    def is_expired(self) -> bool:
        """Verifies if the recommendation has exceeded its expiration timestamp."""
        if not self.expires_at:
            return False
        now = datetime.now(timezone.utc)
        val_utc = self.expires_at if self.expires_at.tzinfo else self.expires_at.replace(tzinfo=timezone.utc)
        return now > val_utc

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

    @computed_field
    @property
    def difference_percentage(self) -> Optional[float]:
        """Computes deviation between trigger value and baseline.

        Formula:
            difference_percentage = (trigger_value - baseline_value) / max(baseline_value, 0.001) * 100
        """
        if self.trigger_value is None or self.baseline_value is None:
            return None
        val = self.trigger_value
        base = self.baseline_value
        denominator = max(base, 0.001)
        diff = ((val - base) / denominator) * 100.0
        return float(round(diff, 1))

    @computed_field
    @property
    def summary(self) -> RecommendationSummary:
        """Returns a simplified nested summary block."""
        return RecommendationSummary(
            id=self.id,
            title=self.title,
            priority=self.priority,
            recommendation_type=self.recommendation_type,
            status=self.status,
            is_personalized=self.is_personalized,
            created_at=self.created_at,
        )

    @computed_field
    @property
    def explainability(self) -> RecommendationExplanation:
        """Generates structured diagnostics data to back the XAI dashboard displays."""
        return RecommendationExplanation(
            metric_name=self.trigger_metric or "N/A",
            current_value=self.trigger_value,
            baseline_value=self.baseline_value,
            difference_percentage=self.difference_percentage,
            explanation=self.generate_explanation(),
            recommended_action=self.recommended_action or "Keep your focus on the road.",
        )

    def is_high_priority(self) -> bool:
        """Indicates if recommendation requires immediate user intervention."""
        return self.is_critical

    def is_pending(self) -> bool:
        """Indicates if the recommendation is pending action."""
        return self.status == RecommendationStatus.PENDING

    def mark_as_read(self) -> None:
        """Flags recommendation as read."""
        self.is_read = True

    def mark_as_acknowledged(self) -> None:
        """Updates status to acknowledged with UTC timestamp."""
        self.status = RecommendationStatus.ACKNOWLEDGED
        self.is_read = True
        self.acknowledged_at = datetime.now(timezone.utc)

    def generate_explanation(self) -> str:
        """Generates human-readable explanation narratives from metrics context."""
        metric = (self.trigger_metric or "").lower()
        diff = self.difference_percentage
        diff_str = f"{abs(diff):.1f}%" if diff is not None else "N/A"
        direction = "exceeded" if (diff is not None and diff >= 0) else "dropped below"

        if "fatigue" in metric or "ear" in metric:
            if diff is not None:
                return f"Fatigue probability exceeded your baseline by {diff_str}."
            return "Fatigue probability exceeded your personalized baseline."
        elif "distraction" in metric or "gaze" in metric or "yaw" in metric or "pitch" in metric:
            return "Distraction frequency is significantly higher than your historical average."
        elif "stress" in metric or "cli" in metric:
            return "Stress score has increased over your previous three sessions."
        
        # Generic explanation fallback
        if diff is not None:
            return (
                f"Sensor metric '{self.trigger_metric}' has {direction} your calibrated "
                f"baseline of {self.baseline_value or 0.0:.2f} by {diff_str}."
            )
        return f"Sensor metric '{self.trigger_metric}' deviated from your baseline average."

    def to_dict(self) -> Dict[str, Any]:
        """Converts model schema properties to a dictionary."""
        return self.model_dump()


class RecommendationBatchResponse(BaseModel):
    """Schema returning collections of recommendations alongside computed priority tallies."""

    recommendations: List[RecommendationResponse] = Field(..., description="List of recommendation responses")
    total_count: int = Field(default=0, description="Total recommendations parsed")
    critical_count: int = Field(default=0, description="Count of HIGH or CRITICAL priority items")
    pending_count: int = Field(default=0, description="Count of PENDING items")

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="before")
    @classmethod
    def compute_aggregates(cls, data: Any) -> Any:
        """Automatically tallies counts when parsing recommendation lists."""
        if isinstance(data, dict):
            recs = data.get("recommendations", [])
            data["total_count"] = len(recs)

            criticals = 0
            pendings = 0
            for r in recs:
                if isinstance(r, dict):
                    priority = r.get("priority")
                    status = r.get("status")
                else:
                    priority = getattr(r, "priority", None)
                    status = getattr(r, "status", None)

                # Check if priority is critical
                if priority in (PriorityLevel.HIGH, PriorityLevel.CRITICAL, "HIGH", "CRITICAL"):
                    criticals += 1
                if status in (RecommendationStatus.PENDING, "PENDING"):
                    pendings += 1

            data["critical_count"] = criticals
            data["pending_count"] = pendings
        return data
