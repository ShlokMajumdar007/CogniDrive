"""SQLAlchemy ORM models — import all models so Base.metadata is populated."""

from backend.database.models.driver_profile import DriverProfile
from backend.database.models.session_data import SessionData, SessionStatus
from backend.database.models.driving_metrics import DrivingMetric, DriverState
from backend.database.models.embeddings import DriverEmbedding
from backend.database.models.face_enrollment import FaceEnrollment
from backend.database.models.recommendations import (
    Recommendation,
    RecommendationStatus,
    RecommendationType,
    PriorityLevel,
)

__all__ = [
    "DriverProfile",
    "SessionData",
    "SessionStatus",
    "DrivingMetric",
    "DriverState",
    "DriverEmbedding",
    "FaceEnrollment",
    "Recommendation",
    "RecommendationStatus",
    "RecommendationType",
    "PriorityLevel",
]
