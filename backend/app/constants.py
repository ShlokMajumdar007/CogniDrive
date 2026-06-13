from enum import Enum
from typing import List, Tuple

# API Configuration
API_V1_PREFIX: str = "/api/v1"

# Database Table Names
class DBTableNames(str, Enum):
    DRIVER_PROFILES = "driver_profiles"
    SESSION_DATA = "session_data"
    DRIVER_EMBEDDINGS = "driver_embeddings"


# Driver Behavioral States
class DriverState(str, Enum):
    ALERT = "ALERT"
    DROWSY = "DROWSY"
    DISTRACTED = "DISTRACTED"
    COGNITIVE_OVERLOAD = "COGNITIVE_OVERLOAD"
    ANOMALOUS = "ANOMALOUS"
    UNKNOWN = "UNKNOWN"


# Alert Levels
class AlertSeverity(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# MediaPipe Face Mesh Landmark Indices
# Ref: MediaPipe Face Mesh Canonical Model Map
class FaceLandmarks:
    # Nose
    NOSE_TIP: int = 1
    NOSE_BRIDGE: int = 168
    NOSE_LEFT: int = 102
    NOSE_RIGHT: int = 331
    NOSE_BOTTOM: int = 2

    # Forehead/Chin
    CHIN: int = 152
    FOREHEAD: int = 10

    # Left Eye (indices around contour)
    LEFT_EYE_INDICES: List[int] = [33, 160, 158, 133, 153, 144]
    # Key indices for EAR calculation (1 horizontal pair, 2 vertical pairs)
    # Horizontal: p1=33, p4=133
    # Vertical 1: p2=160, p6=144
    # Vertical 2: p3=158, p5=153
    LEFT_EYE_EAR_PAIRS: Tuple[int, int, int, int, int, int] = (33, 133, 160, 144, 158, 153)

    # Right Eye (indices around contour)
    RIGHT_EYE_INDICES: List[int] = [362, 385, 387, 263, 373, 380]
    # Key indices for EAR calculation
    # Horizontal: p1=362, p4=263
    # Vertical 1: p2=385, p6=380
    # Vertical 2: p3=387, p5=373
    RIGHT_EYE_EAR_PAIRS: Tuple[int, int, int, int, int, int] = (362, 263, 385, 380, 387, 373)

    # Left Iris (center, right, top, left, bottom)
    LEFT_IRIS_INDICES: List[int] = [468, 469, 470, 471, 472]
    LEFT_IRIS_CENTER: int = 468

    # Right Iris (center, right, top, left, bottom)
    RIGHT_IRIS_INDICES: List[int] = [473, 474, 475, 476, 477]
    RIGHT_IRIS_CENTER: int = 473

    # Inner Lips Contour (for MAR calculation)
    # Horizontal: p1=78, p4=308
    # Vertical 1: p2=82, p6=87
    # Vertical 2: p3=312, p5=317
    # Vertical 3: p0=13, p7=14
    MOUTH_INDICES: List[int] = [78, 82, 312, 308, 317, 87, 13, 14]
    MOUTH_MAR_PAIRS: Tuple[int, int, int, int, int, int, int, int] = (78, 308, 82, 87, 312, 317, 13, 14)

    # Face outline/corners for Head Pose Estimation (3D points comparison)
    # Left eye corner, Right eye corner, Nose tip, Mouth left, Mouth right, Chin
    HEAD_POSE_LANDMARKS: List[int] = [33, 263, 1, 61, 291, 152]


# Default Metric Thresholds (Adjusted dynamically by personalization engine)
class Thresholds:
    # Eye Aspect Ratio (EAR)
    DEFAULT_EAR_THRESHOLD: float = 0.22  # Below this indicates closed eye
    EAR_CLOSED_CONSECUTIVE_FRAMES: int = 9  # ~300ms at 30 fps

    # Mouth Aspect Ratio (MAR)
    DEFAULT_MAR_THRESHOLD: float = 0.55  # Above this indicates yawning

    # PERCLOS (Percentage of Eye Closure over window)
    DEFAULT_PERCLOS_THRESHOLD: float = 0.15  # Alert if eyes are closed > 15% of the window

    # Blink Rate (Blinks per minute)
    MIN_NORMAL_BLINK_RATE: float = 8.0
    MAX_NORMAL_BLINK_RATE: float = 24.0

    # Head Pose Limits (Degrees)
    MAX_PITCH_THRESHOLD: float = 18.0  # Nodding up/down limit
    MAX_YAW_THRESHOLD: float = 22.0    # Looking left/right limit
    MAX_ROLL_THRESHOLD: float = 15.0   # Head tilt limit

    # Gaze Deviation Limit (Ratio of offset from center iris to eye contour corner)
    DEFAULT_GAZE_THRESHOLD: float = 0.25


# Sliding Window & Buffer Sizes
class BufferSizes:
    METRIC_WINDOW_FRAMES: int = 150  # ~5 seconds window at 30fps for instantaneous calculations
    PERCLOS_WINDOW_FRAMES: int = 1800  # ~60 seconds window at 30fps for long-term fatigue metrics
    CALIBRATION_MIN_FRAMES: int = 300  # ~10 seconds of steady face tracking needed for baseline


# ML Engine Constants
class MLConstants:
    COGNITIVE_MODEL_NAME: str = "cognitive_load_xgb.joblib"
    RISK_MODEL_NAME: str = "accident_risk_lgb.joblib"
    EMBEDDING_MODEL_NAME: str = "driver_face_encoder.joblib"
    ANOMALY_MODEL_NAME: str = "anomaly_isolation_forest.joblib"


# Advisory Recommendations
class Recommendations:
    DROWSINESS_ADVICE: List[str] = [
        "High fatigue detected. Please pull over at a safe spot immediately.",
        "Consider drinking a caffeinated beverage or cold water.",
        "Take a brisk 15-minute walk to restore alertness.",
        "Activate high-airflow ventilation or roll down windows."
    ]
    DISTRACTION_ADVICE: List[str] = [
        "Keep eyes focused on the road ahead.",
        "Adjust mirrors and seating before driving, not during.",
        "Avoid using handheld mobile devices while operating the vehicle."
    ]
    COGNITIVE_ADVICE: List[str] = [
        "High cognitive stress detected. Reduce dashboard distractions.",
        "Turn down high-volume audio or music playback.",
        "Avoid complex conversations or heavy thinking tasks while maneuvering.",
        "Take a brief mental recovery break if driving in heavy traffic."
    ]
