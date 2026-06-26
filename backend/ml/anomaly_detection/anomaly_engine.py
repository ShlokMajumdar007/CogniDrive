"""AnomalyEngine — Real-time offline anomaly detection for driving behavior.

Detects erratic driving behavior and atypical biometric patterns using an Isolation Forest 
model. Falls back to a deterministic multi-signal heuristic outlier detector when the trained 
model is unavailable.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("CogniDrive.AnomalyEngine")

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

FEATURE_DIM: int = 21
DEFAULT_MODEL_PATH: Path = Path("backend/ml/models_saved/anomaly_isolation_forest.joblib")


@dataclass
class AnomalyResult:
    """Outcome of anomaly detection inference.

    Attributes:
        is_anomaly: True if the behavior is anomalous (outlier).
        anomaly_score: Continuous outlier score (closer to 1.0 is more anomalous).
        is_fallback: True if the heuristic fallback model was used.
        model_version: Version tag of the model used.
    """

    is_anomaly: bool = False
    anomaly_score: float = 0.0
    is_fallback: bool = False
    model_version: str = "1.0.0"


class _HeuristicAnomalyDetector:
    """Heuristic fallback anomaly detector.

    Scores anomaly based on cumulative threshold breaches of key biometrics
    and normalized values in the 21-D feature vector.
    """

    def predict(self, x: np.ndarray) -> AnomalyResult:
        """Heuristically detects anomaly in a single 21-D feature vector.

        Features of interest (zero-indexed):
            [2]  ear_mean (normal: ~0.28, low: <0.18 is outlier)
            [3]  mar (normal: ~0.05, high: >0.6 is outlier)
            [4]  perclos (normal: <0.15, high: >0.30 is outlier)
            [10] gaze_off_road (binary)
            [14] head_distracted (binary)
            [17] cli_norm (normal: <0.30, high: >0.70 is outlier)
            [18] risk_score (normal: <0.30, high: >0.60 is outlier)
        """
        ear_mean = float(x[2])
        mar = float(x[3])
        perclos = float(x[4])
        gaze_off_road = float(x[10])
        head_distracted = float(x[14])
        cli_norm = float(x[17])
        risk_score = float(x[18])

        score = 0.0

        # Accumulate outlier factors
        if ear_mean < 0.18:
            score += 0.25
        if mar > 0.60:
            score += 0.20
        if perclos > 0.30:
            score += 0.30
        if gaze_off_road > 0.5:
            score += 0.15
        if head_distracted > 0.5:
            score += 0.15
        if cli_norm > 0.70:
            score += 0.20
        if risk_score > 0.60:
            score += 0.30

        # Normalize score to [0, 1]
        anomaly_score = float(np.clip(score, 0.0, 1.0))
        is_anomaly = anomaly_score >= 0.50

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=round(anomaly_score, 4),
            is_fallback=True,
            model_version="1.0.0-fallback",
        )


class AnomalyEngine:
    """Thread-safe singleton wrapping the Isolation Forest offline anomaly detector."""

    _instance: Optional["AnomalyEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        """Initializes the AnomalyEngine by loading the Isolation Forest model."""
        self._model_path = model_path
        self._model, self._is_fallback = self._load_model(model_path)
        self._model_version = "1.0.0-fallback" if self._is_fallback else "1.0.0"
        logger.info(
            "AnomalyEngine initialized — fallback=%s, version=%s",
            self._is_fallback,
            self._model_version,
        )

    @staticmethod
    def _load_model(path: Path):  # type: ignore[return]
        """Loads scikit-learn Isolation Forest model via joblib."""
        if _JOBLIB_AVAILABLE and _SKLEARN_AVAILABLE and path.exists():
            try:
                model = joblib.load(path)
                logger.info("AnomalyEngine loaded model successfully from %s.", path)
                return model, False
            except Exception as exc:
                logger.warning("Failed to load AnomalyEngine from %s: %s. Using fallback.", path, exc)
        return _HeuristicAnomalyDetector(), True

    @classmethod
    def get_instance(cls, model_path: Path = DEFAULT_MODEL_PATH) -> "AnomalyEngine":
        """Returns the singleton AnomalyEngine instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_path=model_path)
        return cls._instance

    def predict(self, feature_vector: np.ndarray) -> AnomalyResult:
        """Runs anomaly detection on a single 21-D feature vector.

        Args:
            feature_vector: Normalised feature vector of shape (21,).

        Returns:
            AnomalyResult containing anomaly classification and continuous score.
        """
        x = feature_vector.reshape(1, -1).astype(np.float32)

        if self._is_fallback:
            return self._model.predict(x.squeeze(0))

        try:
            # IsolationForest predict returns -1 for anomalies, 1 for inliers
            pred = int(self._model.predict(x)[0])
            is_anomaly = pred == -1

            # decision_function returns raw anomaly score (higher means less anomalous, values range roughly [-0.5, 0.5])
            # We map this to [0, 1] range where 1.0 is extremely anomalous
            raw_score = float(self._model.decision_function(x)[0])
            anomaly_score = float(np.clip(0.5 - raw_score, 0.0, 1.0))

            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=round(anomaly_score, 4),
                is_fallback=False,
                model_version=self._model_version,
            )
        except Exception as exc:
            logger.error("AnomalyEngine predict failed: %s. Using fallback heuristic.", exc)
            return _HeuristicAnomalyDetector().predict(x.squeeze(0))

    @property
    def is_fallback(self) -> bool:
        """Returns True if fallback heuristic is active."""
        return self._is_fallback

    @property
    def model_version(self) -> str:
        """Returns active model version tag."""
        return self._model_version
