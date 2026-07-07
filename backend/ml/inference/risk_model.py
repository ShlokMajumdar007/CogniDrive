"""Accident risk inference model to predict immediate driving hazards from driver state."""

from __future__ import annotations

import logging
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from backend.database.models.driving_metrics import DriverState
from backend.ml.inference.cognitive_model import CognitiveResult
from backend.app.config import get_model_path
from backend.app.constants import MLConstants

FEATURE_DIM: int = 21

_RISK_LOW = 0.30
_RISK_MEDIUM = 0.60
_RISK_HIGH = 0.80


@dataclass
class RiskResult:
    """Accident risk prediction outputs."""
    risk_score: float = 0.0
    risk_level: str = "LOW"
    driver_state: DriverState = DriverState.NORMAL
    fatigue_probability: float = 0.0
    distraction_probability: float = 0.0
    aggression_score: float = 0.0
    model_version: str = "1.0.0"
    shap_values: Dict[str, float] = field(default_factory=dict)
    is_fallback: bool = False


def _classify_risk_level(score: float) -> str:
    if score >= _RISK_HIGH:
        return "CRITICAL"
    elif score >= _RISK_MEDIUM:
        return "HIGH"
    elif score >= _RISK_LOW:
        return "MEDIUM"
    return "LOW"


def _classify_driver_state(
    risk_score: float,
    fatigue_prob: float,
    distraction_prob: float,
    cli: float,
) -> DriverState:
    if risk_score >= _RISK_HIGH:
        return DriverState.HIGH_RISK
    if fatigue_prob >= 0.60:
        return DriverState.FATIGUED
    if distraction_prob >= 0.60:
        return DriverState.DISTRACTED
    if cli >= 70.0:
        return DriverState.OVERLOADED
    return DriverState.NORMAL


class _FallbackRiskModel:
    """Heuristic fallback risk calculations when XGBoost model is missing."""

    def predict(
        self, x: np.ndarray, cognitive_result: Optional[CognitiveResult] = None
    ) -> RiskResult:
        perclos = float(x[4])
        fatigue_prob = float(x[5])
        off_road = float(x[10])
        head_distracted = float(x[14])
        prev_risk = float(x[18])

        cli = cognitive_result.cli if cognitive_result else float(x[17]) * 100.0

        risk_raw = (
            0.30 * fatigue_prob
            + 0.25 * perclos
            + 0.20 * off_road
            + 0.15 * head_distracted
            + 0.10 * (cli / 100.0)
        )
        risk_score = float(np.clip(0.80 * risk_raw + 0.20 * prev_risk, 0.0, 1.0))
        distraction_prob = float(np.clip(0.60 * off_road + 0.40 * head_distracted, 0.0, 1.0))

        driver_state = _classify_driver_state(
            risk_score, fatigue_prob, distraction_prob, cli
        )

        return RiskResult(
            risk_score=round(risk_score, 4),
            risk_level=_classify_risk_level(risk_score),
            driver_state=driver_state,
            fatigue_probability=round(fatigue_prob, 4),
            distraction_probability=round(distraction_prob, 4),
            aggression_score=0.0,
            model_version="1.0.0-fallback",
            is_fallback=True,
        )


class RiskModel:
    """Wrapper for the XGBoost accident risk inference model."""

    _instance: Optional["RiskModel"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_path: Optional[Path] = None) -> None:
        if model_path is None:
            model_path = get_model_path(MLConstants.RISK_MODEL_NAME)
        self._model_path = model_path
        self._model, self._is_fallback = self._load_model(model_path)
        self._model_version = "1.0.0-fallback" if self._is_fallback else "1.0.0"
        logger.info(
            "RiskModel initialized (fallback=%s, version=%s)",
            self._is_fallback,
            self._model_version,
        )

    @staticmethod
    def _load_model(path: Path):
        if _JOBLIB_AVAILABLE and path.exists():
            try:
                model = joblib.load(path)
                logger.info("Loaded RiskModel from %s", path)
                return model, False
            except Exception as exc:
                logger.warning("Failed to load RiskModel from %s: %s", path, exc)
        return _FallbackRiskModel(), True

    @classmethod
    def get_instance(cls, model_path: Optional[Path] = None) -> "RiskModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_path=model_path)
        return cls._instance

    def predict(
        self,
        feature_vector: np.ndarray,
        cognitive_result: Optional[CognitiveResult] = None,
    ) -> RiskResult:
        """Infers accident risk level and driver state from feature vector."""
        x = feature_vector.reshape(1, -1).astype(np.float32)

        if self._is_fallback:
            return self._model.predict(x.squeeze(0), cognitive_result)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*feature names.*")
                proba = self._model.predict_proba(x)

            risk_score = float(np.clip(proba[0, 1], 0.0, 1.0))
            fatigue_prob = float(x[0, 5])
            distraction_prob = float(
                np.clip(float(x[0, 10]) * 0.6 + float(x[0, 14]) * 0.4, 0.0, 1.0)
            )
            cli = cognitive_result.cli if cognitive_result else float(x[0, 17]) * 100.0

            driver_state = _classify_driver_state(
                risk_score, fatigue_prob, distraction_prob, cli
            )

            return RiskResult(
                risk_score=round(risk_score, 4),
                risk_level=_classify_risk_level(risk_score),
                driver_state=driver_state,
                fatigue_probability=round(fatigue_prob, 4),
                distraction_probability=round(distraction_prob, 4),
                aggression_score=0.0,
                model_version=self._model_version,
                is_fallback=False,
            )
        except Exception as exc:
            logger.error("RiskModel prediction failed, using fallback: %s", exc)
            return _FallbackRiskModel().predict(x.squeeze(0), cognitive_result)

    def predict_batch(
        self,
        feature_matrix: np.ndarray,
        cognitive_results: Optional[List[CognitiveResult]] = None,
    ) -> List[RiskResult]:
        cog = cognitive_results or [None] * len(feature_matrix)
        return [self.predict(row, c) for row, c in zip(feature_matrix, cog)]

    @property
    def is_fallback(self) -> bool:
        return self._is_fallback

    @property
    def model_version(self) -> str:
        return self._model_version
