"""Cognitive load model to predict driver attention, stress, and workload index."""

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

from backend.app.config import get_model_path
from backend.app.constants import MLConstants

FEATURE_DIM: int = 21

_CLI_ATTENTION_WEIGHT: float = 0.60
_CLI_STRESS_WEIGHT: float = 0.40


@dataclass
class CognitiveResult:
    """Cognitive assessment outputs for a frame."""
    attention_score: float = 100.0
    stress_score: float = 0.0
    cli: float = 0.0
    is_overloaded: bool = False
    is_highly_stressed: bool = False
    is_inattentive: bool = False
    model_version: str = "1.0.0"
    feature_importances: Dict[str, float] = field(default_factory=dict)
    is_fallback: bool = False


class _FallbackCognitiveModel:
    """Heuristic fallback calculation when the LightGBM model isn't available."""

    def predict(self, x: np.ndarray) -> CognitiveResult:
        ear_mean = float(x[2])
        perclos = float(x[4])
        fatigue_prob = float(x[5])
        off_road = float(x[10])
        head_distracted = float(x[14])
        prev_stress_norm = float(x[16])

        attention_raw = (
            1.0
            - 0.40 * off_road
            - 0.25 * head_distracted
            - 0.25 * max(0.0, (0.28 - ear_mean) / 0.28)
            - 0.10 * perclos
        )
        attention_score = float(np.clip(attention_raw * 100.0, 0.0, 100.0))

        stress_raw = (
            0.50 * fatigue_prob
            + 0.30 * perclos
            + 0.20 * prev_stress_norm
        )
        stress_score = float(np.clip(stress_raw * 100.0, 0.0, 100.0))

        cli = float(
            np.clip(
                _CLI_ATTENTION_WEIGHT * (100.0 - attention_score)
                + _CLI_STRESS_WEIGHT * stress_score,
                0.0,
                100.0,
            )
        )

        return CognitiveResult(
            attention_score=round(attention_score, 2),
            stress_score=round(stress_score, 2),
            cli=round(cli, 2),
            is_overloaded=cli > 70.0,
            is_highly_stressed=stress_score > 70.0,
            is_inattentive=attention_score < 40.0,
            model_version="1.0.0-fallback",
            is_fallback=True,
        )


class CognitiveModel:
    """Wrapper for LightGBM cognitive load model with fallback options."""

    _instance: Optional["CognitiveModel"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_path: Optional[Path] = None) -> None:
        if model_path is None:
            model_path = get_model_path(MLConstants.COGNITIVE_MODEL_NAME)
        self._model_path = model_path
        self._model, self._is_fallback = self._load_model(model_path)
        self._model_version = "1.0.0-fallback" if self._is_fallback else "1.0.0"
        logger.info(
            "CognitiveModel initialized (fallback=%s, version=%s)",
            self._is_fallback,
            self._model_version,
        )

    @staticmethod
    def _load_model(path: Path):
        if _JOBLIB_AVAILABLE and path.exists():
            try:
                model = joblib.load(path)
                logger.info("Loaded CognitiveModel from %s", path)
                return model, False
            except Exception as exc:
                logger.warning("Failed to load CognitiveModel from %s: %s", path, exc)
        return _FallbackCognitiveModel(), True

    @classmethod
    def get_instance(cls, model_path: Optional[Path] = None) -> "CognitiveModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_path=model_path)
        return cls._instance

    def predict(self, feature_vector: np.ndarray) -> CognitiveResult:
        """Infers attention, stress, and combined cognitive load index."""
        x = feature_vector.reshape(1, -1).astype(np.float32)

        if self._is_fallback:
            return self._model.predict(x.squeeze(0))

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*feature names.*")
                predictions = self._model.predict(x)

            if predictions.ndim == 2 and predictions.shape[1] == 3:
                attention = float(np.clip(predictions[0, 0], 0.0, 100.0))
                stress = float(np.clip(predictions[0, 1], 0.0, 100.0))
                cli_raw = float(np.clip(predictions[0, 2], 0.0, 100.0))
            else:
                cli_raw = float(np.clip(predictions[0], 0.0, 100.0))
                attention = float(np.clip(100.0 - cli_raw * 0.8, 0.0, 100.0))
                stress = float(np.clip(cli_raw * 0.6, 0.0, 100.0))

            cli = float(
                np.clip(
                    _CLI_ATTENTION_WEIGHT * (100.0 - attention)
                    + _CLI_STRESS_WEIGHT * stress,
                    0.0,
                    100.0,
                )
            )
            return CognitiveResult(
                attention_score=round(attention, 2),
                stress_score=round(stress, 2),
                cli=round(cli, 2),
                is_overloaded=cli > 70.0,
                is_highly_stressed=stress > 70.0,
                is_inattentive=attention < 40.0,
                model_version=self._model_version,
                is_fallback=False,
            )
        except Exception as exc:
            logger.error("CognitiveModel prediction failed, using fallback: %s", exc)
            return _FallbackCognitiveModel().predict(x.squeeze(0))

    def predict_batch(self, feature_matrix: np.ndarray) -> List[CognitiveResult]:
        return [self.predict(row) for row in feature_matrix]

    @property
    def is_fallback(self) -> bool:
        return self._is_fallback

    @property
    def model_version(self) -> str:
        return self._model_version
