"""Cognitive Load Model — real-time inference for attention, stress, and CLI.

Predicts three cognitive state outputs from the 21-dimensional feature vector:
    1. **Attention Score** [0, 100]: How focused the driver is.
    2. **Stress Score** [0, 100]: Physiological stress level.
    3. **Cognitive Load Index (CLI)** [0, 100]: Combined mental workload index.

Model architecture:
    A LightGBM regressor trained on labelled feature-vector sequences.
    The model file is loaded from disk via ``joblib``. A fallback linear
    model is used when the trained model is unavailable.

The CLI is computed as a weighted combination of attention and stress::

    CLI = 0.60 * (100 - attention) + 0.40 * stress

Typical usage::

    model = CognitiveModel.get_instance()
    result = model.predict(feature_vector)
    print(result.cli, result.attention_score, result.stress_score)
"""

from __future__ import annotations

import logging
import threading
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

FEATURE_DIM: int = 21
DEFAULT_MODEL_PATH: Path = Path("backend/ml/models/cognitive_model.joblib")

# CLI weighting constants
_CLI_ATTENTION_WEIGHT: float = 0.60
_CLI_STRESS_WEIGHT: float = 0.40


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CognitiveResult:
    """Cognitive load inference result for a single feature vector.

    Attributes:
        attention_score: Driver attention score [0, 100]. Higher is more attentive.
        stress_score: Physiological stress level [0, 100]. Higher is more stressed.
        cli: Cognitive Load Index [0, 100]. Higher is more overloaded.
        is_overloaded: True when CLI exceeds 70.0.
        is_highly_stressed: True when stress_score exceeds 70.0.
        is_inattentive: True when attention_score falls below 40.0.
        model_version: Version of the model used for inference.
        feature_importances: Optional SHAP-like feature importance dict.
        is_fallback: True when the fallback linear model was used.
    """

    attention_score: float = 100.0
    stress_score: float = 0.0
    cli: float = 0.0
    is_overloaded: bool = False
    is_highly_stressed: bool = False
    is_inattentive: bool = False
    model_version: str = "1.0.0"
    feature_importances: Dict[str, float] = field(default_factory=dict)
    is_fallback: bool = False


# ---------------------------------------------------------------------------
# Fallback linear model
# ---------------------------------------------------------------------------


class _FallbackCognitiveModel:
    """Simple heuristic cognitive model used when the trained model is absent.

    Computes approximate attention and stress from raw EAR, PERCLOS, and
    gaze features using hand-crafted weights based on domain knowledge.

    Feature indices used (from FEATURE_NAMES)::
        [2]  ear_mean
        [4]  perclos
        [5]  fatigue_probability
        [10] gaze_off_road
        [14] head_distracted
        [16] stress_score_norm
        [17] cli_norm
    """

    def predict(self, x: np.ndarray) -> CognitiveResult:
        """Predicts cognitive state from a 21-D feature vector.

        Args:
            x: Feature array of shape (21,), values normalised to model range.

        Returns:
            CognitiveResult: Fallback heuristic prediction.
        """
        ear_mean = float(x[2])           # [0, 0.5] → good: 0.28
        perclos = float(x[4])            # [0, 1]
        fatigue_prob = float(x[5])       # [0, 1]
        off_road = float(x[10])          # binary
        head_distracted = float(x[14])   # binary
        prev_stress_norm = float(x[16])  # [0, 1]
        prev_cli_norm = float(x[17])     # [0, 1]

        # Attention: penalise off-road gaze, head distraction, low EAR
        attention_raw = (
            1.0
            - 0.40 * off_road
            - 0.25 * head_distracted
            - 0.25 * max(0.0, (0.28 - ear_mean) / 0.28)  # EAR below baseline
            - 0.10 * perclos
        )
        attention_score = float(np.clip(attention_raw * 100.0, 0.0, 100.0))

        # Stress: rises with fatigue and previous stress momentum
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


# ---------------------------------------------------------------------------
# Cognitive Model
# ---------------------------------------------------------------------------


class CognitiveModel:
    """Thread-safe singleton LightGBM cognitive load inference model.

    Loads a pre-trained LightGBM pipeline from disk. Falls back to the
    heuristic model when the file is absent or cannot be loaded.

    The model expects Z-score normalised feature vectors produced by
    :class:`~backend.features.normalizer.FeatureNormalizer`.

    Attributes:
        _instance: Class-level singleton reference.
        _lock: Threading lock guarding singleton creation.
    """

    _instance: Optional["CognitiveModel"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        """Initialises the cognitive model.

        Args:
            model_path: Path to the ``joblib``-serialised LightGBM pipeline.
        """
        self._model_path = model_path
        self._model, self._is_fallback = self._load_model(model_path)
        self._model_version = "1.0.0-fallback" if self._is_fallback else "1.0.0"
        logger.info(
            "CognitiveModel initialised — fallback=%s, version=%s",
            self._is_fallback,
            self._model_version,
        )

    @staticmethod
    def _load_model(path: Path):  # type: ignore[return]
        if _JOBLIB_AVAILABLE and path.exists():
            try:
                model = joblib.load(path)
                logger.info("CognitiveModel loaded from %s.", path)
                return model, False
            except Exception as exc:
                logger.warning("Failed to load CognitiveModel from %s: %s", path, exc)
        return _FallbackCognitiveModel(), True

    @classmethod
    def get_instance(
        cls, model_path: Path = DEFAULT_MODEL_PATH
    ) -> "CognitiveModel":
        """Returns the singleton :class:`CognitiveModel` instance.

        Args:
            model_path: Forwarded to ``__init__`` on first call.

        Returns:
            CognitiveModel: Shared singleton.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_path=model_path)
        return cls._instance

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, feature_vector: np.ndarray) -> CognitiveResult:
        """Runs cognitive load inference on a single feature vector.

        Args:
            feature_vector: NumPy array of shape (21,), dtype float32,
                normalised by :class:`~backend.features.normalizer.FeatureNormalizer`.

        Returns:
            CognitiveResult: Predicted attention, stress, CLI, and flags.
        """
        x = feature_vector.reshape(1, -1).astype(np.float32)

        if self._is_fallback:
            return self._model.predict(x.squeeze(0))

        try:
            predictions = self._model.predict(x)  # shape (1, 3) or (1,)
            if predictions.ndim == 2 and predictions.shape[1] == 3:
                attention = float(np.clip(predictions[0, 0], 0.0, 100.0))
                stress = float(np.clip(predictions[0, 1], 0.0, 100.0))
                cli_raw = float(np.clip(predictions[0, 2], 0.0, 100.0))
            else:
                # Single output — treat as CLI
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
            logger.error("CognitiveModel.predict failed: %s. Using fallback.", exc)
            return self._load_model(Path(""))[0].predict(x.squeeze(0))

    def predict_batch(
        self, feature_matrix: np.ndarray
    ) -> List[CognitiveResult]:
        """Runs inference on an (N, 21) feature matrix.

        Args:
            feature_matrix: Array of shape (N, 21), dtype float32.

        Returns:
            List[CognitiveResult]: N results in input order.
        """
        return [self.predict(row) for row in feature_matrix]

    @property
    def is_fallback(self) -> bool:
        """True when the heuristic fallback model is active."""
        return self._is_fallback

    @property
    def model_version(self) -> str:
        """String version tag of the loaded model."""
        return self._model_version
