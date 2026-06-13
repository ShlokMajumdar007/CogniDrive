"""Accident Risk Model — binary and probabilistic risk inference.

Predicts the probability that the current driver state will lead to an
accident or near-miss event within the next 30 seconds. Outputs:

    - **Risk Score** [0, 1]: Continuous probability of imminent accident.
    - **Risk Level**: Discrete categorical label (LOW / MEDIUM / HIGH / CRITICAL).
    - **Driver State**: :class:`~backend.database.models.driving_metrics.DriverState`
      classification.

Model architecture:
    An XGBoost binary classifier trained on labelled high-risk driving
    sequences. The risk score is the model's positive-class probability
    output. A fallback heuristic model is used when the trained model
    is unavailable.

Risk thresholds::

    risk < 0.30  → LOW
    0.30 ≤ risk < 0.60  → MEDIUM
    0.60 ≤ risk < 0.80  → HIGH
    risk ≥ 0.80  → CRITICAL

Typical usage::

    model = RiskModel.get_instance()
    result = model.predict(feature_vector, cognitive_result)
    print(result.risk_score, result.risk_level, result.driver_state)
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

# Lazy import to avoid circular dependency
try:
    from backend.database.models.driving_metrics import DriverState
except ImportError:
    from database.models.driving_metrics import DriverState  # type: ignore[no-redef]

try:
    from backend.ml.inference.cognitive_model import CognitiveResult
except ImportError:
    from ml.inference.cognitive_model import CognitiveResult  # type: ignore[no-redef]

FEATURE_DIM: int = 21
DEFAULT_MODEL_PATH: Path = Path("backend/ml/models/risk_model.joblib")

# Risk level thresholds
_RISK_LOW = 0.30
_RISK_MEDIUM = 0.60
_RISK_HIGH = 0.80


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RiskResult:
    """Accident risk inference result for a single feature vector.

    Attributes:
        risk_score: Probability of imminent accident [0, 1].
        risk_level: Discrete label: LOW / MEDIUM / HIGH / CRITICAL.
        driver_state: :class:`DriverState` classification.
        fatigue_probability: Fatigue sub-probability [0, 1].
        distraction_probability: Distraction sub-probability [0, 1].
        aggression_score: Driving aggression sub-score [0, 1].
        model_version: Version of the model used.
        shap_values: Optional SHAP feature contributions dict.
        is_fallback: True when the fallback heuristic was used.
    """

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
    """Converts a continuous risk score to a discrete label.

    Args:
        score: Risk probability in [0, 1].

    Returns:
        str: One of 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'.
    """
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
    """Maps continuous scores to the most critical DriverState.

    Priority order: HIGH_RISK > FATIGUED > DISTRACTED > OVERLOADED > NORMAL.

    Args:
        risk_score: Overall risk probability.
        fatigue_prob: Fatigue sub-probability.
        distraction_prob: Distraction sub-probability.
        cli: Cognitive Load Index [0, 100].

    Returns:
        DriverState: Most critical state for this frame.
    """
    if risk_score >= _RISK_HIGH:
        return DriverState.HIGH_RISK
    if fatigue_prob >= 0.60:
        return DriverState.FATIGUED
    if distraction_prob >= 0.60:
        return DriverState.DISTRACTED
    if cli >= 70.0:
        return DriverState.OVERLOADED
    return DriverState.NORMAL


# ---------------------------------------------------------------------------
# Fallback heuristic risk model
# ---------------------------------------------------------------------------


class _FallbackRiskModel:
    """Heuristic risk model based on domain-knowledge feature thresholds.

    Used when the XGBoost model file is not available. Combines PERCLOS,
    EAR, gaze off-road, head distraction, and CLI to estimate risk.

    Feature indices used::
        [2]  ear_mean
        [4]  perclos
        [5]  fatigue_probability
        [10] gaze_off_road (binary)
        [14] head_distracted (binary)
        [17] cli_norm
        [18] risk_score (previous frame)
    """

    def predict(
        self, x: np.ndarray, cognitive_result: Optional[CognitiveResult] = None
    ) -> RiskResult:
        """Heuristic risk prediction.

        Args:
            x: Feature vector of shape (21,).
            cognitive_result: Optional cognitive model output for CLI.

        Returns:
            RiskResult: Heuristic risk estimate.
        """
        perclos = float(x[4])
        fatigue_prob = float(x[5])
        off_road = float(x[10])
        head_distracted = float(x[14])
        prev_risk = float(x[18])

        cli = cognitive_result.cli if cognitive_result else float(x[17]) * 100.0

        # Combine sub-risks with weighted sum
        risk_raw = (
            0.30 * fatigue_prob
            + 0.25 * perclos
            + 0.20 * off_road
            + 0.15 * head_distracted
            + 0.10 * (cli / 100.0)
        )
        # Apply temporal smoothing with previous frame's risk
        risk_score = float(np.clip(0.80 * risk_raw + 0.20 * prev_risk, 0.0, 1.0))

        distraction_prob = float(np.clip(0.60 * off_road + 0.40 * head_distracted, 0.0, 1.0))
        aggression_score = 0.0  # Not computable from biometrics alone

        driver_state = _classify_driver_state(
            risk_score, fatigue_prob, distraction_prob, cli
        )

        return RiskResult(
            risk_score=round(risk_score, 4),
            risk_level=_classify_risk_level(risk_score),
            driver_state=driver_state,
            fatigue_probability=round(fatigue_prob, 4),
            distraction_probability=round(distraction_prob, 4),
            aggression_score=aggression_score,
            model_version="1.0.0-fallback",
            is_fallback=True,
        )


# ---------------------------------------------------------------------------
# Risk Model
# ---------------------------------------------------------------------------


class RiskModel:
    """Thread-safe singleton XGBoost accident risk inference model.

    Loads a pre-trained XGBoost classifier from disk. Falls back to the
    heuristic model when unavailable.

    Attributes:
        _instance: Class-level singleton reference.
        _lock: Threading lock guarding singleton creation.
    """

    _instance: Optional["RiskModel"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        """Initialises the risk model.

        Args:
            model_path: Path to the ``joblib``-serialised XGBoost pipeline.
        """
        self._model_path = model_path
        self._model, self._is_fallback = self._load_model(model_path)
        self._model_version = "1.0.0-fallback" if self._is_fallback else "1.0.0"
        logger.info(
            "RiskModel initialised — fallback=%s, version=%s",
            self._is_fallback,
            self._model_version,
        )

    @staticmethod
    def _load_model(path: Path):  # type: ignore[return]
        if _JOBLIB_AVAILABLE and path.exists():
            try:
                model = joblib.load(path)
                logger.info("RiskModel loaded from %s.", path)
                return model, False
            except Exception as exc:
                logger.warning("Failed to load RiskModel from %s: %s", path, exc)
        return _FallbackRiskModel(), True

    @classmethod
    def get_instance(
        cls, model_path: Path = DEFAULT_MODEL_PATH
    ) -> "RiskModel":
        """Returns the singleton :class:`RiskModel` instance.

        Args:
            model_path: Forwarded to ``__init__`` on first call.

        Returns:
            RiskModel: Shared singleton.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_path=model_path)
        return cls._instance

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        feature_vector: np.ndarray,
        cognitive_result: Optional[CognitiveResult] = None,
    ) -> RiskResult:
        """Predicts accident risk from a single feature vector.

        Args:
            feature_vector: NumPy array of shape (21,), dtype float32.
            cognitive_result: Optional :class:`CognitiveResult` for CLI
                enrichment of the fallback model.

        Returns:
            RiskResult: Predicted risk score, level, state, and sub-scores.
        """
        x = feature_vector.reshape(1, -1).astype(np.float32)

        if self._is_fallback:
            return self._model.predict(x.squeeze(0), cognitive_result)

        try:
            proba = self._model.predict_proba(x)  # (1, 2)
            risk_score = float(np.clip(proba[0, 1], 0.0, 1.0))

            # Extract sub-probabilities if the model exposes them
            fatigue_prob = float(x[0, 5])  # from feature vector
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
            logger.error("RiskModel.predict failed: %s. Using fallback.", exc)
            return _FallbackRiskModel().predict(x.squeeze(0), cognitive_result)

    def predict_batch(
        self,
        feature_matrix: np.ndarray,
        cognitive_results: Optional[List[CognitiveResult]] = None,
    ) -> List[RiskResult]:
        """Runs risk inference on an (N, 21) feature matrix.

        Args:
            feature_matrix: Array of shape (N, 21), dtype float32.
            cognitive_results: Optional list of N cognitive results.

        Returns:
            List[RiskResult]: N results in input order.
        """
        cog = cognitive_results or [None] * len(feature_matrix)  # type: ignore[list-item]
        return [self.predict(row, c) for row, c in zip(feature_matrix, cog)]

    @property
    def is_fallback(self) -> bool:
        """True when the heuristic fallback model is active."""
        return self._is_fallback

    @property
    def model_version(self) -> str:
        """String version tag of the loaded model."""
        return self._model_version
