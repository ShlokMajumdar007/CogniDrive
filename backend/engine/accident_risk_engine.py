"""AccidentRiskEngine — Imminent accident and driving risk estimation engine.

Wraps the RiskModel to perform real-time offline accident risk predictions 
from feature vectors and cognitive results.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

# Project imports with fallback
try:
    from backend.ml.inference.risk_model import RiskModel, RiskResult
    from backend.ml.inference.cognitive_model import CognitiveResult
except ImportError:
    from ml.inference.risk_model import RiskModel, RiskResult  # type: ignore[no-redef]
    from ml.inference.cognitive_model import CognitiveResult  # type: ignore[no-redef]

logger = logging.getLogger("CogniDrive.AccidentRiskEngine")


class AccidentRiskEngine:
    """Wrapper engine around the accident risk inference model."""

    def __init__(self) -> None:
        """Initializes the engine with the singleton RiskModel."""
        self._model = RiskModel.get_instance()
        logger.info("AccidentRiskEngine initialized.")

    def estimate_accident_risk(
        self,
        feature_vector: np.ndarray,
        cognitive_result: Optional[CognitiveResult] = None,
    ) -> RiskResult:
        """Estimates accident risk from a single 21-D feature vector.

        Args:
            feature_vector: Normalised feature vector array of shape (21,).
            cognitive_result: Optional CognitiveResult containing CLI.

        Returns:
            RiskResult containing risk score, risk level, driver state, etc.
        """
        return self._model.predict(feature_vector, cognitive_result)

    def estimate_batch(
        self,
        feature_matrix: np.ndarray,
        cognitive_results: Optional[List[CognitiveResult]] = None,
    ) -> List[RiskResult]:
        """Runs risk estimation over a batch of feature vectors.

        Args:
            feature_matrix: Matrix of shape (N, 21).
            cognitive_results: Optional list of N CognitiveResult objects.

        Returns:
            List of RiskResults.
        """
        return self._model.predict_batch(feature_matrix, cognitive_results)
