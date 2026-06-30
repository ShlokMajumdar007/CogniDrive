"""CognitiveLoadEngine — Driver attention and stress estimation engine.

Wraps the CognitiveModel to perform offline attention, stress, and CLI predictions 
from standard normalized feature vectors.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from backend.ml.inference.cognitive_model import CognitiveModel, CognitiveResult

logger = logging.getLogger("CogniDrive.CognitiveLoadEngine")


class CognitiveLoadEngine:
    """Wrapper engine around the cognitive inference model."""

    def __init__(self) -> None:
        """Initializes the engine with the singleton CognitiveModel."""
        self._model = CognitiveModel.get_instance()
        logger.info("CognitiveLoadEngine initialized.")

    def estimate_cognitive_load(self, feature_vector: np.ndarray) -> CognitiveResult:
        """Estimates cognitive parameters from a single 21-D feature vector.

        Args:
            feature_vector: Normalised feature vector array of shape (21,).

        Returns:
            CognitiveResult containing attention, stress, CLI and flags.
        """
        return self._model.predict(feature_vector)

    def estimate_batch(self, feature_matrix: np.ndarray) -> List[CognitiveResult]:
        """Runs estimation over a batch matrix of feature vectors.

        Args:
            feature_matrix: Matrix of shape (N, 21).

        Returns:
            List of CognitiveResults.
        """
        return self._model.predict_batch(feature_matrix)
