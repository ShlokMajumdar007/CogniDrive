"""StateClassifier — Driver behavioral state classification engine.

Applies prioritized classification rules to resolve the driver's current overall 
behavioral state (e.g. NORMAL, FATIGUED, DISTRACTED, OVERLOADED, or HIGH_RISK) 
by evaluating ML model predictions against active personalized thresholds.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from backend.database.models.driving_metrics import DriverState

logger = logging.getLogger("CogniDrive.StateClassifier")


class StateClassifier:
    """Classifies the driver's active state using model outputs and thresholds."""

    def __init__(self) -> None:
        """Initializes the state classifier."""
        logger.info("StateClassifier initialized.")

    def classify_state(
        self,
        risk_score: float,
        fatigue_probability: float,
        distraction_probability: float,
        cli: float,
        is_anomaly: bool = False,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> DriverState:
        """Determines the most critical DriverState based on a prioritized hierarchy.

        Priority hierarchy:
            1. HIGH_RISK: If risk_score exceeds high threshold.
            2. FATIGUED: If fatigue_probability exceeds threshold.
            3. DISTRACTED: If distraction_probability exceeds threshold.
            4. OVERLOADED: If CLI exceeds threshold.
            5. NORMAL: Default baseline state.

        Args:
            risk_score: Probability of accident [0, 1].
            fatigue_probability: Probability of drowsiness [0, 1].
            distraction_probability: Probability of driver distraction [0, 1].
            cli: Cognitive Load Index [0, 100].
            is_anomaly: True if the behavior is anomalous.
            thresholds: Optional dictionary of active thresholds (personalized or global defaults).

        Returns:
            DriverState: Evaluated classification.
        """
        # Resolve thresholds (fallback to defaults if thresholds dictionary is missing)
        thresh_risk = thresholds.get("risk_score_high", 0.60) if thresholds else 0.60
        thresh_fatigue = thresholds.get("fatigue_probability", 0.60) if thresholds else 0.60
        thresh_distraction = thresholds.get("distraction_probability", 0.60) if thresholds else 0.60
        thresh_cli = thresholds.get("cli", 70.0) if thresholds else 70.0

        # Prioritized evaluation
        if risk_score >= thresh_risk or is_anomaly:
            return DriverState.HIGH_RISK
        if fatigue_probability >= thresh_fatigue:
            return DriverState.FATIGUED
        if distraction_probability >= thresh_distraction:
            return DriverState.DISTRACTED
        if cli >= thresh_cli:
            return DriverState.OVERLOADED

        return DriverState.NORMAL
