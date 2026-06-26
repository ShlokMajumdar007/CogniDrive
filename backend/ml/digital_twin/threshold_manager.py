"""ThresholdManager — Personalized Alert Threshold Registry.

Manages global default thresholds and per-driver overrides for all
CogniDrive alert conditions. Thresholds control when the system transitions
from monitoring to alerting for each cognitive/risk metric.

The ThresholdManager works in conjunction with PersonalizationManager:
    - PersonalizationManager *learns* driver baselines from data.
    - ThresholdManager *stores and serves* the computed alert thresholds.

Threshold hierarchy (highest priority first):
    1. Driver-specific override (set manually or by calibration).
    2. Cluster-level threshold (derived from similar drivers).
    3. Global population default.

Drift correction:
    Thresholds are periodically re-evaluated against the driver's updated
    baseline. If the baseline drifts significantly (detected by PersonalizationManager),
    the threshold is corrected proportionally.

Persistence:
    Thresholds are stored in memory with export/import helpers for
    embedding in the DriverProfile.extra_data JSON field or file cache.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("CogniDrive.ThresholdManager")

# ---------------------------------------------------------------------------
# Global default thresholds
# ---------------------------------------------------------------------------

GLOBAL_DEFAULTS: Dict[str, float] = {
    # Cognitive model outputs
    "cli": 70.0,                    # CLI > 70 → OVERLOADED
    "attention_score": 40.0,        # Attention < 40 → INATTENTIVE
    "stress_score": 70.0,           # Stress > 70 → HIGH_STRESS

    # Risk model outputs
    "risk_score_medium": 0.30,      # Risk ≥ 0.30 → MEDIUM
    "risk_score_high": 0.60,        # Risk ≥ 0.60 → HIGH
    "risk_score_critical": 0.80,    # Risk ≥ 0.80 → CRITICAL

    # Biometric thresholds
    "fatigue_probability": 0.60,    # Fatigue ≥ 0.60 → FATIGUED
    "distraction_probability": 0.60, # Distraction ≥ 0.60 → DISTRACTED
    "aggression_score": 0.70,       # Aggression ≥ 0.70 → AGGRESSIVE

    # EAR / PERCLOS
    "ear_low": 0.18,                # EAR < 0.18 → eyes closing
    "perclos": 0.20,                # PERCLOS > 20% → drowsy

    # Face authentication
    "face_verification": 0.65,      # Cosine similarity ≥ 0.65 → verified
}

# Bounds for personalized thresholds (safety clamps)
_THRESHOLD_BOUNDS: Dict[str, tuple] = {
    "cli":                     (50.0,  90.0),
    "attention_score":         (20.0,  60.0),
    "stress_score":            (50.0,  90.0),
    "risk_score_medium":       (0.20,  0.45),
    "risk_score_high":         (0.45,  0.75),
    "risk_score_critical":     (0.70,  0.95),
    "fatigue_probability":     (0.40,  0.80),
    "distraction_probability": (0.40,  0.80),
    "aggression_score":        (0.50,  0.90),
    "ear_low":                 (0.10,  0.25),
    "perclos":                 (0.10,  0.35),
    "face_verification":       (0.50,  0.80),
}


# ---------------------------------------------------------------------------
# Per-driver threshold profile
# ---------------------------------------------------------------------------


@dataclass
class DriverThresholdProfile:
    """Stores all personalized thresholds for one driver.

    Attributes:
        driver_id: Primary key of the DriverProfile.
        overrides: Per-metric threshold overrides.
        cluster_thresholds: Cluster-level thresholds for this driver's cohort.
        confidence: Personalization confidence in [0.0, 1.0].
        last_updated: UTC timestamp of the last threshold update.
        drift_corrections: Number of drift-correction events applied.
    """
    driver_id: int
    overrides: Dict[str, float] = field(default_factory=dict)
    cluster_thresholds: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    last_updated: Optional[datetime] = None
    drift_corrections: int = 0

    def get(self, metric: str, fallback: Optional[float] = None) -> float:
        """Retrieves the effective threshold for a metric.

        Priority: override > cluster > global default > fallback.

        Args:
            metric: Metric name (key in GLOBAL_DEFAULTS).
            fallback: Value to return if no threshold is found anywhere.

        Returns:
            float: Effective threshold value.
        """
        if metric in self.overrides:
            return self.overrides[metric]
        if metric in self.cluster_thresholds:
            # Blend cluster with global weighted by confidence
            cluster = self.cluster_thresholds[metric]
            global_val = GLOBAL_DEFAULTS.get(metric, fallback or 0.0)
            return float(
                self.confidence * cluster + (1.0 - self.confidence) * global_val
            )
        return GLOBAL_DEFAULTS.get(metric, fallback or 0.0)

    def set_override(self, metric: str, value: float) -> None:
        """Sets a driver-specific threshold override.

        Clamps the value to safety bounds defined in ``_THRESHOLD_BOUNDS``.

        Args:
            metric: Metric name.
            value: Desired threshold value.
        """
        if metric in _THRESHOLD_BOUNDS:
            lo, hi = _THRESHOLD_BOUNDS[metric]
            value = float(np.clip(value, lo, hi))
        self.overrides[metric] = round(value, 4)
        self.last_updated = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Serialises to a JSON-safe dictionary.

        Returns:
            Dict[str, Any]: Serialised threshold profile.
        """
        return {
            "driver_id": self.driver_id,
            "overrides": {k: round(v, 4) for k, v in self.overrides.items()},
            "cluster_thresholds": {
                k: round(v, 4) for k, v in self.cluster_thresholds.items()
            },
            "confidence": round(self.confidence, 4),
            "last_updated": (
                self.last_updated.isoformat() if self.last_updated else None
            ),
            "drift_corrections": self.drift_corrections,
        }


# ---------------------------------------------------------------------------
# ThresholdManager
# ---------------------------------------------------------------------------


class ThresholdManager:
    """Registry and resolver for personalized alert thresholds.

    Thread-safe. Maintains one ``DriverThresholdProfile`` per driver in
    memory. Provides methods to:
        - Retrieve effective thresholds (driver > cluster > global).
        - Update thresholds based on PersonalizationManager output.
        - Apply drift corrections.
        - Export/import profiles for persistence.

    Attributes:
        _profiles: Registry of per-driver threshold profiles.
        _lock: Guards concurrent profile access.
    """

    def __init__(self) -> None:
        """Initialises the ThresholdManager with an empty profile registry."""
        self._profiles: Dict[int, DriverThresholdProfile] = {}
        self._lock: threading.Lock = threading.Lock()
        logger.info("ThresholdManager: Initialized.")

    # ------------------------------------------------------------------
    # Threshold retrieval
    # ------------------------------------------------------------------

    def get_threshold(self, driver_id: int, metric: str) -> float:
        """Retrieves the effective threshold for a driver and metric.

        Args:
            driver_id: Primary key of the DriverProfile. Use 0 for global.
            metric: Metric name (key in GLOBAL_DEFAULTS).

        Returns:
            float: Effective threshold.
        """
        if driver_id == 0:
            return GLOBAL_DEFAULTS.get(metric, 0.0)
        profile = self._get_or_create_profile(driver_id)
        return profile.get(metric)

    def get_all_thresholds(self, driver_id: int) -> Dict[str, float]:
        """Returns all effective thresholds for a driver.

        Args:
            driver_id: Primary key of the DriverProfile.

        Returns:
            Dict[str, float]: Full threshold map for this driver.
        """
        profile = self._get_or_create_profile(driver_id)
        return {
            metric: profile.get(metric)
            for metric in GLOBAL_DEFAULTS
        }

    def get_global_defaults(self) -> Dict[str, float]:
        """Returns the global default threshold map.

        Returns:
            Dict[str, float]: Population-level defaults.
        """
        return dict(GLOBAL_DEFAULTS)

    # ------------------------------------------------------------------
    # Threshold updates
    # ------------------------------------------------------------------

    def set_driver_override(
        self, driver_id: int, metric: str, value: float
    ) -> float:
        """Sets a driver-specific threshold override (clamped to safety bounds).

        Args:
            driver_id: Primary key of the DriverProfile.
            metric: Metric name.
            value: Desired threshold value.

        Returns:
            float: The clamped threshold value that was stored.
        """
        profile = self._get_or_create_profile(driver_id)
        with self._lock:
            profile.set_override(metric, value)

        stored = profile.overrides[metric]
        logger.info(
            "ThresholdManager.set_driver_override: driver_id=%d, %s=%.4f.",
            driver_id,
            metric,
            stored,
        )
        return stored

    def update_from_personalization(
        self,
        driver_id: int,
        personalized_thresholds: Dict[str, float],
        confidence: float,
    ) -> None:
        """Updates a driver's thresholds from PersonalizationManager output.

        Args:
            driver_id: Primary key of the DriverProfile.
            personalized_thresholds: Dict of metric → threshold from PersonalizationManager.
            confidence: Current personalization confidence in [0.0, 1.0].
        """
        profile = self._get_or_create_profile(driver_id)
        with self._lock:
            for metric, value in personalized_thresholds.items():
                if metric in GLOBAL_DEFAULTS:
                    profile.set_override(metric, value)
            profile.confidence = float(np.clip(confidence, 0.0, 1.0))
            profile.last_updated = datetime.now(timezone.utc)

        logger.debug(
            "ThresholdManager.update_from_personalization: driver_id=%d, "
            "metrics_updated=%d, confidence=%.3f.",
            driver_id,
            len(personalized_thresholds),
            confidence,
        )

    def apply_cluster_thresholds(
        self, driver_id: int, cluster_thresholds: Dict[str, float]
    ) -> None:
        """Applies cluster-level thresholds for a driver's assigned cohort.

        Args:
            driver_id: Primary key of the DriverProfile.
            cluster_thresholds: Cluster-averaged thresholds for each metric.
        """
        profile = self._get_or_create_profile(driver_id)
        with self._lock:
            profile.cluster_thresholds.update(cluster_thresholds)
            profile.last_updated = datetime.now(timezone.utc)

        logger.debug(
            "ThresholdManager.apply_cluster_thresholds: driver_id=%d, "
            "metrics=%d.",
            driver_id,
            len(cluster_thresholds),
        )

    def apply_drift_correction(
        self,
        driver_id: int,
        drifted_metrics: list,
        drift_factor: float = 0.05,
    ) -> int:
        """Relaxes thresholds for drifted metrics to reduce false-positive rate.

        When a driver's behavior consistently exceeds a threshold (e.g. stress
        is always "high" due to their natural baseline), the threshold is
        nudged upward by ``drift_factor`` to account for the shift.

        Args:
            driver_id: Primary key of the DriverProfile.
            drifted_metrics: List of metric names with detected drift.
            drift_factor: Fractional increase per correction (e.g. 0.05 = 5%).

        Returns:
            int: Number of thresholds corrected.
        """
        profile = self._get_or_create_profile(driver_id)
        corrected = 0

        with self._lock:
            for metric in drifted_metrics:
                current = profile.get(metric)
                bound = _THRESHOLD_BOUNDS.get(metric)
                if bound is None:
                    continue
                lo, hi = bound
                corrected_value = float(np.clip(current * (1.0 + drift_factor), lo, hi))
                profile.overrides[metric] = round(corrected_value, 4)
                corrected += 1

            if corrected > 0:
                profile.drift_corrections += 1
                profile.last_updated = datetime.now(timezone.utc)

        logger.info(
            "ThresholdManager.apply_drift_correction: driver_id=%d, "
            "corrected=%d metrics.",
            driver_id,
            corrected,
        )
        return corrected

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def _get_or_create_profile(self, driver_id: int) -> DriverThresholdProfile:
        """Thread-safe profile retrieval or creation."""
        if driver_id not in self._profiles:
            with self._lock:
                if driver_id not in self._profiles:
                    self._profiles[driver_id] = DriverThresholdProfile(
                        driver_id=driver_id
                    )
                    logger.debug(
                        "ThresholdManager: Created new profile for driver_id=%d.", driver_id
                    )
        return self._profiles[driver_id]

    def export_profile(self, driver_id: int) -> Dict[str, Any]:
        """Exports a driver's threshold profile to a JSON-serialisable dict.

        Args:
            driver_id: Primary key of the DriverProfile.

        Returns:
            Dict[str, Any]: Serialised threshold profile.
        """
        profile = self._get_or_create_profile(driver_id)
        return profile.to_dict()

    def import_profile(self, profile_dict: Dict[str, Any]) -> None:
        """Imports a previously exported threshold profile.

        Args:
            profile_dict: Dict produced by ``export_profile()``.
        """
        driver_id = int(profile_dict["driver_id"])
        profile = self._get_or_create_profile(driver_id)
        with self._lock:
            profile.overrides = {
                k: float(v) for k, v in profile_dict.get("overrides", {}).items()
            }
            profile.cluster_thresholds = {
                k: float(v)
                for k, v in profile_dict.get("cluster_thresholds", {}).items()
            }
            profile.confidence = float(profile_dict.get("confidence", 0.0))
            profile.drift_corrections = int(profile_dict.get("drift_corrections", 0))
            if profile_dict.get("last_updated"):
                try:
                    profile.last_updated = datetime.fromisoformat(
                        profile_dict["last_updated"]
                    )
                except (ValueError, TypeError):
                    profile.last_updated = datetime.now(timezone.utc)

        logger.info(
            "ThresholdManager.import_profile: Imported profile for driver_id=%d.",
            driver_id,
        )

    def reset_driver_thresholds(self, driver_id: int) -> None:
        """Resets a driver's overrides to global defaults.

        Args:
            driver_id: Primary key of the DriverProfile.
        """
        with self._lock:
            if driver_id in self._profiles:
                self._profiles[driver_id].overrides.clear()
                self._profiles[driver_id].cluster_thresholds.clear()
                self._profiles[driver_id].last_updated = datetime.now(timezone.utc)
        logger.info(
            "ThresholdManager.reset_driver_thresholds: Reset overrides for driver_id=%d.",
            driver_id,
        )

    def to_json(self, driver_id: int) -> str:
        """Exports a driver's threshold profile to a JSON string.

        Args:
            driver_id: Primary key of the DriverProfile.

        Returns:
            str: JSON representation.
        """
        return json.dumps(self.export_profile(driver_id), indent=2)
