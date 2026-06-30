"""PersonalizationManager — Driver Behavioral Baseline Learning and Adaptation.

Implements online learning of per-driver baselines and adaptive threshold
updates for CogniDrive's Digital Twin. Each driver has a unique baseline
for key biometric signals; deviations from these baselines are what trigger
alerts and recommendations.

Core responsibilities:
    - Learn driver baselines from calibration session data.
    - Detect baseline drift using exponential moving statistics.
    - Compute personalization confidence as data accumulates.
    - Handle cold-start drivers with population-level priors.
    - Provide personalized scaling factors for recommendation intensity.

Baseline metrics tracked:
    EAR (eye aspect ratio), MAR (mouth aspect ratio), PERCLOS,
    blink rate, attention score, stress score, CLI, risk score,
    head pitch/yaw/roll, gaze horizontal/vertical.

Online learning algorithm:
    Welford's online algorithm for numerically stable mean and variance
    estimation without storing all historical values. Updates are O(1)
    per sample.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.features.feature_vector import FEATURE_NAMES

logger = logging.getLogger("CogniDrive.PersonalizationManager")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Population-level priors (healthy adult driver averages)
_POPULATION_PRIORS: Dict[str, float] = {
    "ear_left": 0.25,
    "ear_right": 0.25,
    "ear_mean": 0.25,
    "mar": 0.08,
    "perclos": 0.08,
    "fatigue_probability": 0.10,
    "blink_rate_norm": 0.50,   # 15 bpm / 30 cap
    "yawn_rate_norm": 0.10,
    "gaze_horizontal": 0.50,
    "gaze_vertical": 0.50,
    "gaze_off_road": 0.05,
    "head_pitch_norm": 0.0,
    "head_yaw_norm": 0.0,
    "head_roll_norm": 0.0,
    "head_distracted": 0.05,
    "attention_score_norm": 0.85,
    "stress_score_norm": 0.15,
    "cli_norm": 0.20,
    "risk_score": 0.10,
    "blink_consec_norm": 0.05,
    "yawn_consec_norm": 0.02,
}

# Minimum sessions required for personalization to be considered "confident"
MIN_CONFIDENCE_SESSIONS: int = 3
HIGH_CONFIDENCE_SESSIONS: int = 15

# Drift detection: flag if current window mean deviates > N sigma from baseline
DRIFT_SIGMA_THRESHOLD: float = 2.5

# EMA decay for drift detection
DRIFT_EMA_ALPHA: float = 0.10


# ---------------------------------------------------------------------------
# Welford online statistics
# ---------------------------------------------------------------------------


@dataclass
class OnlineStats:
    """Numerically stable online mean and variance via Welford's algorithm.

    Attributes:
        count: Number of samples seen.
        mean: Running mean.
        M2: Running sum of squared deviations from the mean (for variance).
    """
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def update(self, value: float) -> None:
        """Updates the running statistics with a new observation.

        Args:
            value: New scalar observation.
        """
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        """Sample variance (Bessel-corrected).

        Returns:
            float: Sample variance, or 0.0 if fewer than 2 samples.
        """
        if self.count < 2:
            return 0.0
        return self.M2 / (self.count - 1)

    @property
    def std(self) -> float:
        """Sample standard deviation.

        Returns:
            float: Std dev, or 0.0 if fewer than 2 samples.
        """
        return math.sqrt(max(0.0, self.variance))

    def to_dict(self) -> Dict[str, float]:
        """Serialises statistics to a plain dictionary.

        Returns:
            Dict[str, float]: Keys: count, mean, std.
        """
        return {
            "count": float(self.count),
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
        }


# ---------------------------------------------------------------------------
# Personalization profile dataclass
# ---------------------------------------------------------------------------


@dataclass
class DriverPersonalizationProfile:
    """In-memory representation of a driver's personalized baseline statistics.

    Attributes:
        driver_id: Primary key of the DriverProfile.
        feature_stats: Per-feature Welford statistics keyed by feature name.
        sessions_count: Total calibration sessions processed.
        last_updated: UTC timestamp of the last baseline update.
        confidence: Personalization confidence in [0.0, 1.0].
        drift_ema: EMA of recent feature drift signals per feature.
        is_cold_start: True until MIN_CONFIDENCE_SESSIONS sessions seen.
    """
    driver_id: int
    feature_stats: Dict[str, OnlineStats] = field(default_factory=dict)
    sessions_count: int = 0
    last_updated: Optional[datetime] = None
    confidence: float = 0.0
    drift_ema: Dict[str, float] = field(default_factory=dict)
    is_cold_start: bool = True

    def get_baseline(self, feature_name: str) -> float:
        """Returns the learned baseline mean for a feature.

        Falls back to the population prior if the feature has < 2 samples.

        Args:
            feature_name: Feature name from FEATURE_NAMES.

        Returns:
            float: Baseline mean value.
        """
        stats = self.feature_stats.get(feature_name)
        if stats is None or stats.count < 2:
            return _POPULATION_PRIORS.get(feature_name, 0.0)
        return stats.mean

    def get_std(self, feature_name: str) -> float:
        """Returns the standard deviation for a feature.

        Args:
            feature_name: Feature name.

        Returns:
            float: Std dev or a default of 0.05 (small prior uncertainty).
        """
        stats = self.feature_stats.get(feature_name)
        if stats is None or stats.count < 2:
            return 0.05
        return max(stats.std, 1e-6)


# ---------------------------------------------------------------------------
# PersonalizationManager
# ---------------------------------------------------------------------------


class PersonalizationManager:
    """Online driver baseline learning and adaptive threshold engine.

    Thread-safe. Maintains one ``DriverPersonalizationProfile`` per driver
    in an in-memory registry. Profiles are populated from session feature
    matrices and can be exported/imported to/from plain dictionaries for
    persistence in the database ``extra_data`` JSON fields.

    Attributes:
        _profiles: Registry mapping driver_id → DriverPersonalizationProfile.
        _lock: Guards concurrent profile reads/writes.
    """

    def __init__(self) -> None:
        """Initialises the manager with an empty profile registry."""
        self._profiles: Dict[int, DriverPersonalizationProfile] = {}
        self._lock: threading.Lock = threading.Lock()
        logger.info("PersonalizationManager: Initialized.")

    # ------------------------------------------------------------------
    # Baseline learning
    # ------------------------------------------------------------------

    def update_baseline(
        self,
        driver_id: int,
        feature_matrix: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> DriverPersonalizationProfile:
        """Updates a driver's baseline statistics from a session feature matrix.

        For each of the 21 features, computes the session mean and feeds it
        to the Welford online estimator. This accumulates across sessions
        without storing the full history.

        Args:
            driver_id: Primary key of the DriverProfile.
            feature_matrix: ``(N, 21)`` float32 feature matrix for this session.
            feature_names: Feature name list (defaults to ``FEATURE_NAMES``).

        Returns:
            DriverPersonalizationProfile: Updated profile.
        """
        names = feature_names or FEATURE_NAMES

        with self._lock:
            profile = self._get_or_create_profile(driver_id)

            # Compute per-feature session mean and update Welford stats
            session_means = feature_matrix.mean(axis=0).tolist()
            for idx, fname in enumerate(names):
                if idx >= len(session_means):
                    break
                if fname not in profile.feature_stats:
                    profile.feature_stats[fname] = OnlineStats()
                profile.feature_stats[fname].update(session_means[idx])

            # Update drift EMA
            self._update_drift_ema(profile, session_means, names)

            profile.sessions_count += 1
            profile.last_updated = datetime.now(timezone.utc)
            profile.confidence = self._compute_confidence(profile.sessions_count)
            profile.is_cold_start = profile.sessions_count < MIN_CONFIDENCE_SESSIONS

        logger.debug(
            "PersonalizationManager.update_baseline: driver_id=%d, "
            "sessions=%d, confidence=%.3f, cold_start=%s.",
            driver_id,
            profile.sessions_count,
            profile.confidence,
            profile.is_cold_start,
        )
        return profile

    def _update_drift_ema(
        self,
        profile: DriverPersonalizationProfile,
        session_means: List[float],
        feature_names: List[str],
    ) -> None:
        """Updates the drift EMA signal for each feature.

        Computes Z-score deviation of this session's mean from the running
        baseline and applies EMA smoothing. Values > DRIFT_SIGMA_THRESHOLD
        indicate potential behavioral drift.

        Args:
            profile: Driver's personalization profile.
            session_means: Per-feature session means.
            feature_names: Feature names corresponding to session_means.
        """
        for idx, fname in enumerate(feature_names):
            if idx >= len(session_means):
                break
            baseline = profile.get_baseline(fname)
            std = profile.get_std(fname)
            z_score = abs(session_means[idx] - baseline) / max(std, 1e-6)
            prev_ema = profile.drift_ema.get(fname, 0.0)
            profile.drift_ema[fname] = (
                DRIFT_EMA_ALPHA * z_score + (1.0 - DRIFT_EMA_ALPHA) * prev_ema
            )

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_drift(
        self,
        driver_id: int,
        current_feature_vector: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Detects if the current feature vector represents behavioral drift.

        Computes Z-scores for all 21 features relative to the driver's
        learned baseline. Features with |Z| > DRIFT_SIGMA_THRESHOLD are
        flagged as drifted.

        Args:
            driver_id: Primary key of the DriverProfile.
            current_feature_vector: Current 21-D feature vector.
            feature_names: Feature names list.

        Returns:
            Dict[str, Any]: Drift report with keys:
                - ``drift_detected`` (bool)
                - ``drifted_features`` (List[str])
                - ``z_scores`` (Dict[str, float])
                - ``max_drift_feature`` (str)
                - ``max_z_score`` (float)
        """
        names = feature_names or FEATURE_NAMES
        profile = self.get_profile(driver_id)

        z_scores: Dict[str, float] = {}
        drifted: List[str] = []

        for idx, fname in enumerate(names):
            if idx >= len(current_feature_vector):
                break
            baseline = profile.get_baseline(fname)
            std = profile.get_std(fname)
            z = abs(float(current_feature_vector[idx]) - baseline) / max(std, 1e-6)
            z_scores[fname] = round(z, 3)
            if z > DRIFT_SIGMA_THRESHOLD and not profile.is_cold_start:
                drifted.append(fname)

        max_feature = max(z_scores, key=z_scores.get) if z_scores else ""
        max_z = z_scores.get(max_feature, 0.0)

        return {
            "drift_detected": bool(drifted),
            "drifted_features": drifted,
            "z_scores": z_scores,
            "max_drift_feature": max_feature,
            "max_z_score": round(max_z, 3),
        }

    # ------------------------------------------------------------------
    # Adaptive threshold computation
    # ------------------------------------------------------------------

    def get_personalized_threshold(
        self,
        driver_id: int,
        metric_name: str,
        global_threshold: float,
        sigma_multiplier: float = 2.0,
    ) -> float:
        """Computes a personalized alert threshold for a specific metric.

        Formula::

            threshold = baseline + sigma_multiplier * std

        For metrics where higher = worse (e.g. risk_score, cli_norm):
            threshold = global_threshold adjusted by driver deviation.

        Falls back to global_threshold if the driver is cold-start.

        Args:
            driver_id: Primary key of the DriverProfile.
            metric_name: Feature name to compute threshold for.
            global_threshold: Population-level default threshold.
            sigma_multiplier: How many sigma above baseline triggers an alert.

        Returns:
            float: Personalized threshold value.
        """
        profile = self.get_profile(driver_id)

        if profile.is_cold_start:
            return global_threshold

        baseline = profile.get_baseline(metric_name)
        std = profile.get_std(metric_name)
        confidence = profile.confidence

        # Weighted interpolation between global and personalized threshold
        personalized = baseline + sigma_multiplier * std
        blended = confidence * personalized + (1.0 - confidence) * global_threshold

        return float(np.clip(blended, 0.0, 1.0))

    def get_personalization_confidence(self, driver_id: int) -> float:
        """Returns the personalization confidence for a driver.

        Args:
            driver_id: Primary key of the DriverProfile.

        Returns:
            float: Confidence in [0.0, 1.0].
        """
        profile = self.get_profile(driver_id)
        return profile.confidence

    @staticmethod
    def _compute_confidence(sessions_count: int) -> float:
        """Computes personalization confidence from session count.

        Uses a sigmoid-like curve: 0% at 0 sessions, ~63% at
        MIN_CONFIDENCE_SESSIONS, 95%+ at HIGH_CONFIDENCE_SESSIONS.

        Args:
            sessions_count: Total sessions processed.

        Returns:
            float: Confidence in [0.0, 1.0].
        """
        if sessions_count == 0:
            return 0.0
        x = sessions_count / HIGH_CONFIDENCE_SESSIONS
        return float(np.clip(1.0 - np.exp(-2.5 * x), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Personalization scaling
    # ------------------------------------------------------------------

    def get_recommendation_intensity(
        self,
        driver_id: int,
        metric_name: str,
        current_value: float,
    ) -> float:
        """Computes how much to scale recommendation urgency based on driver baseline.

        A driver who always drives with slightly elevated stress should not
        receive HIGH urgency for their normal stress level. This method returns
        a 0–1 multiplier based on how far current_value deviates from baseline.

        Args:
            driver_id: Primary key of the DriverProfile.
            metric_name: Feature name of the triggering metric.
            current_value: Current measured value.

        Returns:
            float: Intensity multiplier in [0.0, 1.0].
        """
        profile = self.get_profile(driver_id)
        baseline = profile.get_baseline(metric_name)
        std = max(profile.get_std(metric_name), 1e-6)
        z = abs(current_value - baseline) / std
        # Clamp: 0 sigma → 0.0 intensity, 3 sigma → 1.0 intensity
        return float(np.clip(z / 3.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def get_profile(self, driver_id: int) -> DriverPersonalizationProfile:
        """Retrieves or creates a driver's personalization profile.

        Args:
            driver_id: Primary key of the DriverProfile.

        Returns:
            DriverPersonalizationProfile: The driver's profile.
        """
        with self._lock:
            return self._get_or_create_profile(driver_id)

    def _get_or_create_profile(
        self, driver_id: int
    ) -> DriverPersonalizationProfile:
        """Internal: retrieves or creates a profile. Must be called with lock held."""
        if driver_id not in self._profiles:
            self._profiles[driver_id] = DriverPersonalizationProfile(
                driver_id=driver_id
            )
            logger.debug(
                "PersonalizationManager: Created new profile for driver_id=%d.",
                driver_id,
            )
        return self._profiles[driver_id]

    def export_profile(self, driver_id: int) -> Dict[str, Any]:
        """Exports a driver's profile to a JSON-serialisable dictionary.

        Args:
            driver_id: Primary key of the DriverProfile.

        Returns:
            Dict[str, Any]: Serialised profile dictionary.
        """
        profile = self.get_profile(driver_id)
        return {
            "driver_id": profile.driver_id,
            "sessions_count": profile.sessions_count,
            "confidence": profile.confidence,
            "is_cold_start": profile.is_cold_start,
            "last_updated": (
                profile.last_updated.isoformat() if profile.last_updated else None
            ),
            "feature_stats": {
                k: v.to_dict() for k, v in profile.feature_stats.items()
            },
            "drift_ema": {k: round(v, 4) for k, v in profile.drift_ema.items()},
        }

    def import_profile(self, profile_dict: Dict[str, Any]) -> None:
        """Imports a previously exported profile dictionary.

        Args:
            profile_dict: Dict produced by ``export_profile()``.
        """
        driver_id = int(profile_dict["driver_id"])
        with self._lock:
            profile = self._get_or_create_profile(driver_id)
            profile.sessions_count = int(profile_dict.get("sessions_count", 0))
            profile.confidence = float(profile_dict.get("confidence", 0.0))
            profile.is_cold_start = bool(profile_dict.get("is_cold_start", True))

            raw_stats = profile_dict.get("feature_stats", {})
            for fname, stats_dict in raw_stats.items():
                stats = OnlineStats()
                stats.count = int(stats_dict.get("count", 0))
                stats.mean = float(stats_dict.get("mean", 0.0))
                profile.feature_stats[fname] = stats

            profile.drift_ema = {
                k: float(v)
                for k, v in profile_dict.get("drift_ema", {}).items()
            }

            if profile_dict.get("last_updated"):
                try:
                    profile.last_updated = datetime.fromisoformat(
                        profile_dict["last_updated"]
                    )
                except (ValueError, TypeError):
                    profile.last_updated = datetime.now(timezone.utc)

        logger.info(
            "PersonalizationManager.import_profile: Imported profile for driver_id=%d "
            "(sessions=%d, confidence=%.3f).",
            driver_id,
            profile.sessions_count,
            profile.confidence,
        )

    def get_baseline_summary(self, driver_id: int) -> Dict[str, float]:
        """Returns the current baseline means for all features.

        Args:
            driver_id: Primary key of the DriverProfile.

        Returns:
            Dict[str, float]: Feature name → baseline mean.
        """
        profile = self.get_profile(driver_id)
        return {
            fname: profile.get_baseline(fname)
            for fname in FEATURE_NAMES
        }

    def clear_profile(self, driver_id: int) -> None:
        """Clears a driver's in-memory profile (does not affect database).

        Args:
            driver_id: Primary key of the DriverProfile.
        """
        with self._lock:
            if driver_id in self._profiles:
                del self._profiles[driver_id]
                logger.info(
                    "PersonalizationManager.clear_profile: Cleared profile for driver_id=%d.",
                    driver_id,
                )
