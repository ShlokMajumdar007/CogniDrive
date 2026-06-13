"""Feature vector construction for CogniDrive's ML inference pipeline.

This module assembles the canonical 21-dimensional feature vector consumed
by all CogniDrive ML models (cognitive load, accident risk, and embeddings).
Every feature is pre-normalised to [0, 1] or a known bounded range.

Feature vector layout (21 features, zero-indexed)::

    [0]  EAR left                — Eye Aspect Ratio, left eye [0, ~0.40]
    [1]  EAR right               — Eye Aspect Ratio, right eye [0, ~0.40]
    [2]  EAR mean                — Mean of left and right EAR [0, ~0.40]
    [3]  MAR                     — Mouth Aspect Ratio [0, ~1.0]
    [4]  PERCLOS                 — Proportion of window closed [0, 1]
    [5]  Fatigue probability     — PERCLOS-derived [0, 1]
    [6]  Blink rate (norm.)      — Blinks per minute / 30.0 [0, 1]
    [7]  Yawn count (norm.)      — Yawns per hour / 10.0 [0, 1]
    [8]  Gaze horizontal ratio   — Iris offset from centre [0, 1]
    [9]  Gaze vertical ratio     — Iris offset from centre [0, 1]
    [10] Gaze off-road (binary)  — 1.0 if off-road else 0.0
    [11] Head pitch (norm.)      — Pitch / 90° clamped to [-1, 1]
    [12] Head yaw (norm.)        — Yaw / 90° clamped to [-1, 1]
    [13] Head roll (norm.)       — Roll / 90° clamped to [-1, 1]
    [14] Head distracted (bin.)  — 1.0 if head pose distracted else 0.0
    [15] Attention score (norm.) — From previous prediction / 100.0 [0, 1]
    [16] Stress score (norm.)    — From previous prediction / 100.0 [0, 1]
    [17] CLI (norm.)             — Cognitive Load Index / 100.0 [0, 1]
    [18] Risk score              — Previous risk score [0, 1]
    [19] Blink consec (norm.)    — Consecutive closed frames / 90 [0, 1]
    [20] Yawn consec (norm.)     — Consecutive yawn frames / 45 [0, 1]

Typical usage::

    from backend.features.feature_vector import FeatureVectorBuilder, RawSignals
    builder = FeatureVectorBuilder()
    signals = RawSignals(ear_left=0.22, ear_right=0.21, ...)
    fv = builder.build(signals)
    array = fv.to_numpy()  # shape (21,), dtype float32
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature vector dimension
# ---------------------------------------------------------------------------

FEATURE_DIM: int = 21

# Normalisation caps for blink and yawn rates
_BLINK_RATE_CAP: float = 30.0   # blinks per minute
_YAWN_RATE_CAP: float = 10.0    # yawns per hour
_CONSEC_BLINK_CAP: float = 90.0  # frames ≈ 3 s at 30 FPS
_CONSEC_YAWN_CAP: float = 45.0   # frames ≈ 1.5 s at 30 FPS
_ANGLE_CAP: float = 90.0         # degrees


# ---------------------------------------------------------------------------
# Raw signals dataclass
# ---------------------------------------------------------------------------


@dataclass
class RawSignals:
    """Container for all per-frame raw biometric measurements.

    All fields are optional — missing values default to safe neutral values
    so the feature vector can still be computed from partial data.

    Attributes:
        ear_left: Left eye EAR value.
        ear_right: Right eye EAR value.
        mar: Mouth Aspect Ratio value.
        perclos: PERCLOS proportion [0, 1].
        fatigue_probability: PERCLOS-derived fatigue probability [0, 1].
        blink_rate_bpm: Blinks per minute.
        yawn_rate_per_hour: Yawns per hour.
        gaze_horizontal: Normalised horizontal gaze ratio [0, 1].
        gaze_vertical: Normalised vertical gaze ratio [0, 1].
        gaze_off_road: True if gaze is classified as off-road.
        head_pitch: Head pitch angle in degrees.
        head_yaw: Head yaw angle in degrees.
        head_roll: Head roll angle in degrees.
        head_distracted: True if head pose indicates distraction.
        attention_score: Attention score from previous inference [0, 100].
        stress_score: Stress score from previous inference [0, 100].
        cli: Cognitive Load Index from previous inference [0, 100].
        risk_score: Risk score from previous inference [0, 1].
        blink_consec_frames: Current consecutive closed-eye frame count.
        yawn_consec_frames: Current consecutive yawning frame count.
    """

    ear_left: float = 0.25
    ear_right: float = 0.25
    mar: float = 0.0
    perclos: float = 0.0
    fatigue_probability: float = 0.0
    blink_rate_bpm: float = 15.0
    yawn_rate_per_hour: float = 0.0
    gaze_horizontal: float = 0.5
    gaze_vertical: float = 0.5
    gaze_off_road: bool = False
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    head_distracted: bool = False
    attention_score: float = 100.0
    stress_score: float = 0.0
    cli: float = 0.0
    risk_score: float = 0.0
    blink_consec_frames: int = 0
    yawn_consec_frames: int = 0


# ---------------------------------------------------------------------------
# Feature vector result
# ---------------------------------------------------------------------------


@dataclass
class FeatureVector:
    """Assembled and normalised 21-dimensional feature vector.

    Attributes:
        features: Ordered list of 21 normalised float values.
        feature_names: Corresponding human-readable feature names.
        is_valid: True when all features are finite.
    """

    features: List[float] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    is_valid: bool = False

    def to_numpy(self) -> np.ndarray:
        """Converts to a float32 NumPy array of shape (21,).

        Returns:
            np.ndarray: Feature array for model inference.
        """
        return np.array(self.features, dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        """Returns a {name: value} mapping for debugging and XAI.

        Returns:
            Dict[str, float]: Feature name to value mapping.
        """
        return dict(zip(self.feature_names, self.features))

    def to_list(self) -> List[float]:
        """Returns raw feature values as a Python list.

        Returns:
            List[float]: 21-element feature list.
        """
        return self.features

    def __len__(self) -> int:
        """Returns the number of features in this vector."""
        return len(self.features)


# ---------------------------------------------------------------------------
# Feature names registry
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "ear_left",
    "ear_right",
    "ear_mean",
    "mar",
    "perclos",
    "fatigue_probability",
    "blink_rate_norm",
    "yawn_rate_norm",
    "gaze_horizontal",
    "gaze_vertical",
    "gaze_off_road",
    "head_pitch_norm",
    "head_yaw_norm",
    "head_roll_norm",
    "head_distracted",
    "attention_score_norm",
    "stress_score_norm",
    "cli_norm",
    "risk_score",
    "blink_consec_norm",
    "yawn_consec_norm",
]

assert len(FEATURE_NAMES) == FEATURE_DIM, (
    f"FEATURE_NAMES length {len(FEATURE_NAMES)} must equal FEATURE_DIM {FEATURE_DIM}"
)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    """Clamps value to [low, high]."""
    return max(low, min(high, value))


def _norm_angle(angle_deg: float) -> float:
    """Normalises a head angle from degrees to [-1, 1].

    Args:
        angle_deg: Angle in degrees.

    Returns:
        float: Normalised angle in [-1, 1].
    """
    return _clamp(angle_deg / _ANGLE_CAP, -1.0, 1.0)


class FeatureVectorBuilder:
    """Stateless builder that assembles a 21-dimensional feature vector.

    Converts :class:`RawSignals` into a :class:`FeatureVector` with all
    features clipped to valid ranges and checked for finiteness.

    Typical usage::

        builder = FeatureVectorBuilder()
        fv = builder.build(signals)
    """

    def build(self, signals: RawSignals) -> FeatureVector:
        """Assembles the normalised feature vector from raw biometric signals.

        Args:
            signals: :class:`RawSignals` containing all per-frame measurements.

        Returns:
            FeatureVector: 21-feature vector with validity flag.
        """
        ear_mean = (signals.ear_left + signals.ear_right) / 2.0

        features: List[float] = [
            _clamp(signals.ear_left, 0.0, 0.5),                              # [0]
            _clamp(signals.ear_right, 0.0, 0.5),                             # [1]
            _clamp(ear_mean, 0.0, 0.5),                                      # [2]
            _clamp(signals.mar, 0.0, 1.5),                                   # [3]
            _clamp(signals.perclos, 0.0, 1.0),                               # [4]
            _clamp(signals.fatigue_probability, 0.0, 1.0),                   # [5]
            _clamp(signals.blink_rate_bpm / _BLINK_RATE_CAP, 0.0, 1.0),     # [6]
            _clamp(signals.yawn_rate_per_hour / _YAWN_RATE_CAP, 0.0, 1.0),  # [7]
            _clamp(signals.gaze_horizontal, 0.0, 1.0),                       # [8]
            _clamp(signals.gaze_vertical, 0.0, 1.0),                         # [9]
            1.0 if signals.gaze_off_road else 0.0,                           # [10]
            _norm_angle(signals.head_pitch),                                  # [11]
            _norm_angle(signals.head_yaw),                                    # [12]
            _norm_angle(signals.head_roll),                                   # [13]
            1.0 if signals.head_distracted else 0.0,                         # [14]
            _clamp(signals.attention_score / 100.0, 0.0, 1.0),               # [15]
            _clamp(signals.stress_score / 100.0, 0.0, 1.0),                  # [16]
            _clamp(signals.cli / 100.0, 0.0, 1.0),                           # [17]
            _clamp(signals.risk_score, 0.0, 1.0),                            # [18]
            _clamp(signals.blink_consec_frames / _CONSEC_BLINK_CAP, 0.0, 1.0),  # [19]
            _clamp(signals.yawn_consec_frames / _CONSEC_YAWN_CAP, 0.0, 1.0),    # [20]
        ]

        is_valid = all(np.isfinite(f) for f in features)

        if not is_valid:
            invalid = [
                (FEATURE_NAMES[i], features[i])
                for i in range(len(features))
                if not np.isfinite(features[i])
            ]
            logger.warning(
                "Feature vector contains non-finite values: %s", invalid
            )
            # Replace NaN/Inf with 0.0 to prevent model crashes
            features = [0.0 if not np.isfinite(f) else f for f in features]

        return FeatureVector(
            features=features,
            feature_names=FEATURE_NAMES,
            is_valid=is_valid,
        )

    def build_batch(
        self, signals_list: List[RawSignals]
    ) -> List[FeatureVector]:
        """Builds multiple feature vectors from a list of raw signal snapshots.

        Args:
            signals_list: List of :class:`RawSignals` instances.

        Returns:
            List[FeatureVector]: Corresponding feature vectors.
        """
        return [self.build(s) for s in signals_list]

    def to_numpy_matrix(
        self, signals_list: List[RawSignals]
    ) -> np.ndarray:
        """Builds a (N, 21) NumPy matrix from a list of raw signals.

        Useful for batch inference with XGBoost / LightGBM which expect
        a 2-D feature matrix.

        Args:
            signals_list: List of :class:`RawSignals` instances.

        Returns:
            np.ndarray: Matrix of shape (N, 21), dtype float32.
        """
        fvs = self.build_batch(signals_list)
        return np.stack([fv.to_numpy() for fv in fvs], axis=0)
