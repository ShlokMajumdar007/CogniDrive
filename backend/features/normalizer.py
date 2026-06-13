"""Feature normalisation for CogniDrive's ML inference pipeline.

This module provides the :class:`FeatureNormalizer` which applies
per-feature Z-score and Min-Max normalisation to the 21-dimensional
feature vector produced by :mod:`backend.features.feature_vector`.

Normalisation strategy:
    - **Z-score** (StandardScaler): Used for EAR, MAR, and head angles
      where the signal distribution is approximately Gaussian.
    - **Min-Max**: Used for PERCLOS, fatigue probability, risk score,
      gaze ratios, and binary features — values already in [0, 1].
    - **Passthrough**: Binary features (gaze_off_road, head_distracted)
      are already in {0, 1} and require no transformation.

The normaliser supports:
    1. **Driver-personalised stats**: A driver's own baseline mean/std is
       used when available (loaded from the Digital Twin).
    2. **Population stats fallback**: Pre-computed population statistics
       used when no driver baseline exists (new driver onboarding).
    3. **Incremental fit**: Online updating of running stats from a stream
       of feature vectors during a session.

Typical usage::

    normalizer = FeatureNormalizer()
    normalizer.load_population_stats()          # or load_driver_stats(...)
    normed = normalizer.normalize(feature_vector)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from backend.features.feature_vector import FEATURE_DIM, FEATURE_NAMES, FeatureVector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default population-level statistics (computed from a synthetic baseline)
# The values approximate a resting, attentive driver at 30 FPS.
# These are used when no driver-specific baseline is loaded.
# ---------------------------------------------------------------------------

#: Population mean for each of the 21 features.
POPULATION_MEAN: np.ndarray = np.array(
    [
        0.28,  # [0]  ear_left
        0.28,  # [1]  ear_right
        0.28,  # [2]  ear_mean
        0.05,  # [3]  mar
        0.08,  # [4]  perclos
        0.08,  # [5]  fatigue_probability
        0.50,  # [6]  blink_rate_norm  (15 bpm / 30 cap)
        0.00,  # [7]  yawn_rate_norm
        0.50,  # [8]  gaze_horizontal
        0.50,  # [9]  gaze_vertical
        0.05,  # [10] gaze_off_road (binary, low base rate)
        0.00,  # [11] head_pitch_norm
        0.00,  # [12] head_yaw_norm
        0.00,  # [13] head_roll_norm
        0.05,  # [14] head_distracted (binary)
        0.95,  # [15] attention_score_norm
        0.10,  # [16] stress_score_norm
        0.15,  # [17] cli_norm
        0.05,  # [18] risk_score
        0.02,  # [19] blink_consec_norm
        0.00,  # [20] yawn_consec_norm
    ],
    dtype=np.float64,
)

#: Population standard deviation for each of the 21 features.
POPULATION_STD: np.ndarray = np.array(
    [
        0.05,  # [0]  ear_left
        0.05,  # [1]  ear_right
        0.05,  # [2]  ear_mean
        0.08,  # [3]  mar
        0.10,  # [4]  perclos
        0.10,  # [5]  fatigue_probability
        0.20,  # [6]  blink_rate_norm
        0.05,  # [7]  yawn_rate_norm
        0.12,  # [8]  gaze_horizontal
        0.10,  # [9]  gaze_vertical
        0.22,  # [10] gaze_off_road
        0.08,  # [11] head_pitch_norm
        0.10,  # [12] head_yaw_norm
        0.05,  # [13] head_roll_norm
        0.22,  # [14] head_distracted
        0.08,  # [15] attention_score_norm
        0.12,  # [16] stress_score_norm
        0.12,  # [17] cli_norm
        0.08,  # [18] risk_score
        0.05,  # [19] blink_consec_norm
        0.03,  # [20] yawn_consec_norm
    ],
    dtype=np.float64,
)

assert POPULATION_MEAN.shape == (FEATURE_DIM,)
assert POPULATION_STD.shape == (FEATURE_DIM,)


# ---------------------------------------------------------------------------
# Normaliser statistics container
# ---------------------------------------------------------------------------


@dataclass
class NormalizerStats:
    """Per-feature normalisation statistics.

    Attributes:
        mean: Array of per-feature means.
        std: Array of per-feature standard deviations.
        source: Where the statistics came from ("population" or "driver").
        driver_id: Driver ID if driver-specific stats are loaded.
        sample_count: Number of samples used to compute the stats.
    """

    mean: np.ndarray = field(default_factory=lambda: POPULATION_MEAN.copy())
    std: np.ndarray = field(default_factory=lambda: POPULATION_STD.copy())
    source: str = "population"
    driver_id: Optional[int] = None
    sample_count: int = 0

    def to_dict(self) -> Dict:
        """Serialises stats to a JSON-compatible dictionary.

        Returns:
            Dict: Serialisable statistics mapping.
        """
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "source": self.source,
            "driver_id": self.driver_id,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "NormalizerStats":
        """Deserialises stats from a dictionary.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            NormalizerStats: Reconstructed statistics.
        """
        return cls(
            mean=np.array(data["mean"], dtype=np.float64),
            std=np.array(data["std"], dtype=np.float64),
            source=data.get("source", "population"),
            driver_id=data.get("driver_id"),
            sample_count=data.get("sample_count", 0),
        )


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------


class FeatureNormalizer:
    """Applies Z-score and Min-Max normalisation to 21-dimensional feature vectors.

    The normaliser is configured with a set of per-feature statistics
    (mean and std) which can be loaded from:
        - Population defaults (built-in).
        - A driver's personalised Digital Twin baseline.
        - A JSON file on disk.

    After normalisation, all features are clipped to [-3σ, +3σ] (Z-score)
    or [0, 1] (passthrough), ensuring no outlier values reach the model.

    Attributes:
        _stats: Active :class:`NormalizerStats` used for normalisation.
        _online_sum: Running sum for incremental mean estimation.
        _online_sq_sum: Running sum of squares for incremental std estimation.
        _online_count: Count of samples seen during incremental fit.
    """

    # Indices using Z-score normalisation (Gaussian-distributed features)
    _ZSCORE_INDICES: List[int] = [0, 1, 2, 3, 8, 9, 11, 12, 13]

    # Binary features — kept as-is, no normalisation
    _BINARY_INDICES: List[int] = [10, 14]

    # Remaining indices use Min-Max [0, 1] clipping (already bounded)
    _MINMAX_INDICES: List[int] = [4, 5, 6, 7, 15, 16, 17, 18, 19, 20]

    def __init__(self) -> None:
        """Initialises the normaliser with population-level statistics."""
        self._stats = NormalizerStats()
        self._online_sum = np.zeros(FEATURE_DIM, dtype=np.float64)
        self._online_sq_sum = np.zeros(FEATURE_DIM, dtype=np.float64)
        self._online_count: int = 0

    # ------------------------------------------------------------------
    # Stats loading
    # ------------------------------------------------------------------

    def load_population_stats(self) -> None:
        """Resets to built-in population-level statistics."""
        self._stats = NormalizerStats(
            mean=POPULATION_MEAN.copy(),
            std=POPULATION_STD.copy(),
            source="population",
        )
        logger.info("FeatureNormalizer loaded population stats.")

    def load_driver_stats(
        self,
        driver_id: int,
        mean: List[float],
        std: List[float],
    ) -> None:
        """Loads driver-personalised normalisation statistics.

        Args:
            driver_id: Driver's primary key.
            mean: Per-feature mean list of length 21.
            std: Per-feature std list of length 21.

        Raises:
            ValueError: If mean or std do not have length 21.
        """
        if len(mean) != FEATURE_DIM or len(std) != FEATURE_DIM:
            raise ValueError(
                f"Driver stats must have length {FEATURE_DIM}. "
                f"Got mean={len(mean)}, std={len(std)}."
            )
        self._stats = NormalizerStats(
            mean=np.array(mean, dtype=np.float64),
            std=np.array(std, dtype=np.float64),
            source="driver",
            driver_id=driver_id,
        )
        logger.info(
            "FeatureNormalizer loaded driver stats for driver_id=%d.", driver_id
        )

    def load_from_file(self, path: Path) -> None:
        """Loads normalisation statistics from a JSON file.

        Args:
            path: Path to a JSON file produced by :meth:`save_to_file`.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file content is invalid.
        """
        if not path.exists():
            raise FileNotFoundError(f"Normaliser stats file not found: {path}")
        with path.open("r") as f:
            data = json.load(f)
        self._stats = NormalizerStats.from_dict(data)
        logger.info(
            "FeatureNormalizer loaded stats from %s (source=%s).",
            path,
            self._stats.source,
        )

    def save_to_file(self, path: Path) -> None:
        """Saves current normalisation statistics to a JSON file.

        Args:
            path: Target file path (parent directories must exist).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self._stats.to_dict(), f, indent=2)
        logger.info("FeatureNormalizer stats saved to %s.", path)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize(self, feature_vector: FeatureVector) -> FeatureVector:
        """Normalises a feature vector using the active statistics.

        Z-score features are transformed as ``(x - mean) / std`` and
        clipped to [-3, 3]. Min-Max features are clipped to [0, 1].
        Binary features are passed through unchanged.

        Args:
            feature_vector: Input :class:`FeatureVector` with 21 raw features.

        Returns:
            FeatureVector: Normalised feature vector. ``is_valid`` reflects
                whether all output values are finite.
        """
        raw = np.array(feature_vector.features, dtype=np.float64)
        normed = raw.copy()

        # Z-score transform
        for i in self._ZSCORE_INDICES:
            std_i = self._stats.std[i]
            if std_i < 1e-8:
                std_i = 1.0
            z = (raw[i] - self._stats.mean[i]) / std_i
            normed[i] = float(np.clip(z, -3.0, 3.0))

        # Min-Max: already [0, 1], just clip for safety
        for i in self._MINMAX_INDICES:
            normed[i] = float(np.clip(raw[i], 0.0, 1.0))

        # Binary: passthrough
        for i in self._BINARY_INDICES:
            normed[i] = float(raw[i])

        is_valid = bool(np.all(np.isfinite(normed)))
        if not is_valid:
            normed = np.nan_to_num(normed, nan=0.0, posinf=3.0, neginf=-3.0)

        return FeatureVector(
            features=normed.tolist(),
            feature_names=feature_vector.feature_names,
            is_valid=is_valid,
        )

    def normalize_batch(
        self, feature_vectors: List[FeatureVector]
    ) -> List[FeatureVector]:
        """Normalises a list of feature vectors.

        Args:
            feature_vectors: List of raw :class:`FeatureVector` instances.

        Returns:
            List[FeatureVector]: Normalised feature vectors.
        """
        return [self.normalize(fv) for fv in feature_vectors]

    def normalize_numpy(self, raw: np.ndarray) -> np.ndarray:
        """Normalises a (N, 21) NumPy matrix directly.

        Useful for batch inference where feature vectors are already stacked.

        Args:
            raw: Float array of shape (N, 21) or (21,).

        Returns:
            np.ndarray: Normalised array of the same shape, dtype float32.
        """
        squeeze = raw.ndim == 1
        if squeeze:
            raw = raw.reshape(1, -1)

        normed = raw.astype(np.float64).copy()

        # Z-score
        for i in self._ZSCORE_INDICES:
            std_i = self._stats.std[i] if self._stats.std[i] > 1e-8 else 1.0
            normed[:, i] = np.clip(
                (normed[:, i] - self._stats.mean[i]) / std_i, -3.0, 3.0
            )

        # Min-Max clip
        for i in self._MINMAX_INDICES:
            normed[:, i] = np.clip(normed[:, i], 0.0, 1.0)

        normed = np.nan_to_num(normed, nan=0.0, posinf=3.0, neginf=-3.0)

        if squeeze:
            return normed.squeeze(0).astype(np.float32)
        return normed.astype(np.float32)

    # ------------------------------------------------------------------
    # Incremental fitting
    # ------------------------------------------------------------------

    def update_online(self, feature_vector: FeatureVector) -> None:
        """Updates running statistics with a new feature vector sample.

        Uses Welford's online algorithm for numerically stable incremental
        mean and variance estimation.

        Args:
            feature_vector: New raw feature vector sample to incorporate.
        """
        x = np.array(feature_vector.features, dtype=np.float64)
        self._online_count += 1
        self._online_sum += x
        self._online_sq_sum += x ** 2

    def commit_online_stats(self, min_samples: int = 100) -> bool:
        """Computes and applies the online-fitted statistics.

        Args:
            min_samples: Minimum samples required before committing.

        Returns:
            bool: True if stats were committed, False if insufficient data.
        """
        if self._online_count < min_samples:
            logger.warning(
                "Not enough samples to commit online stats (%d < %d).",
                self._online_count,
                min_samples,
            )
            return False

        n = self._online_count
        mean = self._online_sum / n
        variance = (self._online_sq_sum / n) - (mean ** 2)
        std = np.sqrt(np.maximum(variance, 1e-8))

        self._stats = NormalizerStats(
            mean=mean,
            std=std,
            source="online",
            sample_count=n,
        )
        logger.info(
            "FeatureNormalizer online stats committed from %d samples.", n
        )
        return True

    def reset_online(self) -> None:
        """Resets the online accumulator without affecting active stats."""
        self._online_sum = np.zeros(FEATURE_DIM, dtype=np.float64)
        self._online_sq_sum = np.zeros(FEATURE_DIM, dtype=np.float64)
        self._online_count = 0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def stats(self) -> NormalizerStats:
        """Returns the currently active normalisation statistics."""
        return self._stats

    @property
    def is_personalised(self) -> bool:
        """True if driver-specific statistics are loaded."""
        return self._stats.source == "driver"
