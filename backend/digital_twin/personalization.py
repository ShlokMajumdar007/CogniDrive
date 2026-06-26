"""Driver Digital Twin — personalization engine.

This module answers the question that makes CogniDrive more than a generic
drowsiness detector: *"is this specific value normal for THIS driver?"*

It learns a per-driver running baseline (mean + variance) over the same raw
biometric signals defined in :class:`backend.features.feature_vector.RawSignals`
— EAR, MAR, PERCLOS, blink rate, gaze, head pose — and derives **adaptive,
personalised thresholds** that replace the global defaults in
:class:`backend.app.constants.Thresholds`.

Why raw-signal space, not the 21-D model feature space
---------------------------------------------------------
:mod:`backend.ml.digital_twin.driver_embedding` personalises the *128-D
Digital Twin embedding* (used for similarity/clustering/recommendation).
This module personalises the *raw, human-interpretable signals* that the
global :class:`backend.app.constants.Thresholds` already operate on (EAR,
MAR, PERCLOS, blink rate, gaze, head pose). Operating in raw-signal space
means the learned thresholds remain directly substitutable wherever
``Thresholds.DEFAULT_EAR_THRESHOLD`` etc. are currently used by the alert /
state-classification logic, with zero changes needed downstream.

Mathematical foundation
------------------------
Each tracked signal maintains a Welford running mean/variance (see
:class:`backend.ml.digital_twin.driver_embedding.WelfordState`, reused here
for numerical consistency and to avoid duplicating the algorithm).
A new observation ``x`` is converted to a personalised z-score:

.. math::

    z = \\frac{x - \\mu_{\\text{driver}}}{\\sigma_{\\text{driver}} + \\epsilon}

Two drivers presenting the *same raw* EAR value can therefore receive
*different* personalised risk contributions, because the same raw value
maps to a different z-score relative to each driver's own learned
distribution — this is the literal mechanism behind CogniDrive's "same
sensor reading, different driver, different prediction" claim.

Personalised thresholds are then derived as ``mu ± k * sigma`` (a
Chebyshev-style adaptive band), blended with the global default via a
confidence-weighted interpolation so that cold-start drivers smoothly fall
back to population defaults and seasoned drivers fully transition to their
own learned baseline.

All computation is local NumPy / Python; nothing in this module performs
network I/O.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from backend.ml.digital_twin.driver_embedding import WelfordState
    from backend.app.constants import Thresholds
except ImportError:
    from ml.digital_twin.driver_embedding import WelfordState  # type: ignore[no-redef]
    from app.constants import Thresholds  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-6

#: Number of sessions at which a driver's baseline is considered "fully
#: calibrated" (confidence saturates to 1.0). Mirrors the embedding
#: manager's _MIN_SESSIONS_FOR_CONFIDENCE for consistency across the twin.
MIN_SESSIONS_FOR_FULL_CONFIDENCE: int = 10

#: Minimum number of raw observations (not sessions — individual frame-level
#: samples) required before a signal's learned mean/std are trusted at all.
#: Below this, the signal falls back 100% to the global default threshold
#: regardless of session count, since within-session sample count can be
#: tiny for short/aborted sessions.
MIN_SAMPLES_FOR_BASELINE: int = 50

#: Default z-score multiplier (k) used to derive a personalised threshold
#: band from the learned mean/std: threshold = mu ± k * sigma.
DEFAULT_Z_MULTIPLIER: float = 1.5


# ---------------------------------------------------------------------------
# Tracked signal registry
# ---------------------------------------------------------------------------


class TrackedSignal(str, Enum):
    """Raw biometric signals for which CogniDrive learns a personalised baseline.

    Names correspond directly to :class:`backend.features.feature_vector.RawSignals`
    field names so callers can pass that dataclass's attributes straight
    through without renaming.
    """

    EAR_MEAN = "ear_mean"
    MAR = "mar"
    PERCLOS = "perclos"
    BLINK_RATE_BPM = "blink_rate_bpm"
    GAZE_HORIZONTAL = "gaze_horizontal"
    GAZE_VERTICAL = "gaze_vertical"
    HEAD_PITCH = "head_pitch"
    HEAD_YAW = "head_yaw"
    HEAD_ROLL = "head_roll"


#: Whether a HIGHER personalised threshold than the global default would be
#: more permissive (True, e.g. MAR — "this driver's mouth is just shaped
#: this way") or whether the signal's risk direction is "below threshold is
#: bad" (False, e.g. EAR — low EAR means closed eyes). This determines the
#: sign of the z-multiplier when deriving the adaptive threshold band.
_LOWER_IS_RISKIER: Dict[TrackedSignal, bool] = {
    TrackedSignal.EAR_MEAN: True,           # low EAR = eyes closing = risky
    TrackedSignal.MAR: False,               # high MAR = yawning = risky
    TrackedSignal.PERCLOS: False,           # high PERCLOS = risky
    TrackedSignal.BLINK_RATE_BPM: None,     # type: ignore[dict-item]  # two-sided (too low OR too high abnormal)
    TrackedSignal.GAZE_HORIZONTAL: None,    # type: ignore[dict-item]  # two-sided (deviation from center)
    TrackedSignal.GAZE_VERTICAL: None,      # type: ignore[dict-item]  # two-sided
    TrackedSignal.HEAD_PITCH: None,         # type: ignore[dict-item]  # two-sided (abs deviation matters)
    TrackedSignal.HEAD_YAW: None,           # type: ignore[dict-item]  # two-sided
    TrackedSignal.HEAD_ROLL: None,          # type: ignore[dict-item]  # two-sided
}

#: Maps each tracked signal to its corresponding global default in
#: backend.app.constants.Thresholds, used as the cold-start fallback and as
#: the blend target for low-confidence drivers. For two-sided signals
#: without a single scalar default, a sentinel float-based equivalent is
#: derived inline (see DriverBaselineTracker._global_default).
def _global_defaults() -> Dict[TrackedSignal, float]:
    """Builds the signal -> global-default lookup table from app constants.

    Centralised in a function (rather than a module-level dict) so it is
    evaluated lazily, avoiding import-order issues with
    :mod:`backend.app.constants`.

    Returns:
        Dict[TrackedSignal, float]: Global default threshold per signal.
    """
    return {
        TrackedSignal.EAR_MEAN: Thresholds.DEFAULT_EAR_THRESHOLD,
        TrackedSignal.MAR: Thresholds.DEFAULT_MAR_THRESHOLD,
        TrackedSignal.PERCLOS: Thresholds.DEFAULT_PERCLOS_THRESHOLD,
        TrackedSignal.BLINK_RATE_BPM: (
            Thresholds.MIN_NORMAL_BLINK_RATE + Thresholds.MAX_NORMAL_BLINK_RATE
        ) / 2.0,
        TrackedSignal.GAZE_HORIZONTAL: Thresholds.DEFAULT_GAZE_THRESHOLD,
        TrackedSignal.GAZE_VERTICAL: Thresholds.DEFAULT_GAZE_THRESHOLD,
        TrackedSignal.HEAD_PITCH: Thresholds.MAX_PITCH_THRESHOLD,
        TrackedSignal.HEAD_YAW: Thresholds.MAX_YAW_THRESHOLD,
        TrackedSignal.HEAD_ROLL: Thresholds.MAX_ROLL_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Per-driver baseline state
# ---------------------------------------------------------------------------


@dataclass
class DriverBaselineState:
    """Per-driver learned baseline across all tracked signals.

    Attributes:
        driver_id: Primary key of the driver profile.
        trackers: One :class:`WelfordState` per :class:`TrackedSignal`,
            each a 1-D (scalar) tracker (shape ``(1,)``).
        n_sessions: Number of distinct driving sessions contributing to
            this baseline (incremented once per :meth:`PersonalizationEngine.end_session`,
            not per-frame).
        last_updated: UTC timestamp of the most recent observation.
    """

    driver_id: int
    trackers: Dict[TrackedSignal, WelfordState] = field(default_factory=dict)
    n_sessions: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Initialises a fresh scalar Welford tracker for every tracked signal."""
        if not self.trackers:
            self.trackers = {
                signal: WelfordState(
                    mean=np.zeros(1, dtype=np.float64),
                    m2=np.zeros(1, dtype=np.float64),
                )
                for signal in TrackedSignal
            }

    @classmethod
    def new(cls, driver_id: int) -> "DriverBaselineState":
        """Creates a fresh, empty baseline state for a new driver.

        Args:
            driver_id: Primary key of the driver profile.

        Returns:
            DriverBaselineState: Cold-start state with all trackers empty.
        """
        return cls(driver_id=driver_id)

    @property
    def is_cold_start(self) -> bool:
        """True if no sessions have been incorporated into this baseline yet."""
        return self.n_sessions == 0


# ---------------------------------------------------------------------------
# Adaptive threshold result
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveThreshold:
    """A single personalised threshold, blended between the driver's learned
    baseline and the global population default.

    Attributes:
        signal: Which tracked signal this threshold applies to.
        value: The final, blended threshold value to use in place of the
            corresponding ``Thresholds.DEFAULT_*`` constant.
        driver_mean: The driver's learned mean for this signal (``None`` if
            insufficient samples).
        driver_std: The driver's learned standard deviation for this signal
            (``None`` if insufficient samples).
        global_default: The population-level default this threshold falls
            back to / blends with.
        confidence: Blend weight in ``[0, 1]`` given to the driver-specific
            baseline (``1.0`` = fully personalised, ``0.0`` = pure global
            default).
        sample_count: Number of raw observations backing the driver-specific
            estimate.
    """

    signal: TrackedSignal
    value: float
    driver_mean: Optional[float]
    driver_std: Optional[float]
    global_default: float
    confidence: float
    sample_count: int


# ---------------------------------------------------------------------------
# Personalization Engine
# ---------------------------------------------------------------------------


class PersonalizationEngine:
    """Learns per-driver behavioural baselines and computes adaptive thresholds.

    Responsibilities (per the project specification):

        1. **Learn driver baselines** — running mean/std per tracked raw
           signal via Welford's algorithm (:meth:`observe`).
        2. **Compute adaptive thresholds** — confidence-weighted blend of
           the driver's own ``mu ± k*sigma`` band and the global default
           (:meth:`compute_threshold`, :meth:`compute_all_thresholds`).
        3. **Update thresholds online** — every call to :meth:`observe`
           incrementally refines the baseline without ever needing to
           replay history.
        4. **Support cold-start drivers** — confidence starts at 0 (pure
           global default) and rises smoothly as both session count and
           per-signal sample count grow.
        5. **Calculate personalization confidence** — :meth:`personalization_confidence`
           and the per-threshold ``confidence`` field above.

    Thread-safety: a per-instance lock guards the in-memory baseline cache.

    Attributes:
        z_multiplier: Default z-score multiplier used when deriving
            threshold bands from a driver's learned mean/std.
    """

    def __init__(self, z_multiplier: float = DEFAULT_Z_MULTIPLIER) -> None:
        """Initialises the personalization engine with an empty baseline cache.

        Args:
            z_multiplier: Default standard-deviation multiplier (k) for
                deriving ``mu ± k*sigma`` adaptive threshold bands.

        Raises:
            ValueError: If ``z_multiplier`` is not positive.
        """
        if z_multiplier <= 0.0:
            raise ValueError(f"z_multiplier must be positive, got {z_multiplier}.")

        self.z_multiplier = z_multiplier
        self._states: Dict[int, DriverBaselineState] = {}
        self._lock = threading.Lock()
        self._global_defaults = _global_defaults()

        logger.info(
            "PersonalizationEngine initialised (z_multiplier=%.2f, "
            "tracked_signals=%d).",
            z_multiplier,
            len(TrackedSignal),
        )

    # ------------------------------------------------------------------
    # Baseline retrieval
    # ------------------------------------------------------------------

    def get_or_create(self, driver_id: int) -> DriverBaselineState:
        """Retrieves a driver's baseline state, creating a cold-start state if absent.

        Args:
            driver_id: Primary key of the driver profile.

        Returns:
            DriverBaselineState: Existing or newly created baseline state.

        Raises:
            ValueError: If ``driver_id`` is not a positive integer.
        """
        if driver_id <= 0:
            raise ValueError(f"driver_id must be a positive integer, got {driver_id}.")

        with self._lock:
            state = self._states.get(driver_id)
            if state is None:
                state = DriverBaselineState.new(driver_id)
                self._states[driver_id] = state
                logger.info(
                    "Created cold-start baseline state for driver_id=%d.", driver_id
                )
            return state

    # ------------------------------------------------------------------
    # Online learning
    # ------------------------------------------------------------------

    def observe(self, driver_id: int, signal_values: Dict[TrackedSignal, float]) -> None:
        """Incorporates one frame's worth of raw signal observations into the baseline.

        This is the "learn driver baselines" / "update thresholds online"
        entry point — called once per processed frame (or per
        sliding-window aggregate) during a live session.

        Args:
            driver_id: Primary key of the driver profile being observed.
            signal_values: Mapping from tracked signal to its current raw
                value (e.g. ``{TrackedSignal.EAR_MEAN: 0.21, ...}``). Signals
                not present in the mapping are simply skipped for this call.

        Raises:
            ValueError: If ``driver_id`` is invalid or any provided value
                is non-finite.
        """
        if driver_id <= 0:
            raise ValueError(f"driver_id must be a positive integer, got {driver_id}.")

        for signal, value in signal_values.items():
            if not np.isfinite(value):
                raise ValueError(
                    f"Non-finite value for signal {signal.value}: {value!r}."
                )

        with self._lock:
            state = self._states.get(driver_id)
            if state is None:
                state = DriverBaselineState.new(driver_id)
                self._states[driver_id] = state

            for signal, value in signal_values.items():
                tracker = state.trackers[signal]
                tracker.update(np.array([value], dtype=np.float64))

            state.last_updated = datetime.now(timezone.utc)

    def end_session(self, driver_id: int) -> DriverBaselineState:
        """Marks the completion of a driving session for a driver.

        Increments the driver's session counter, which feeds the
        session-based component of :meth:`personalization_confidence`. This
        is distinct from per-frame sample counts (tracked automatically by
        each signal's Welford tracker via :meth:`observe`).

        Args:
            driver_id: Primary key of the driver profile whose session just ended.

        Returns:
            DriverBaselineState: The updated baseline state.

        Raises:
            ValueError: If ``driver_id`` is invalid.
        """
        if driver_id <= 0:
            raise ValueError(f"driver_id must be a positive integer, got {driver_id}.")

        with self._lock:
            state = self._states.get(driver_id)
            if state is None:
                state = DriverBaselineState.new(driver_id)
                self._states[driver_id] = state

            state.n_sessions += 1
            state.last_updated = datetime.now(timezone.utc)

            logger.info(
                "Session ended for driver_id=%d: n_sessions=%d.",
                driver_id,
                state.n_sessions,
            )
            return state

    # ------------------------------------------------------------------
    # Z-score computation
    # ------------------------------------------------------------------

    def compute_z_score(
        self, driver_id: int, signal: TrackedSignal, value: float
    ) -> Optional[float]:
        """Computes a personalised z-score for a new observation.

        .. math::

            z = \\frac{x - \\mu_{\\text{driver}}}{\\sigma_{\\text{driver}} + \\epsilon}

        Args:
            driver_id: Primary key of the driver profile.
            signal: Which tracked signal ``value`` corresponds to.
            value: The raw observation to score.

        Returns:
            Optional[float]: The personalised z-score, or ``None`` if the
            driver does not yet have enough samples
            (:data:`MIN_SAMPLES_FOR_BASELINE`) for this signal to compute a
            meaningful baseline.

        Raises:
            ValueError: If ``driver_id`` is invalid or ``value`` is
                non-finite.
        """
        if driver_id <= 0:
            raise ValueError(f"driver_id must be a positive integer, got {driver_id}.")
        if not np.isfinite(value):
            raise ValueError(f"value must be finite, got {value!r}.")

        state = self._states.get(driver_id)
        if state is None:
            return None

        tracker = state.trackers[signal]
        if tracker.count < MIN_SAMPLES_FOR_BASELINE:
            return None

        mu = float(tracker.mean[0])
        sigma = float(tracker.std[0])
        z = (value - mu) / (sigma + _EPS)
        return float(z)

    # ------------------------------------------------------------------
    # Adaptive threshold computation
    # ------------------------------------------------------------------

    def compute_threshold(
        self, driver_id: int, signal: TrackedSignal, z_multiplier: Optional[float] = None
    ) -> AdaptiveThreshold:
        """Computes a single confidence-weighted personalised threshold.

        The personalised candidate threshold is derived as ``mu - k*sigma``
        (for "lower is riskier" signals like EAR), ``mu + k*sigma`` (for
        "higher is riskier" signals like PERCLOS/MAR), or
        ``mu + k*sigma`` applied to the *absolute deviation* for two-sided
        signals (gaze, head pose), then blended with the global default:

        .. math::

            T = c \\cdot T_{\\text{driver}} + (1 - c) \\cdot T_{\\text{global}}

        where ``c`` is :meth:`personalization_confidence` for this driver
        and signal, ensuring cold-start drivers transparently use the
        global default and seasoned drivers converge to a fully
        personalised threshold.

        Args:
            driver_id: Primary key of the driver profile.
            signal: Which tracked signal to compute a threshold for.
            z_multiplier: Optional override for this call's z-score
                multiplier; defaults to :attr:`z_multiplier`.

        Returns:
            AdaptiveThreshold: The blended threshold plus full provenance
            (driver mean/std, global default, confidence, sample count).

        Raises:
            ValueError: If ``driver_id`` is invalid.
        """
        if driver_id <= 0:
            raise ValueError(f"driver_id must be a positive integer, got {driver_id}.")

        k = z_multiplier if z_multiplier is not None else self.z_multiplier
        global_default = self._global_defaults[signal]

        state = self._states.get(driver_id)
        tracker = state.trackers[signal] if state is not None else None
        sample_count = tracker.count if tracker is not None else 0

        if tracker is None or sample_count < MIN_SAMPLES_FOR_BASELINE:
            return AdaptiveThreshold(
                signal=signal,
                value=global_default,
                driver_mean=None,
                driver_std=None,
                global_default=global_default,
                confidence=0.0,
                sample_count=sample_count,
            )

        mu = float(tracker.mean[0])
        sigma = float(tracker.std[0])

        lower_is_riskier = _LOWER_IS_RISKIER[signal]
        if lower_is_riskier is True:
            driver_threshold = mu - k * sigma
        elif lower_is_riskier is False:
            driver_threshold = mu + k * sigma
        else:
            # Two-sided signal: personalised threshold is an allowed deviation
            # band radius around the driver's own mean, in the same units as
            # the global default (which is itself expressed as a deviation
            # limit, e.g. MAX_YAW_THRESHOLD or DEFAULT_GAZE_THRESHOLD).
            driver_threshold = k * sigma

        confidence = self.personalization_confidence(driver_id, signal)
        blended = confidence * driver_threshold + (1.0 - confidence) * global_default

        return AdaptiveThreshold(
            signal=signal,
            value=float(blended),
            driver_mean=mu,
            driver_std=sigma,
            global_default=global_default,
            confidence=confidence,
            sample_count=sample_count,
        )

    def compute_all_thresholds(self, driver_id: int) -> Dict[TrackedSignal, AdaptiveThreshold]:
        """Computes personalised thresholds for every tracked signal at once.

        Convenience batch wrapper around :meth:`compute_threshold`, typically
        called once per inference cycle by
        :mod:`backend.ml.digital_twin.threshold_manager`.

        Args:
            driver_id: Primary key of the driver profile.

        Returns:
            Dict[TrackedSignal, AdaptiveThreshold]: One entry per
            :class:`TrackedSignal`.
        """
        return {signal: self.compute_threshold(driver_id, signal) for signal in TrackedSignal}

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def personalization_confidence(
        self, driver_id: int, signal: Optional[TrackedSignal] = None
    ) -> float:
        """Computes how much a driver's learned baseline should be trusted.

        Combines two independent factors, each in ``[0, 1]``:

            - **Session maturity**: ``min(n_sessions / MIN_SESSIONS_FOR_FULL_CONFIDENCE, 1)``
            - **Sample sufficiency** (if ``signal`` given): ``min(sample_count / (2 * MIN_SAMPLES_FOR_BASELINE), 1)``,
              otherwise defaults to ``1.0`` (ignored) when no specific
              signal is requested.

        The final confidence is the product of both factors, so a driver
        with many sessions but very few samples for one particular signal
        (e.g. a signal that briefly failed to track) is still appropriately
        discounted for that signal.

        Args:
            driver_id: Primary key of the driver profile.
            signal: Optional specific signal to factor sample sufficiency
                for. If omitted, only session maturity is used.

        Returns:
            float: Confidence in ``[0, 1]``. ``0.0`` for an unknown driver.
        """
        state = self._states.get(driver_id)
        if state is None:
            return 0.0

        session_factor = min(state.n_sessions / float(MIN_SESSIONS_FOR_FULL_CONFIDENCE), 1.0)

        if signal is None:
            return float(session_factor)

        tracker = state.trackers[signal]
        sample_factor = min(tracker.count / float(2 * MIN_SAMPLES_FOR_BASELINE), 1.0)

        return float(session_factor * sample_factor)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def evict(self, driver_id: int) -> bool:
        """Removes a driver's in-memory baseline state.

        Args:
            driver_id: Primary key of the driver profile to evict.

        Returns:
            bool: True if a cached state was found and removed.
        """
        with self._lock:
            return self._states.pop(driver_id, None) is not None

    def reset_signal(self, driver_id: int, signal: TrackedSignal) -> None:
        """Resets a single signal's learned baseline for a driver to cold-start.

        Useful when a sensor/landmark tracker is suspected to have drifted
        or been miscalibrated for a specific signal without discarding the
        driver's entire baseline.

        Args:
            driver_id: Primary key of the driver profile.
            signal: Which tracked signal to reset.

        Raises:
            ValueError: If ``driver_id`` is invalid.
        """
        if driver_id <= 0:
            raise ValueError(f"driver_id must be a positive integer, got {driver_id}.")

        with self._lock:
            state = self._states.get(driver_id)
            if state is None:
                return
            state.trackers[signal] = WelfordState(
                mean=np.zeros(1, dtype=np.float64), m2=np.zeros(1, dtype=np.float64)
            )
            logger.info(
                "Reset baseline for driver_id=%d, signal=%s.", driver_id, signal.value
            )
