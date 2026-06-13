"""PERCLOS (Percentage of Eye Closure) fatigue measurement.

PERCLOS is the gold-standard psychophysiological metric for measuring
drowsiness. It is defined as the proportion of a rolling time window
during which the eyes are at least 80% closed (P80 standard).

References:
    - Wierwille et al. (1994) — PERCLOS as drowsiness predictor.
    - NHTSA Technical Report DOT HS 808 762 (1998).

Interpretation thresholds (P80)::

    PERCLOS < 0.15  → ALERT
    0.15 ≤ PERCLOS < 0.30  → DROWSY
    PERCLOS ≥ 0.30  → SEVERELY_DROWSY

Typical usage::

    from backend.vision.perclos import PERCLOSCalculator, PERCLOSResult
    calculator = PERCLOSCalculator(fps=30, window_seconds=60)
    result = calculator.update(ear_value=0.18)
    print(result.perclos, result.state)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PERCLOS configuration constants
# ---------------------------------------------------------------------------

#: P80 — EAR below this fraction of full-open is considered "closed" for PERCLOS.
PERCLOS_CLOSURE_RATIO: float = 0.80  # eye ≥ 80% closed when EAR < 0.20

#: Absolute EAR threshold for the P80 closure criterion.
PERCLOS_EAR_CLOSED_THRESHOLD: float = 0.20

#: Default rolling window duration in seconds.
DEFAULT_WINDOW_SECONDS: int = 60

#: Default camera frame rate.
DEFAULT_FPS: float = 30.0

# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------

#: PERCLOS above this triggers a DROWSY alert.
PERCLOS_DROWSY_THRESHOLD: float = 0.15

#: PERCLOS above this triggers a SEVERELY_DROWSY alert.
PERCLOS_SEVERE_THRESHOLD: float = 0.30


# ---------------------------------------------------------------------------
# Drowsiness state enum
# ---------------------------------------------------------------------------


class DrowsinessState(str, Enum):
    """Discrete drowsiness state derived from the PERCLOS value.

    Attributes:
        ALERT: Driver is alert (PERCLOS < 0.15).
        DROWSY: Driver shows signs of drowsiness (0.15 ≤ PERCLOS < 0.30).
        SEVERELY_DROWSY: Driver is severely drowsy (PERCLOS ≥ 0.30).
        INSUFFICIENT_DATA: Not enough frames collected to compute PERCLOS.
    """

    ALERT = "ALERT"
    DROWSY = "DROWSY"
    SEVERELY_DROWSY = "SEVERELY_DROWSY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PERCLOSResult:
    """PERCLOS measurement result for the current rolling window.

    Attributes:
        perclos: Proportion of the window with eyes closed [0, 1].
        state: Discrete :class:`DrowsinessState` classification.
        window_frames: Number of frames in the current analysis window.
        closed_frames: Number of frames classified as eyes-closed.
        mean_ear: Mean EAR value over the window.
        min_ear: Minimum EAR value in the window (worst-case closure).
        fatigue_probability: Estimated fatigue probability [0, 1] derived
            from a linear mapping of PERCLOS to [0, 1] using the thresholds.
        is_drowsy: True when state is DROWSY or SEVERELY_DROWSY.
    """

    perclos: float = 0.0
    state: DrowsinessState = DrowsinessState.INSUFFICIENT_DATA
    window_frames: int = 0
    closed_frames: int = 0
    mean_ear: float = 0.0
    min_ear: float = 0.0
    fatigue_probability: float = 0.0
    is_drowsy: bool = False


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class PERCLOSCalculator:
    """Stateful rolling-window PERCLOS calculator.

    Maintains a sliding window of EAR values (length = fps × window_seconds)
    and computes PERCLOS on every update. This design is O(1) per frame
    using a ``collections.deque`` with a fixed max-length.

    Attributes:
        _fps: Camera frames per second.
        _window_seconds: Rolling window duration in seconds.
        _window_size: Number of frames in the window (fps × window_seconds).
        _ear_window: Rolling deque of EAR float values.
        _closed_count: Count of closed-eye frames currently in the window.
        _total_frames_processed: Cumulative frames processed since reset.
    """

    def __init__(
        self,
        fps: float = DEFAULT_FPS,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        ear_closed_threshold: float = PERCLOS_EAR_CLOSED_THRESHOLD,
    ) -> None:
        """Initialises the PERCLOS calculator.

        Args:
            fps: Camera frames per second.
            window_seconds: Duration of the rolling analysis window.
            ear_closed_threshold: EAR value below which the eye is counted
                as closed for the PERCLOS P80 criterion.
        """
        self._fps = fps
        self._window_seconds = window_seconds
        self._ear_closed_threshold = ear_closed_threshold
        self._window_size: int = max(1, int(fps * window_seconds))

        self._ear_window: Deque[float] = deque(maxlen=self._window_size)
        self._closed_count: int = 0
        self._total_frames_processed: int = 0

        logger.info(
            "PERCLOSCalculator initialised — fps=%.1f, window=%ds (%d frames), "
            "closure_threshold=%.2f",
            fps,
            window_seconds,
            self._window_size,
            ear_closed_threshold,
        )

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, ear_value: float) -> PERCLOSResult:
        """Ingests a new EAR sample and returns the current PERCLOS result.

        Thread-safety: This method is NOT thread-safe. The caller must
        ensure single-threaded access or use an external lock.

        Args:
            ear_value: EAR value for the current frame (range 0 – ~0.40).

        Returns:
            PERCLOSResult: Updated PERCLOS result for the current window.
        """
        self._total_frames_processed += 1
        is_closed = ear_value < self._ear_closed_threshold

        # Evict the oldest sample if the window is full
        if len(self._ear_window) == self._window_size:
            evicted = self._ear_window[0]
            if evicted < self._ear_closed_threshold:
                self._closed_count -= 1

        self._ear_window.append(ear_value)
        if is_closed:
            self._closed_count += 1

        return self._compute_result()

    def _compute_result(self) -> PERCLOSResult:
        """Computes the PERCLOS metric and derives fatigue probability.

        Returns:
            PERCLOSResult: Current window PERCLOS measurement.
        """
        window_len = len(self._ear_window)

        if window_len < int(self._fps * 5):
            # Need at least 5 seconds of data for meaningful PERCLOS
            return PERCLOSResult(
                state=DrowsinessState.INSUFFICIENT_DATA,
                window_frames=window_len,
            )

        ear_arr = np.array(self._ear_window, dtype=np.float32)
        perclos = self._closed_count / window_len
        mean_ear = float(ear_arr.mean())
        min_ear = float(ear_arr.min())

        # Classify state
        if perclos >= PERCLOS_SEVERE_THRESHOLD:
            state = DrowsinessState.SEVERELY_DROWSY
        elif perclos >= PERCLOS_DROWSY_THRESHOLD:
            state = DrowsinessState.DROWSY
        else:
            state = DrowsinessState.ALERT

        # Map PERCLOS → fatigue_probability (linear, capped at 1.0)
        fatigue_prob = min(1.0, perclos / PERCLOS_SEVERE_THRESHOLD)

        return PERCLOSResult(
            perclos=round(perclos, 4),
            state=state,
            window_frames=window_len,
            closed_frames=self._closed_count,
            mean_ear=round(mean_ear, 4),
            min_ear=round(min_ear, 4),
            fatigue_probability=round(fatigue_prob, 4),
            is_drowsy=(state in (DrowsinessState.DROWSY, DrowsinessState.SEVERELY_DROWSY)),
        )

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def update_batch(self, ear_values: List[float]) -> PERCLOSResult:
        """Processes a batch of EAR values and returns the final result.

        Args:
            ear_values: List of EAR values in chronological order.

        Returns:
            PERCLOSResult: PERCLOS result after ingesting all values.
        """
        result = PERCLOSResult(state=DrowsinessState.INSUFFICIENT_DATA)
        for ear in ear_values:
            result = self.update(ear)
        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def window_fill_ratio(self) -> float:
        """Fraction of the window that has been filled with samples.

        Returns:
            float: Fill ratio in [0, 1]. Reaches 1.0 after ``window_size`` frames.
        """
        return len(self._ear_window) / self._window_size

    @property
    def total_frames_processed(self) -> int:
        """Returns the cumulative number of frames processed since the last reset.

        Returns:
            int: Total processed frame count.
        """
        return self._total_frames_processed

    @property
    def window_size(self) -> int:
        """Returns the configured rolling window size in frames.

        Returns:
            int: Window size.
        """
        return self._window_size

    def current_ear_history(self) -> List[float]:
        """Returns a copy of the current EAR rolling window.

        Returns:
            List[float]: List of EAR values in the window, oldest first.
        """
        return list(self._ear_window)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clears the rolling window and resets all counters for a new session."""
        self._ear_window.clear()
        self._closed_count = 0
        self._total_frames_processed = 0
        logger.info("PERCLOSCalculator reset.")
