"""Sliding window aggregation for temporal feature engineering.

The vision pipeline produces measurements at ~30 FPS. ML models require
temporally aggregated statistics over sliding windows (e.g., mean EAR over
the last 5 seconds) rather than single-frame snapshots.

This module provides:
    - :class:`SlidingWindow`: A generic O(1) rolling statistics accumulator.
    - :class:`SignalWindowManager`: Manages independent windows for each
      of the 6 key biometric signals.
    - :func:`aggregate_window`: Computes statistical descriptors over a window.

Window statistics produced::

    mean, std, min, max, median, p25, p75, last_value

Typical usage::

    mgr = SignalWindowManager(window_seconds=5, fps=30)
    mgr.update(ear_left=0.22, ear_right=0.21, mar=0.05, ...)
    stats = mgr.get_stats()
    # stats["ear_mean"]["mean"] → 0.235
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Window statistics
# ---------------------------------------------------------------------------


@dataclass
class WindowStats:
    """Descriptive statistics computed over a sliding window.

    Attributes:
        mean: Arithmetic mean of window values.
        std: Standard deviation of window values.
        minimum: Minimum value in the window.
        maximum: Maximum value in the window.
        median: Median value in the window.
        p25: 25th percentile.
        p75: 75th percentile.
        last: Most recent value added to the window.
        count: Number of values in the window.
    """

    mean: float = 0.0
    std: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    median: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    last: float = 0.0
    count: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Converts statistics to a flat dictionary.

        Returns:
            Dict[str, float]: Statistic name → value mapping.
        """
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "median": self.median,
            "p25": self.p25,
            "p75": self.p75,
            "last": self.last,
        }


# ---------------------------------------------------------------------------
# Generic sliding window
# ---------------------------------------------------------------------------


class SlidingWindow:
    """O(1) amortised sliding window backed by a fixed-size deque.

    Appending a new value and retrieving statistics are both O(1) amortised
    since NumPy operations are only performed on ``get_stats()`` calls, not
    on every append.

    Attributes:
        name: Human-readable name for logging.
        _maxlen: Maximum number of samples in the window.
        _window: Internal deque of float values.
    """

    def __init__(self, name: str, maxlen: int) -> None:
        """Initialises the sliding window.

        Args:
            name: Signal name for logging.
            maxlen: Maximum number of samples (window capacity).
        """
        self.name = name
        self._maxlen = maxlen
        self._window: Deque[float] = deque(maxlen=maxlen)

    def push(self, value: float) -> None:
        """Appends a new sample to the window.

        Args:
            value: New float measurement.
        """
        if not np.isfinite(value):
            logger.debug("SlidingWindow[%s] received non-finite value %s.", self.name, value)
            return
        self._window.append(value)

    def get_stats(self) -> WindowStats:
        """Computes descriptive statistics over the current window contents.

        Returns:
            WindowStats: Statistics dataclass. Returns default zero values
                if the window is empty.
        """
        if not self._window:
            return WindowStats(count=0)

        arr = np.array(self._window, dtype=np.float32)
        return WindowStats(
            mean=float(arr.mean()),
            std=float(arr.std()),
            minimum=float(arr.min()),
            maximum=float(arr.max()),
            median=float(np.median(arr)),
            p25=float(np.percentile(arr, 25)),
            p75=float(np.percentile(arr, 75)),
            last=float(self._window[-1]),
            count=len(self._window),
        )

    def get_values(self) -> List[float]:
        """Returns a copy of current window values, oldest first.

        Returns:
            List[float]: Window values.
        """
        return list(self._window)

    @property
    def is_full(self) -> bool:
        """True when the window has reached maximum capacity."""
        return len(self._window) == self._maxlen

    @property
    def fill_ratio(self) -> float:
        """Current fill ratio [0, 1] of the window."""
        return len(self._window) / self._maxlen

    @property
    def count(self) -> int:
        """Number of samples currently in the window."""
        return len(self._window)

    def reset(self) -> None:
        """Clears all values from the window."""
        self._window.clear()


# ---------------------------------------------------------------------------
# Signal window manager
# ---------------------------------------------------------------------------


@dataclass
class WindowSnapshot:
    """Aggregated statistics snapshot for all tracked biometric signals.

    Attributes:
        ear_left: Window stats for left eye EAR.
        ear_right: Window stats for right eye EAR.
        ear_mean: Window stats for mean EAR.
        mar: Window stats for MAR.
        perclos: Window stats for PERCLOS.
        gaze_horizontal: Window stats for horizontal gaze ratio.
        gaze_vertical: Window stats for vertical gaze ratio.
        head_pitch: Window stats for head pitch angle.
        head_yaw: Window stats for head yaw angle.
        head_roll: Window stats for head roll angle.
        window_fill_ratio: Fraction of the window that has been filled.
    """

    ear_left: WindowStats = field(default_factory=WindowStats)
    ear_right: WindowStats = field(default_factory=WindowStats)
    ear_mean: WindowStats = field(default_factory=WindowStats)
    mar: WindowStats = field(default_factory=WindowStats)
    perclos: WindowStats = field(default_factory=WindowStats)
    gaze_horizontal: WindowStats = field(default_factory=WindowStats)
    gaze_vertical: WindowStats = field(default_factory=WindowStats)
    head_pitch: WindowStats = field(default_factory=WindowStats)
    head_yaw: WindowStats = field(default_factory=WindowStats)
    head_roll: WindowStats = field(default_factory=WindowStats)
    window_fill_ratio: float = 0.0


class SignalWindowManager:
    """Manages independent sliding windows for all key biometric signals.

    Each signal has its own :class:`SlidingWindow` with capacity:
    ``window_size = int(fps * window_seconds)``.

    Attributes:
        _window_seconds: Duration of each window in seconds.
        _fps: Frames per second.
        _window_size: Number of frames per window.
        windows: Dictionary of signal name → :class:`SlidingWindow`.
    """

    _SIGNAL_NAMES: List[str] = [
        "ear_left",
        "ear_right",
        "ear_mean",
        "mar",
        "perclos",
        "gaze_horizontal",
        "gaze_vertical",
        "head_pitch",
        "head_yaw",
        "head_roll",
    ]

    def __init__(
        self,
        window_seconds: float = 5.0,
        fps: float = 30.0,
    ) -> None:
        """Initialises all signal windows.

        Args:
            window_seconds: Duration of each rolling window in seconds.
            fps: Camera frame rate used to convert seconds to frame count.
        """
        self._window_seconds = window_seconds
        self._fps = fps
        self._window_size = max(1, int(fps * window_seconds))

        self.windows: Dict[str, SlidingWindow] = {
            name: SlidingWindow(name=name, maxlen=self._window_size)
            for name in self._SIGNAL_NAMES
        }
        logger.info(
            "SignalWindowManager initialised — window=%.1fs (%d frames) at %.1f FPS.",
            window_seconds,
            self._window_size,
            fps,
        )

    def update(
        self,
        ear_left: float = 0.25,
        ear_right: float = 0.25,
        mar: float = 0.0,
        perclos: float = 0.0,
        gaze_horizontal: float = 0.5,
        gaze_vertical: float = 0.5,
        head_pitch: float = 0.0,
        head_yaw: float = 0.0,
        head_roll: float = 0.0,
    ) -> None:
        """Pushes a new frame's measurements into all signal windows.

        Args:
            ear_left: Left eye EAR value.
            ear_right: Right eye EAR value.
            mar: Mouth Aspect Ratio value.
            perclos: PERCLOS proportion.
            gaze_horizontal: Horizontal gaze ratio.
            gaze_vertical: Vertical gaze ratio.
            head_pitch: Head pitch in degrees.
            head_yaw: Head yaw in degrees.
            head_roll: Head roll in degrees.
        """
        ear_mean = (ear_left + ear_right) / 2.0

        self.windows["ear_left"].push(ear_left)
        self.windows["ear_right"].push(ear_right)
        self.windows["ear_mean"].push(ear_mean)
        self.windows["mar"].push(mar)
        self.windows["perclos"].push(perclos)
        self.windows["gaze_horizontal"].push(gaze_horizontal)
        self.windows["gaze_vertical"].push(gaze_vertical)
        self.windows["head_pitch"].push(head_pitch)
        self.windows["head_yaw"].push(head_yaw)
        self.windows["head_roll"].push(head_roll)

    def get_snapshot(self) -> WindowSnapshot:
        """Returns a statistics snapshot across all windows.

        Returns:
            WindowSnapshot: Per-signal :class:`WindowStats` for the current window.
        """
        fill = max(w.fill_ratio for w in self.windows.values())
        return WindowSnapshot(
            ear_left=self.windows["ear_left"].get_stats(),
            ear_right=self.windows["ear_right"].get_stats(),
            ear_mean=self.windows["ear_mean"].get_stats(),
            mar=self.windows["mar"].get_stats(),
            perclos=self.windows["perclos"].get_stats(),
            gaze_horizontal=self.windows["gaze_horizontal"].get_stats(),
            gaze_vertical=self.windows["gaze_vertical"].get_stats(),
            head_pitch=self.windows["head_pitch"].get_stats(),
            head_yaw=self.windows["head_yaw"].get_stats(),
            head_roll=self.windows["head_roll"].get_stats(),
            window_fill_ratio=fill,
        )

    def get_flat_stats(self) -> Dict[str, WindowStats]:
        """Returns a flat dict of signal name → :class:`WindowStats`.

        Returns:
            Dict[str, WindowStats]: All window statistics.
        """
        return {name: w.get_stats() for name, w in self.windows.items()}

    def reset(self) -> None:
        """Resets all windows for a new session."""
        for w in self.windows.values():
            w.reset()
        logger.info("SignalWindowManager reset.")


# ---------------------------------------------------------------------------
# Standalone aggregation utility
# ---------------------------------------------------------------------------


def aggregate_window(values: List[float]) -> WindowStats:
    """Computes descriptive statistics for an arbitrary list of values.

    Useful for one-off aggregation without maintaining a full manager.

    Args:
        values: List of float measurements.

    Returns:
        WindowStats: Computed statistics. Returns defaults if list is empty.
    """
    if not values:
        return WindowStats(count=0)
    arr = np.array(values, dtype=np.float32)
    return WindowStats(
        mean=float(arr.mean()),
        std=float(arr.std()),
        minimum=float(arr.min()),
        maximum=float(arr.max()),
        median=float(np.median(arr)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
        last=float(values[-1]),
        count=len(values),
    )
