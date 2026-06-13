"""Eye Aspect Ratio (EAR) computation for blink and drowsiness detection.

The EAR metric quantifies how open the eye is on a frame-by-frame basis.
A sharp drop in EAR indicates a blink; a sustained low EAR indicates
drowsiness or eye closure associated with microsleep.

Mathematical definition (Soukupová & Čech, 2016)::

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

Where p1..p6 are the six eye landmark points in the order returned by
:data:`~backend.vision.landmark_extractor.LEFT_EYE_INDICES` and
:data:`~backend.vision.landmark_extractor.RIGHT_EYE_INDICES`.

Typical usage::

    from backend.vision.ear import compute_ear, EARResult, EARState
    result = compute_ear(left_eye_pts, right_eye_pts)
    if result.state == EARState.CLOSED:
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

#: EAR below this value is considered a closed eye / blink.
EAR_CLOSED_THRESHOLD: float = 0.20

#: EAR below this value but above CLOSED is considered partially open.
EAR_PARTIAL_THRESHOLD: float = 0.25

#: EAR above this value is considered fully open.
EAR_OPEN_THRESHOLD: float = 0.25

#: Minimum number of consecutive closed frames before blink is confirmed.
BLINK_CONSEC_FRAMES: int = 2


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class EARState(str, Enum):
    """Discrete eye-openness state derived from the EAR value.

    Attributes:
        OPEN: Eye is fully open (EAR ≥ 0.25).
        PARTIAL: Eye is partially open (0.20 ≤ EAR < 0.25).
        CLOSED: Eye is closed (EAR < 0.20).
        UNKNOWN: Landmarks were invalid or EAR could not be computed.
    """

    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EARResult:
    """Per-frame Eye Aspect Ratio result.

    Attributes:
        left_ear: EAR value for the left eye.
        right_ear: EAR value for the right eye.
        mean_ear: Arithmetic mean of left and right EAR.
        state: Discrete :class:`EARState` classification.
        is_blink: True when a blink event is detected.
        is_closed: True when mean EAR is below :data:`EAR_CLOSED_THRESHOLD`.
    """

    left_ear: float = 0.0
    right_ear: float = 0.0
    mean_ear: float = 0.0
    state: EARState = EARState.UNKNOWN
    is_blink: bool = False
    is_closed: bool = False


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Computes Euclidean distance between two 2-D points.

    Args:
        p1: First point as (x, y).
        p2: Second point as (x, y).

    Returns:
        float: Euclidean distance.
    """
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def _ear_single(eye: List[Tuple[float, float]]) -> float:
    """Computes EAR for a single eye using the 6-point Soukupová formula.

    Args:
        eye: List of exactly 6 (x, y) landmark tuples in the order:
            [p1_outer, p2_top-outer, p3_top-inner,
             p4_inner, p5_bottom-inner, p6_bottom-outer].

    Returns:
        float: EAR value in range [0, ~0.40]. Returns 0.0 on invalid input.

    Raises:
        ValueError: If ``eye`` does not contain exactly 6 points.
    """
    if len(eye) != 6:
        raise ValueError(
            f"Expected exactly 6 eye landmarks for EAR. Received: {len(eye)}"
        )

    # Vertical distances
    v1 = _euclidean(eye[1], eye[5])
    v2 = _euclidean(eye[2], eye[4])

    # Horizontal distance
    h = _euclidean(eye[0], eye[3])

    if h < 1e-6:
        return 0.0

    return (v1 + v2) / (2.0 * h)


def compute_ear(
    left_eye: List[Tuple[float, float]],
    right_eye: List[Tuple[float, float]],
    blink_consec_frames: int = BLINK_CONSEC_FRAMES,
    closed_frame_count: int = 0,
) -> EARResult:
    """Computes the combined Eye Aspect Ratio for both eyes.

    Args:
        left_eye: Six (x, y) landmark tuples for the left eye.
        right_eye: Six (x, y) landmark tuples for the right eye.
        blink_consec_frames: Number of consecutive closed frames required
            to confirm a blink event.
        closed_frame_count: External counter tracking how many consecutive
            frames the eyes have been closed so far. Used to determine
            ``is_blink`` without maintaining internal state.

    Returns:
        EARResult: Per-frame EAR result containing both eye EARs, the mean,
            the discrete state classification, and blink/closed flags.
    """
    try:
        left_ear = _ear_single(left_eye)
        right_ear = _ear_single(right_eye)
    except (ValueError, Exception) as exc:
        logger.warning("EAR computation failed: %s", exc)
        return EARResult(state=EARState.UNKNOWN)

    mean_ear = (left_ear + right_ear) / 2.0

    # Classify state
    if mean_ear < EAR_CLOSED_THRESHOLD:
        state = EARState.CLOSED
    elif mean_ear < EAR_OPEN_THRESHOLD:
        state = EARState.PARTIAL
    else:
        state = EARState.OPEN

    is_closed = state == EARState.CLOSED
    is_blink = is_closed and closed_frame_count >= blink_consec_frames

    return EARResult(
        left_ear=round(left_ear, 4),
        right_ear=round(right_ear, 4),
        mean_ear=round(mean_ear, 4),
        state=state,
        is_blink=is_blink,
        is_closed=is_closed,
    )


# ---------------------------------------------------------------------------
# Blink rate tracker
# ---------------------------------------------------------------------------


class BlinkTracker:
    """Stateful blink event counter for a driving session.

    Maintains a rolling window of EAR values and counts confirmed blink
    events across frames. Blinks per minute (BPM) is a key drowsiness
    indicator — dangerously low BPM (<6) suggests microsleep.

    Attributes:
        total_blinks: Cumulative blink count for the session.
        consec_closed: Current consecutive closed-frame counter.
        ear_history: Rolling list of recent EAR values.
        blink_timestamps: List of frame indices at which blinks occurred.
    """

    def __init__(
        self,
        blink_consec_frames: int = BLINK_CONSEC_FRAMES,
        history_size: int = 300,
    ) -> None:
        """Initialises the blink tracker.

        Args:
            blink_consec_frames: Consecutive closed frames to confirm a blink.
            history_size: Maximum number of EAR samples to retain in history.
        """
        self._blink_consec_frames = blink_consec_frames
        self._history_size = history_size
        self.total_blinks: int = 0
        self.consec_closed: int = 0
        self.ear_history: List[float] = []
        self.blink_timestamps: List[int] = []
        self._frame_index: int = 0

    def update(self, ear_result: EARResult) -> EARResult:
        """Processes a new EAR result and updates the blink counter.

        Args:
            ear_result: :class:`EARResult` for the current frame.

        Returns:
            EARResult: The same result with ``is_blink`` set correctly
                based on accumulated state.
        """
        self._frame_index += 1

        # Update history
        self.ear_history.append(ear_result.mean_ear)
        if len(self.ear_history) > self._history_size:
            self.ear_history.pop(0)

        if ear_result.is_closed:
            self.consec_closed += 1
        else:
            # Eye re-opened — confirm blink if threshold met
            if self.consec_closed >= self._blink_consec_frames:
                self.total_blinks += 1
                self.blink_timestamps.append(self._frame_index)
                ear_result.is_blink = True
            self.consec_closed = 0

        return ear_result

    def blinks_per_minute(self, fps: float = 30.0) -> float:
        """Estimates blinks per minute from the recent timestamp history.

        Args:
            fps: Camera frames per second used to convert frame index to time.

        Returns:
            float: Estimated blinks per minute. Returns 0.0 if fewer than
                two blink events have been recorded.
        """
        if len(self.blink_timestamps) < 2:
            return 0.0
        elapsed_frames = self.blink_timestamps[-1] - self.blink_timestamps[0]
        elapsed_minutes = elapsed_frames / (fps * 60.0)
        if elapsed_minutes < 1e-6:
            return 0.0
        return (len(self.blink_timestamps) - 1) / elapsed_minutes

    def mean_ear(self) -> float:
        """Returns the mean EAR over the rolling history window.

        Returns:
            float: Mean EAR or 0.0 if history is empty.
        """
        if not self.ear_history:
            return 0.0
        return float(np.mean(self.ear_history))

    def reset(self) -> None:
        """Resets all counters and history for a new session."""
        self.total_blinks = 0
        self.consec_closed = 0
        self.ear_history.clear()
        self.blink_timestamps.clear()
        self._frame_index = 0
