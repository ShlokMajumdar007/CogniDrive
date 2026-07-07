"""Eye Aspect Ratio (EAR) calculations for tracking eye blinks and closures.

Calculates how open the eyes are to detect blinks and potential drowsiness.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Eye state thresholds
EAR_CLOSED_THRESHOLD: float = 0.20
EAR_PARTIAL_THRESHOLD: float = 0.25
EAR_OPEN_THRESHOLD: float = 0.25
BLINK_CONSEC_FRAMES: int = 2


class EARState(str, Enum):
    """Eye openness state derived from the EAR value."""
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


@dataclass
class EARResult:
    """Eye Aspect Ratio results for a single frame."""
    left_ear: float = 0.0
    right_ear: float = 0.0
    mean_ear: float = 0.0
    state: EARState = EARState.UNKNOWN
    is_blink: bool = False
    is_closed: bool = False


def _ear_single(eye: List[Tuple[float, float]]) -> float:
    """Computes EAR for a single eye using 6 landmark points."""
    if len(eye) != 6:
        raise ValueError(
            f"Expected exactly 6 eye landmarks for EAR. Received: {len(eye)}"
        )

    # Vertical distances
    v1 = math.dist(eye[1], eye[5])
    v2 = math.dist(eye[2], eye[4])

    # Horizontal distance
    h = math.dist(eye[0], eye[3])

    if h < 1e-6:
        return 0.0

    return (v1 + v2) / (2.0 * h)


def compute_ear(
    left_eye: List[Tuple[float, float]],
    right_eye: List[Tuple[float, float]],
    blink_consec_frames: int = BLINK_CONSEC_FRAMES,
    closed_frame_count: int = 0,
) -> EARResult:
    """Computes combined Eye Aspect Ratio for both eyes."""
    try:
        left_ear = _ear_single(left_eye)
        right_ear = _ear_single(right_eye)
    except (ValueError, Exception) as exc:
        logger.warning("EAR computation failed: %s", exc)
        return EARResult(state=EARState.UNKNOWN)

    mean_ear = (left_ear + right_ear) / 2.0

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


class BlinkTracker:
    """Tracks eye blinks and rolling EAR history over a driving session."""

    def __init__(
        self,
        blink_consec_frames: int = BLINK_CONSEC_FRAMES,
        history_size: int = 300,
    ) -> None:
        self._blink_consec_frames = blink_consec_frames
        self._history_size = history_size
        self.total_blinks: int = 0
        self.consec_closed: int = 0
        self.ear_history: List[float] = []
        self.blink_timestamps: List[int] = []
        self._frame_index: int = 0

    def update(self, ear_result: EARResult) -> EARResult:
        """Processes a new frame's EAR result and updates blink count."""
        self._frame_index += 1

        self.ear_history.append(ear_result.mean_ear)
        if len(self.ear_history) > self._history_size:
            self.ear_history.pop(0)

        if ear_result.is_closed:
            self.consec_closed += 1
        else:
            if self.consec_closed >= self._blink_consec_frames:
                self.total_blinks += 1
                self.blink_timestamps.append(self._frame_index)
                ear_result.is_blink = True
            self.consec_closed = 0

        return ear_result

    def blinks_per_minute(self, fps: float = 30.0) -> float:
        """Estimates blink rate based on recent blink timestamps."""
        if len(self.blink_timestamps) < 2:
            return 0.0
        elapsed_frames = self.blink_timestamps[-1] - self.blink_timestamps[0]
        elapsed_minutes = elapsed_frames / (fps * 60.0)
        if elapsed_minutes < 1e-6:
            return 0.0
        return (len(self.blink_timestamps) - 1) / elapsed_minutes

    def mean_ear(self) -> float:
        """Returns the mean EAR over the rolling history window."""
        if not self.ear_history:
            return 0.0
        return sum(self.ear_history) / len(self.ear_history)

    def reset(self) -> None:
        """Resets all session trackers."""
        self.total_blinks = 0
        self.consec_closed = 0
        self.ear_history.clear()
        self.blink_timestamps.clear()
        self._frame_index = 0
