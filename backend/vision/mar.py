"""Mouth Aspect Ratio (MAR) computation for yawn detection.

The MAR metric quantifies how wide the mouth is open on a frame-by-frame
basis. A high MAR sustained over several frames indicates a yawn — a key
fatigue signal used in drowsiness detection systems.

Mathematical definition::

    MAR = (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (2 * ||p1 - p5||)

Where p1..p8 are the eight mouth landmark points in the order returned by
:data:`~backend.vision.landmark_extractor.MOUTH_INDICES`.

Typical usage::

    from backend.vision.mar import compute_mar, MARResult, MARState
    result = compute_mar(mouth_landmarks)
    if result.state == MARState.YAWNING:
        log_yawn_event()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

#: MAR above this value is considered a yawn.
MAR_YAWN_THRESHOLD: float = 0.60

#: MAR above this value but below YAWN is considered a wide mouth open.
MAR_OPEN_THRESHOLD: float = 0.35

#: MAR below this value is a closed or neutral mouth.
MAR_CLOSED_THRESHOLD: float = 0.35

#: Consecutive frames above threshold to confirm a yawn event.
YAWN_CONSEC_FRAMES: int = 15


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class MARState(str, Enum):
    """Discrete mouth-openness state derived from the MAR value.

    Attributes:
        CLOSED: Mouth is closed or slightly open (MAR < 0.35).
        OPEN: Mouth is notably open (0.35 ≤ MAR < 0.60).
        YAWNING: Mouth is fully open in a yawn (MAR ≥ 0.60).
        UNKNOWN: Landmarks were invalid or MAR could not be computed.
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    YAWNING = "YAWNING"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MARResult:
    """Per-frame Mouth Aspect Ratio result.

    Attributes:
        mar: MAR value for the frame.
        state: Discrete :class:`MARState` classification.
        is_yawning: True when MAR exceeds :data:`MAR_YAWN_THRESHOLD`.
    """

    mar: float = 0.0
    state: MARState = MARState.UNKNOWN
    is_yawning: bool = False


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


def compute_mar(mouth: List[Tuple[float, float]]) -> MARResult:
    """Computes the Mouth Aspect Ratio from 8 mouth landmark coordinates.

    The 8-point MAR uses three vertical distances and one horizontal
    distance to robustly capture both yawn width and height.

    Expected landmark order (matched to :data:`~backend.vision.landmark_extractor.MOUTH_INDICES`)::

        [p1_left-corner, p2_top-outer-left, p3_top-inner,
         p4_top-inner-right, p5_right-corner, p6_bottom-inner-right,
         p7_bottom-inner, p8_bottom-outer-left]

    Args:
        mouth: List of exactly 8 (x, y) landmark tuples.

    Returns:
        MARResult: Per-frame MAR result with state classification.
    """
    if len(mouth) != 8:
        logger.warning(
            "MAR requires exactly 8 mouth landmarks. Received: %d", len(mouth)
        )
        return MARResult(state=MARState.UNKNOWN)

    try:
        # Three vertical distances (top ↔ bottom pairs)
        v1 = _euclidean(mouth[1], mouth[7])
        v2 = _euclidean(mouth[2], mouth[6])
        v3 = _euclidean(mouth[3], mouth[5])

        # One horizontal distance (left corner ↔ right corner)
        h = _euclidean(mouth[0], mouth[4])

        if h < 1e-6:
            return MARResult(mar=0.0, state=MARState.CLOSED, is_yawning=False)

        mar = (v1 + v2 + v3) / (2.0 * h)

    except Exception as exc:
        logger.warning("MAR computation failed: %s", exc)
        return MARResult(state=MARState.UNKNOWN)

    # Classify state
    if mar >= MAR_YAWN_THRESHOLD:
        state = MARState.YAWNING
    elif mar >= MAR_OPEN_THRESHOLD:
        state = MARState.OPEN
    else:
        state = MARState.CLOSED

    return MARResult(
        mar=round(mar, 4),
        state=state,
        is_yawning=(state == MARState.YAWNING),
    )


# ---------------------------------------------------------------------------
# Yawn tracker
# ---------------------------------------------------------------------------


class YawnTracker:
    """Stateful yawn event counter for a driving session.

    A yawn is confirmed only when the mouth stays open for at least
    :data:`YAWN_CONSEC_FRAMES` consecutive frames. This prevents
    brief speech or smiling from being counted as yawns.

    Attributes:
        total_yawns: Cumulative yawn event count.
        consec_yawning: Current consecutive yawning-frame counter.
        mar_history: Rolling list of recent MAR values.
    """

    def __init__(
        self,
        yawn_consec_frames: int = YAWN_CONSEC_FRAMES,
        history_size: int = 300,
    ) -> None:
        """Initialises the yawn tracker.

        Args:
            yawn_consec_frames: Consecutive yawning frames to confirm event.
            history_size: Maximum number of MAR samples to retain in history.
        """
        self._yawn_consec_frames = yawn_consec_frames
        self._history_size = history_size
        self.total_yawns: int = 0
        self.consec_yawning: int = 0
        self.mar_history: List[float] = []

    def update(self, mar_result: MARResult) -> MARResult:
        """Processes a new MAR result and updates the yawn counter.

        Args:
            mar_result: :class:`MARResult` for the current frame.

        Returns:
            MARResult: The same result with ``is_yawning`` set correctly
                based on accumulated state.
        """
        # Update history
        self.mar_history.append(mar_result.mar)
        if len(self.mar_history) > self._history_size:
            self.mar_history.pop(0)

        if mar_result.state == MARState.YAWNING:
            self.consec_yawning += 1
            if self.consec_yawning == self._yawn_consec_frames:
                # Rising edge — confirmed yawn start
                self.total_yawns += 1
                mar_result.is_yawning = True
            elif self.consec_yawning > self._yawn_consec_frames:
                mar_result.is_yawning = True
        else:
            self.consec_yawning = 0
            mar_result.is_yawning = False

        return mar_result

    def mean_mar(self) -> float:
        """Returns the mean MAR over the rolling history window.

        Returns:
            float: Mean MAR or 0.0 if history is empty.
        """
        if not self.mar_history:
            return 0.0
        return float(np.mean(self.mar_history))

    def yawns_per_minute(self, fps: float = 30.0, elapsed_frames: int = 0) -> float:
        """Estimates yawns per minute given elapsed frame count.

        Args:
            fps: Camera frames per second.
            elapsed_frames: Total frames processed in the current session.

        Returns:
            float: Estimated yawns per minute. Returns 0.0 if fewer than
                60 frames have elapsed.
        """
        if elapsed_frames < fps or fps < 1e-6:
            return 0.0
        elapsed_minutes = elapsed_frames / (fps * 60.0)
        return self.total_yawns / elapsed_minutes

    def reset(self) -> None:
        """Resets all counters and history for a new session."""
        self.total_yawns = 0
        self.consec_yawning = 0
        self.mar_history.clear()
