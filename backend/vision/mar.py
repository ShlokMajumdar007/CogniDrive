"""Mouth Aspect Ratio (MAR) calculations for yawn detection.

Calculates mouth opening to detect yawns and assess driver fatigue.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Mouth state thresholds
MAR_YAWN_THRESHOLD: float = 0.60
MAR_OPEN_THRESHOLD: float = 0.35
MAR_CLOSED_THRESHOLD: float = 0.35
YAWN_CONSEC_FRAMES: int = 15


class MARState(str, Enum):
    """Discrete mouth-openness state."""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    YAWNING = "YAWNING"
    UNKNOWN = "UNKNOWN"


@dataclass
class MARResult:
    """Mouth Aspect Ratio metrics for a single frame."""
    mar: float = 0.0
    state: MARState = MARState.UNKNOWN
    is_yawning: bool = False


def compute_mar(mouth: List[Tuple[float, float]]) -> MARResult:
    """Computes MAR from 8 landmark coordinates."""
    if len(mouth) != 8:
        logger.warning(
            "MAR requires exactly 8 mouth landmarks. Received: %d", len(mouth)
        )
        return MARResult(state=MARState.UNKNOWN)

    try:
        # Vertical distances
        v1 = math.dist(mouth[1], mouth[7])
        v2 = math.dist(mouth[2], mouth[6])
        v3 = math.dist(mouth[3], mouth[5])

        # Horizontal distance (corners)
        h = math.dist(mouth[0], mouth[4])

        if h < 1e-6:
            return MARResult(mar=0.0, state=MARState.CLOSED, is_yawning=False)

        mar = (v1 + v2 + v3) / (2.0 * h)

    except Exception as exc:
        logger.warning("MAR computation failed: %s", exc)
        return MARResult(state=MARState.UNKNOWN)

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


class YawnTracker:
    """Tracks yawn events and rolling MAR history over a driving session."""

    def __init__(
        self,
        yawn_consec_frames: int = YAWN_CONSEC_FRAMES,
        history_size: int = 300,
    ) -> None:
        self._yawn_consec_frames = yawn_consec_frames
        self._history_size = history_size
        self.total_yawns: int = 0
        self.consec_yawning: int = 0
        self.mar_history: List[float] = []

    def update(self, mar_result: MARResult) -> MARResult:
        """Processes a new MAR result and updates the yawn counter."""
        self.mar_history.append(mar_result.mar)
        if len(self.mar_history) > self._history_size:
            self.mar_history.pop(0)

        if mar_result.state == MARState.YAWNING:
            self.consec_yawning += 1
            if self.consec_yawning == self._yawn_consec_frames:
                self.total_yawns += 1
                mar_result.is_yawning = True
            elif self.consec_yawning > self._yawn_consec_frames:
                mar_result.is_yawning = True
        else:
            self.consec_yawning = 0
            mar_result.is_yawning = False

        return mar_result

    def mean_mar(self) -> float:
        """Returns the average MAR over the rolling history window."""
        if not self.mar_history:
            return 0.0
        return sum(self.mar_history) / len(self.mar_history)

    def yawns_per_minute(self, fps: float = 30.0, elapsed_frames: int = 0) -> float:
        """Estimates yawns per minute given total elapsed frames."""
        if elapsed_frames < fps or fps < 1e-6:
            return 0.0
        elapsed_minutes = elapsed_frames / (fps * 60.0)
        return self.total_yawns / elapsed_minutes

    def reset(self) -> None:
        """Resets all session counters."""
        self.total_yawns = 0
        self.consec_yawning = 0
        self.mar_history.clear()
