"""Gaze direction estimation to detect driver distraction.

Determines if the driver is looking at the road or looking away.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Gaze detection thresholds (normalized eye-space coords)
GAZE_HORIZONTAL_THRESHOLD: float = 0.35
GAZE_VERTICAL_THRESHOLD: float = 0.30
DISTRACTION_CONSEC_FRAMES: int = 10


class GazeDirection(str, Enum):
    """Discrete gaze directions."""
    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    OFF_ROAD = "OFF_ROAD"
    UNKNOWN = "UNKNOWN"


@dataclass
class GazeResult:
    """Gaze tracking results for a single frame."""
    direction: GazeDirection = GazeDirection.UNKNOWN
    horizontal_ratio: float = 0.5
    vertical_ratio: float = 0.5
    left_gaze: Tuple[float, float] = (0.5, 0.5)
    right_gaze: Tuple[float, float] = (0.5, 0.5)
    is_off_road: bool = False
    confidence: float = 0.0


def _compute_iris_ratio(
    eye: List[Tuple[float, float]],
    iris: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """Finds normalized iris center coordinate inside eye bounding box."""
    if not eye or not iris:
        return 0.5, 0.5

    try:
        eye_arr = np.array(eye, dtype=np.float32)
        iris_arr = np.array(iris, dtype=np.float32)
        iris_centre = iris_arr.mean(axis=0)

        eye_min = eye_arr.min(axis=0)
        eye_max = eye_arr.max(axis=0)
        eye_range = eye_max - eye_min

        if eye_range[0] < 1e-6 or eye_range[1] < 1e-6:
            return 0.5, 0.5

        h_ratio = float((iris_centre[0] - eye_min[0]) / eye_range[0])
        v_ratio = float((iris_centre[1] - eye_min[1]) / eye_range[1])

        h_ratio = float(np.clip(h_ratio, 0.0, 1.0))
        v_ratio = float(np.clip(v_ratio, 0.0, 1.0))

        return h_ratio, v_ratio

    except Exception as exc:
        logger.debug("Iris ratio computation failed: %s", exc)
        return 0.5, 0.5


def _fallback_horizontal_ratio(eye: List[Tuple[float, float]]) -> float:
    """Fallback horizontal estimation when iris landmarks are missing."""
    if len(eye) < 6:
        return 0.5
    try:
        mid_x = (eye[2][0] + eye[4][0]) / 2.0
        left_x = eye[0][0]
        right_x = eye[3][0]
        width = right_x - left_x
        if width < 1e-6:
            return 0.5
        return float(np.clip((mid_x - left_x) / width, 0.0, 1.0))
    except Exception:
        return 0.5


def _classify_direction(h_ratio: float, v_ratio: float) -> GazeDirection:
    h_offset = abs(h_ratio - 0.5)
    v_offset = abs(v_ratio - 0.5)

    off_h = h_offset > GAZE_HORIZONTAL_THRESHOLD
    off_v = v_offset > GAZE_VERTICAL_THRESHOLD

    if off_h and off_v:
        return GazeDirection.OFF_ROAD

    if off_h:
        return GazeDirection.LEFT if h_ratio < 0.5 else GazeDirection.RIGHT

    if off_v:
        return GazeDirection.UP if v_ratio < 0.5 else GazeDirection.DOWN

    return GazeDirection.FORWARD


class GazeEstimator:
    """Estimates gaze direction using iris-based mapping or eye corners fallback."""

    def estimate(
        self,
        left_eye: List[Tuple[float, float]],
        right_eye: List[Tuple[float, float]],
        left_iris: Optional[List[Tuple[float, float]]] = None,
        right_iris: Optional[List[Tuple[float, float]]] = None,
    ) -> GazeResult:
        if not left_eye or not right_eye:
            return GazeResult(direction=GazeDirection.UNKNOWN, confidence=0.0)

        has_iris = bool(left_iris and right_iris)

        if has_iris:
            left_h, left_v = _compute_iris_ratio(left_eye, left_iris)
            right_h, right_v = _compute_iris_ratio(right_eye, right_iris)
            confidence = 0.95
        else:
            left_h = _fallback_horizontal_ratio(left_eye)
            right_h = _fallback_horizontal_ratio(right_eye)
            left_v = right_v = 0.5
            confidence = 0.50

        mean_h = (left_h + right_h) / 2.0
        mean_v = (left_v + right_v) / 2.0

        direction = _classify_direction(mean_h, mean_v)
        is_off_road = direction in (
            GazeDirection.OFF_ROAD,
            GazeDirection.LEFT,
            GazeDirection.RIGHT,
            GazeDirection.DOWN,
        )

        return GazeResult(
            direction=direction,
            horizontal_ratio=round(mean_h, 4),
            vertical_ratio=round(mean_v, 4),
            left_gaze=(round(left_h, 4), round(left_v, 4)),
            right_gaze=(round(right_h, 4), round(right_v, 4)),
            is_off_road=is_off_road,
            confidence=confidence,
        )


class DistractionTracker:
    """Tracks driver's off-road attention duration."""

    def __init__(
        self,
        distraction_consec_frames: int = DISTRACTION_CONSEC_FRAMES,
        history_size: int = 300,
    ) -> None:
        self._distraction_consec_frames = distraction_consec_frames
        self._history_size = history_size
        self.total_distraction_events: int = 0
        self.consec_off_road: int = 0
        self.total_off_road_frames: int = 0
        self.gaze_history: List[GazeDirection] = []

    def update(self, gaze_result: GazeResult) -> GazeResult:
        self.gaze_history.append(gaze_result.direction)
        if len(self.gaze_history) > self._history_size:
            self.gaze_history.pop(0)

        if gaze_result.is_off_road:
            self.consec_off_road += 1
            self.total_off_road_frames += 1
            if self.consec_off_road == self._distraction_consec_frames:
                self.total_distraction_events += 1
        else:
            self.consec_off_road = 0

        return gaze_result

    def distraction_rate(self, total_frames: int) -> float:
        """Returns the ratio of distracted frames over total frames."""
        if total_frames <= 0:
            return 0.0
        return min(1.0, self.total_off_road_frames / total_frames)

    def reset(self) -> None:
        self.total_distraction_events = 0
        self.consec_off_road = 0
        self.total_off_road_frames = 0
        self.gaze_history.clear()
