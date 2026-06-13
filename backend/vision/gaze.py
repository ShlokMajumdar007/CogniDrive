"""Gaze direction estimation for driver attention and distraction detection.

Computes gaze direction by measuring the relative displacement of the iris
centre within the eye bounding box. A gaze vector that points far from the
forward-facing direction indicates the driver is looking away from the road.

Two methods are implemented:
    1. **Iris-based** (preferred): Uses MediaPipe refined iris landmarks to
       compute a normalised gaze offset. Requires ``refine_landmarks=True``.
    2. **EAR-based fallback**: Uses only eye corner landmarks to estimate
       gaze horizontally when iris data is unavailable.

Typical usage::

    from backend.vision.gaze import GazeEstimator, GazeResult, GazeDirection
    estimator = GazeEstimator()
    result = estimator.estimate(left_eye, right_eye, left_iris, right_iris)
    if result.direction == GazeDirection.OFF_ROAD:
        trigger_alert()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (normalised eye-space coordinates, range 0–1)
# ---------------------------------------------------------------------------

#: Horizontal offset from centre beyond which gaze is considered lateral.
GAZE_HORIZONTAL_THRESHOLD: float = 0.35

#: Vertical offset from centre beyond which gaze is considered vertical.
GAZE_VERTICAL_THRESHOLD: float = 0.30

#: Consecutive off-road frames before distraction alert is raised.
DISTRACTION_CONSEC_FRAMES: int = 10


# ---------------------------------------------------------------------------
# Direction enum
# ---------------------------------------------------------------------------


class GazeDirection(str, Enum):
    """Discrete gaze direction classification.

    Attributes:
        FORWARD: Driver is looking toward the road ahead.
        LEFT: Driver is looking to the left of the road axis.
        RIGHT: Driver is looking to the right of the road axis.
        UP: Driver is looking upward (e.g., dashboard overhead).
        DOWN: Driver is looking downward (e.g., phone, centre console).
        OFF_ROAD: Gaze is significantly off-road in any direction.
        UNKNOWN: Insufficient landmark data to estimate gaze.
    """

    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    OFF_ROAD = "OFF_ROAD"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GazeResult:
    """Per-frame gaze estimation result.

    Attributes:
        direction: Discrete :class:`GazeDirection` classification.
        horizontal_ratio: Normalised iris offset in the horizontal axis.
            Range [0, 1] where 0.5 is centred. Values near 0 indicate
            extreme left gaze; values near 1 indicate extreme right gaze.
        vertical_ratio: Normalised iris offset in the vertical axis.
            Range [0, 1] where 0.5 is centred.
        left_gaze: (h_ratio, v_ratio) for the left eye.
        right_gaze: (h_ratio, v_ratio) for the right eye.
        is_off_road: True when gaze is significantly off the forward axis.
        confidence: Estimation confidence [0, 1]. Lower when iris data
            is unavailable and the fallback method is used.
    """

    direction: GazeDirection = GazeDirection.UNKNOWN
    horizontal_ratio: float = 0.5
    vertical_ratio: float = 0.5
    left_gaze: Tuple[float, float] = (0.5, 0.5)
    right_gaze: Tuple[float, float] = (0.5, 0.5)
    is_off_road: bool = False
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compute_iris_ratio(
    eye: List[Tuple[float, float]],
    iris: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """Computes the normalised iris position within the eye bounding box.

    The iris centre is projected into a coordinate system defined by the
    eye corners (p1=outer-left, p4=outer-right for horizontal; bounding box
    top/bottom for vertical), yielding values in [0, 1].

    Args:
        eye: Six (x, y) landmark tuples for the eye region.
        iris: Five (x, y) landmark tuples for the iris contour.

    Returns:
        Tuple[float, float]: (horizontal_ratio, vertical_ratio) in [0, 1].
            Returns (0.5, 0.5) on invalid input.
    """
    if not eye or not iris:
        return 0.5, 0.5

    try:
        eye_arr = np.array(eye, dtype=np.float32)
        iris_arr = np.array(iris, dtype=np.float32)
        iris_centre = iris_arr.mean(axis=0)  # (cx, cy)

        eye_min = eye_arr.min(axis=0)  # (x_min, y_min)
        eye_max = eye_arr.max(axis=0)  # (x_max, y_max)
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
    """Estimates horizontal gaze from the midpoint of vertical landmarks.

    Used as fallback when iris data is unavailable. Measures the horizontal
    position of the vertical midpoint relative to the eye bounding box.

    Args:
        eye: Six (x, y) landmark tuples for the eye region.

    Returns:
        float: Horizontal ratio in [0, 1]. Returns 0.5 on error.
    """
    if len(eye) < 6:
        return 0.5
    try:
        # Midpoint of upper-inner and lower-inner vertical pair
        mid_x = (eye[2][0] + eye[4][0]) / 2.0
        left_x = eye[0][0]
        right_x = eye[3][0]
        width = right_x - left_x
        if width < 1e-6:
            return 0.5
        return float(np.clip((mid_x - left_x) / width, 0.0, 1.0))
    except Exception:
        return 0.5


def _classify_direction(
    h_ratio: float,
    v_ratio: float,
) -> GazeDirection:
    """Maps normalised gaze ratios to a discrete :class:`GazeDirection`.

    Args:
        h_ratio: Normalised horizontal iris position in [0, 1].
        v_ratio: Normalised vertical iris position in [0, 1].

    Returns:
        GazeDirection: Classified gaze direction.
    """
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


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class GazeEstimator:
    """Stateless gaze estimator that wraps iris-based and fallback methods.

    Prefers iris-based estimation when iris landmarks are available;
    falls back to the eye-corner-only method otherwise.

    Typical usage::

        estimator = GazeEstimator()
        result = estimator.estimate(
            left_eye=result.left_eye,
            right_eye=result.right_eye,
            left_iris=result.left_iris,
            right_iris=result.right_iris,
        )
    """

    def estimate(
        self,
        left_eye: List[Tuple[float, float]],
        right_eye: List[Tuple[float, float]],
        left_iris: Optional[List[Tuple[float, float]]] = None,
        right_iris: Optional[List[Tuple[float, float]]] = None,
    ) -> GazeResult:
        """Estimates gaze direction from eye and optional iris landmarks.

        Args:
            left_eye: Six (x, y) landmark tuples for the left eye.
            right_eye: Six (x, y) landmark tuples for the right eye.
            left_iris: Optional five (x, y) iris landmark tuples (left eye).
            right_iris: Optional five (x, y) iris landmark tuples (right eye).

        Returns:
            GazeResult: Estimated gaze with direction, ratios, and flags.
        """
        if not left_eye or not right_eye:
            return GazeResult(direction=GazeDirection.UNKNOWN, confidence=0.0)

        has_iris = bool(left_iris and right_iris)

        if has_iris:
            left_h, left_v = _compute_iris_ratio(left_eye, left_iris)  # type: ignore[arg-type]
            right_h, right_v = _compute_iris_ratio(right_eye, right_iris)  # type: ignore[arg-type]
            confidence = 0.95
        else:
            # Fallback: use eye-corner midpoint method (horizontal only)
            left_h = _fallback_horizontal_ratio(left_eye)
            right_h = _fallback_horizontal_ratio(right_eye)
            left_v = right_v = 0.5  # vertical unknown
            confidence = 0.50

        # Average both eyes
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


# ---------------------------------------------------------------------------
# Distraction tracker
# ---------------------------------------------------------------------------


class DistractionTracker:
    """Stateful tracker that accumulates off-road gaze events per session.

    Attributes:
        total_distraction_events: Count of confirmed distraction events.
        consec_off_road: Current consecutive off-road frame counter.
        total_off_road_frames: Total frames where gaze was off-road.
        gaze_history: Rolling list of recent :class:`GazeDirection` values.
    """

    def __init__(
        self,
        distraction_consec_frames: int = DISTRACTION_CONSEC_FRAMES,
        history_size: int = 300,
    ) -> None:
        """Initialises the distraction tracker.

        Args:
            distraction_consec_frames: Consecutive off-road frames required
                to confirm a distraction event.
            history_size: Maximum number of gaze samples to retain.
        """
        self._distraction_consec_frames = distraction_consec_frames
        self._history_size = history_size
        self.total_distraction_events: int = 0
        self.consec_off_road: int = 0
        self.total_off_road_frames: int = 0
        self.gaze_history: List[GazeDirection] = []

    def update(self, gaze_result: GazeResult) -> GazeResult:
        """Processes a new gaze result and updates distraction state.

        Args:
            gaze_result: :class:`GazeResult` for the current frame.

        Returns:
            GazeResult: The same result (is_off_road may be updated).
        """
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
        """Computes the fraction of frames where gaze was off-road.

        Args:
            total_frames: Total frames processed in the session.

        Returns:
            float: Distraction rate in [0, 1]. Returns 0.0 if no frames.
        """
        if total_frames <= 0:
            return 0.0
        return min(1.0, self.total_off_road_frames / total_frames)

    def reset(self) -> None:
        """Resets all counters and history for a new session."""
        self.total_distraction_events = 0
        self.consec_off_road = 0
        self.total_off_road_frames = 0
        self.gaze_history.clear()
