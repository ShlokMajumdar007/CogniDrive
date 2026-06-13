"""Head pose estimation using solvePnP for roll, pitch, and yaw angles.

Estimates the driver's head orientation in 3-D space by solving the
Perspective-n-Point (PnP) problem using 6 stable facial landmarks matched
to a canonical 3-D face model. The resulting Euler angles describe:

- **Pitch** (nodding up/down): Positive = looking up, Negative = looking down.
- **Yaw** (shaking left/right): Positive = looking right, Negative = looking left.
- **Roll** (tilting head sideways): Positive = tilt right, Negative = tilt left.

High pitch-down or large yaw angles indicate the driver is looking away from
the road — critical inputs to both the distraction and risk engines.

Typical usage::

    from backend.vision.head_pose import HeadPoseEstimator, HeadPoseResult
    estimator = HeadPoseEstimator(frame_width=640, frame_height=480)
    result = estimator.estimate(pose_2d_points)
    print(result.pitch, result.yaw, result.roll)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 3-D model points — canonical face geometry (identical to landmark_extractor.py)
_CANONICAL_3D: np.ndarray = np.array(
    [
        [0.0, 0.0, 0.0],          # Nose tip
        [0.0, -330.0, -65.0],     # Chin
        [-225.0, 170.0, -135.0],  # Left eye outer corner
        [225.0, 170.0, -135.0],   # Right eye outer corner
        [-150.0, -150.0, -125.0], # Left mouth corner
        [150.0, -150.0, -125.0],  # Right mouth corner
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# Angle thresholds (degrees)
# ---------------------------------------------------------------------------

#: Yaw beyond this magnitude signals lateral gaze distraction.
YAW_DISTRACTION_THRESHOLD: float = 30.0

#: Pitch below this value (head down) signals downward gaze (phone, console).
PITCH_DOWN_THRESHOLD: float = -20.0

#: Pitch above this value (head up) is unusual and may indicate microsleep.
PITCH_UP_THRESHOLD: float = 25.0

#: Roll beyond this magnitude signals head tilt (microsleep indicator).
ROLL_THRESHOLD: float = 20.0


# ---------------------------------------------------------------------------
# Head orientation enum
# ---------------------------------------------------------------------------


class HeadOrientation(str, Enum):
    """Discrete head orientation classification from pitch/yaw/roll angles.

    Attributes:
        FORWARD: Driver is looking straight ahead.
        LOOKING_LEFT: Significant leftward yaw.
        LOOKING_RIGHT: Significant rightward yaw.
        LOOKING_DOWN: Significant downward pitch (phone/console).
        LOOKING_UP: Significant upward pitch.
        TILTED: Significant roll (fatigue indicator).
        UNKNOWN: Pose estimation failed or landmarks invalid.
    """

    FORWARD = "FORWARD"
    LOOKING_LEFT = "LOOKING_LEFT"
    LOOKING_RIGHT = "LOOKING_RIGHT"
    LOOKING_DOWN = "LOOKING_DOWN"
    LOOKING_UP = "LOOKING_UP"
    TILTED = "TILTED"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HeadPoseResult:
    """Per-frame head pose estimation result.

    Attributes:
        pitch: Vertical rotation in degrees. Positive = up, Negative = down.
        yaw: Horizontal rotation in degrees. Positive = right, Negative = left.
        roll: Tilt rotation in degrees. Positive = tilt right, Negative = left.
        orientation: Discrete :class:`HeadOrientation` classification.
        is_distracted: True when yaw or pitch exceeds distraction thresholds.
        rotation_vector: Raw solvePnP rotation vector (3,).
        translation_vector: Raw solvePnP translation vector (3,).
        success: True when solvePnP converged successfully.
    """

    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    orientation: HeadOrientation = HeadOrientation.UNKNOWN
    is_distracted: bool = False
    rotation_vector: Optional[np.ndarray] = None
    translation_vector: Optional[np.ndarray] = None
    success: bool = False


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class HeadPoseEstimator:
    """Estimates head pose angles from 6 facial landmark 2-D image points.

    Uses OpenCV ``solvePnP`` with the ITERATIVE method and a pinhole camera
    model approximated from frame dimensions. For a 640×480 camera the focal
    length is estimated as ``max(width, height)`` pixels.

    The estimator is stateless — call :meth:`estimate` per frame.

    Attributes:
        _camera_matrix: 3×3 intrinsic camera matrix (pinhole approximation).
        _dist_coeffs: 4-element distortion coefficients (assumed zero).
        _frame_width: Frame width used to build the camera matrix.
        _frame_height: Frame height used to build the camera matrix.
    """

    def __init__(self, frame_width: int = 640, frame_height: int = 480) -> None:
        """Initialises the estimator with a pinhole camera approximation.

        Args:
            frame_width: Width of the camera frame in pixels.
            frame_height: Height of the camera frame in pixels.
        """
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._camera_matrix, self._dist_coeffs = self._build_camera_matrix(
            frame_width, frame_height
        )
        logger.debug(
            "HeadPoseEstimator initialised for %dx%d frame.", frame_width, frame_height
        )

    @staticmethod
    def _build_camera_matrix(
        w: int, h: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Builds an approximate pinhole camera intrinsic matrix.

        Args:
            w: Frame width in pixels.
            h: Frame height in pixels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (camera_matrix, dist_coeffs).
        """
        focal_length = float(max(w, h))
        cx, cy = w / 2.0, h / 2.0
        camera_matrix = np.array(
            [
                [focal_length, 0.0, cx],
                [0.0, focal_length, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        return camera_matrix, dist_coeffs

    def update_frame_size(self, frame_width: int, frame_height: int) -> None:
        """Updates the camera matrix when the frame dimensions change.

        Args:
            frame_width: New frame width in pixels.
            frame_height: New frame height in pixels.
        """
        if frame_width != self._frame_width or frame_height != self._frame_height:
            self._frame_width = frame_width
            self._frame_height = frame_height
            self._camera_matrix, self._dist_coeffs = self._build_camera_matrix(
                frame_width, frame_height
            )

    def estimate(self, pose_2d_points: np.ndarray) -> HeadPoseResult:
        """Estimates roll, pitch, and yaw from 6 facial 2-D landmarks.

        Args:
            pose_2d_points: Array of shape (6, 2) containing pixel-space
                (x, y) coordinates for the 6 landmarks that correspond to
                :data:`~backend.vision.landmark_extractor.CANONICAL_3D_POINTS`.

        Returns:
            HeadPoseResult: Estimated angles and orientation classification.
                ``success=False`` if solvePnP failed to converge.
        """
        if pose_2d_points is None or pose_2d_points.shape != (6, 2):
            logger.debug(
                "HeadPoseEstimator received invalid points shape: %s",
                getattr(pose_2d_points, "shape", None),
            )
            return HeadPoseResult(success=False)

        try:
            success, rvec, tvec = cv2.solvePnP(
                _CANONICAL_3D,
                pose_2d_points.astype(np.float64),
                self._camera_matrix,
                self._dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error as exc:
            logger.warning("solvePnP failed: %s", exc)
            return HeadPoseResult(success=False)

        if not success:
            return HeadPoseResult(success=False)

        # Convert rotation vector to rotation matrix
        rot_mat, _ = cv2.Rodrigues(rvec)

        # Decompose into Euler angles via RQ decomposition
        pitch, yaw, roll = self._rotation_matrix_to_euler(rot_mat)

        orientation = self._classify_orientation(pitch, yaw, roll)
        is_distracted = (
            abs(yaw) > YAW_DISTRACTION_THRESHOLD
            or pitch < PITCH_DOWN_THRESHOLD
            or pitch > PITCH_UP_THRESHOLD
        )

        return HeadPoseResult(
            pitch=round(pitch, 2),
            yaw=round(yaw, 2),
            roll=round(roll, 2),
            orientation=orientation,
            is_distracted=is_distracted,
            rotation_vector=rvec,
            translation_vector=tvec,
            success=True,
        )

    @staticmethod
    def _rotation_matrix_to_euler(rot_mat: np.ndarray) -> Tuple[float, float, float]:
        """Converts a 3×3 rotation matrix to Euler angles (degrees).

        Uses the atan2-based decomposition for ZYX Euler convention which
        is standard for head pose estimation. Handles the gimbal-lock
        singularity at pitch = ±90°.

        Args:
            rot_mat: 3×3 rotation matrix from ``cv2.Rodrigues``.

        Returns:
            Tuple[float, float, float]: (pitch_deg, yaw_deg, roll_deg).
        """
        sy = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
            pitch = math.atan2(-rot_mat[2, 0], sy)
            yaw = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
        else:
            roll = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
            pitch = math.atan2(-rot_mat[2, 0], sy)
            yaw = 0.0

        return (
            math.degrees(pitch),
            math.degrees(yaw),
            math.degrees(roll),
        )

    @staticmethod
    def _classify_orientation(
        pitch: float, yaw: float, roll: float
    ) -> HeadOrientation:
        """Maps Euler angles to a discrete :class:`HeadOrientation`.

        Args:
            pitch: Pitch angle in degrees.
            yaw: Yaw angle in degrees.
            roll: Roll angle in degrees.

        Returns:
            HeadOrientation: Classified orientation.
        """
        if abs(roll) > ROLL_THRESHOLD:
            return HeadOrientation.TILTED
        if pitch < PITCH_DOWN_THRESHOLD:
            return HeadOrientation.LOOKING_DOWN
        if pitch > PITCH_UP_THRESHOLD:
            return HeadOrientation.LOOKING_UP
        if yaw < -YAW_DISTRACTION_THRESHOLD:
            return HeadOrientation.LOOKING_LEFT
        if yaw > YAW_DISTRACTION_THRESHOLD:
            return HeadOrientation.LOOKING_RIGHT
        return HeadOrientation.FORWARD
