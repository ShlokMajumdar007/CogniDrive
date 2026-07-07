"""Head pose estimation using solvePnP for roll, pitch, and yaw angles.

Calculates the driver's head orientation in 3D space relative to the camera.
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

# 3D model points — canonical face geometry
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

# Angle thresholds (degrees)
YAW_DISTRACTION_THRESHOLD: float = 30.0
PITCH_DOWN_THRESHOLD: float = -20.0
PITCH_UP_THRESHOLD: float = 25.0
ROLL_THRESHOLD: float = 20.0


class HeadOrientation(str, Enum):
    """Discrete head orientation directions."""
    FORWARD = "FORWARD"
    LOOKING_LEFT = "LOOKING_LEFT"
    LOOKING_RIGHT = "LOOKING_RIGHT"
    LOOKING_DOWN = "LOOKING_DOWN"
    LOOKING_UP = "LOOKING_UP"
    TILTED = "TILTED"
    UNKNOWN = "UNKNOWN"


@dataclass
class HeadPoseResult:
    """Head pose estimation outputs for a single frame."""
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    orientation: HeadOrientation = HeadOrientation.UNKNOWN
    is_distracted: bool = False
    rotation_vector: Optional[np.ndarray] = None
    translation_vector: Optional[np.ndarray] = None
    success: bool = False


class HeadPoseEstimator:
    """Estimates head angles from 6 2D facial landmarks using solvePnP."""

    def __init__(self, frame_width: int = 640, frame_height: int = 480) -> None:
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._camera_matrix, self._dist_coeffs = self._build_camera_matrix(
            frame_width, frame_height
        )
        logger.debug(
            "HeadPoseEstimator initialized for %dx%d frame.", frame_width, frame_height
        )

    @staticmethod
    def _build_camera_matrix(w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
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
        """Updates internal camera matrix parameters if resolution changes."""
        if frame_width != self._frame_width or frame_height != self._frame_height:
            self._frame_width = frame_width
            self._frame_height = frame_height
            self._camera_matrix, self._dist_coeffs = self._build_camera_matrix(
                frame_width, frame_height
            )

    def estimate(self, pose_2d_points: np.ndarray) -> HeadPoseResult:
        """Calculates yaw, pitch, and roll from landmark coordinates."""
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

        rot_mat, _ = cv2.Rodrigues(rvec)
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
        """Calculates ZYX Euler convention angles from rotation matrix."""
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
    def _classify_orientation(pitch: float, yaw: float, roll: float) -> HeadOrientation:
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
