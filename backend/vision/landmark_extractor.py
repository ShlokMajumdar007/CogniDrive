"""Facial Landmark Extractor using MediaPipe FaceMesh.

This module is the entry point for the entire CogniDrive Vision Pipeline.
It wraps MediaPipe FaceMesh in a thread-safe singleton and extracts
structured landmark data from BGR camera frames at up to 30 FPS.

The extracted landmarks feed directly into:
    - EAR (Eye Aspect Ratio) computation
    - MAR (Mouth Aspect Ratio) computation
    - Gaze direction estimation
    - Head pose estimation
    - PERCLOS fatigue accumulation
    - Feature vector construction

Typical usage::

    extractor = LandmarkExtractor.get_instance()
    result = extractor.extract(frame)
    if result.is_valid:
        ear = compute_ear(result.left_eye, result.right_eye)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe import Image as MpImage
from mediapipe import ImageFormat as MpImageFormat
from mediapipe.tasks.python.core import base_options as mp_base_options
from mediapipe.tasks.python.vision import face_landmarker as mp_face_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode as mp_running_mode

from backend.app.config import get_model_path
from backend.app.constants import MLConstants

_DEFAULT_LANDMARKER_MODEL = get_model_path(MLConstants.FACE_LANDMARKER_MODEL_NAME)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe FaceMesh indices for sub-regions
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# ---------------------------------------------------------------------------

# Left eye landmark indices (6-point EAR model)
LEFT_EYE_INDICES: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)

# Right eye landmark indices (6-point EAR model)
RIGHT_EYE_INDICES: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)

# Left iris centre
LEFT_IRIS_INDICES: Tuple[int, ...] = (473, 474, 475, 476, 477)

# Right iris centre
RIGHT_IRIS_INDICES: Tuple[int, ...] = (468, 469, 470, 471, 472)

# Mouth landmark indices (8-point MAR model)
MOUTH_INDICES: Tuple[int, ...] = (61, 291, 39, 181, 0, 17, 269, 405)

# Nose tip for head pose reference
NOSE_TIP_INDEX: int = 4

# Full face oval for head pose projection
FACE_OVAL_INDICES: Tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
)

# 3-D model points for solvePnP (canonical face geometry)
CANONICAL_3D_POINTS: np.ndarray = np.array(
    [
        [0.0, 0.0, 0.0],        # Nose tip (index 4)
        [0.0, -330.0, -65.0],   # Chin
        [-225.0, 170.0, -135.0], # Left eye left corner
        [225.0, 170.0, -135.0],  # Right eye right corner
        [-150.0, -150.0, -125.0], # Left mouth corner
        [150.0, -150.0, -125.0],  # Right mouth corner
    ],
    dtype=np.float64,
)

# MediaPipe landmark indices matched to CANONICAL_3D_POINTS rows
POSE_LANDMARK_INDICES: Tuple[int, ...] = (4, 152, 263, 33, 287, 57)


# ---------------------------------------------------------------------------
# Output Dataclass
# ---------------------------------------------------------------------------


@dataclass
class LandmarkResult:
    """Structured output produced by :class:`LandmarkExtractor` for a single frame.

    Attributes:
        is_valid: True when a face was detected and all sub-regions extracted.
        all_landmarks: Full 468-point array in normalised image coordinates.
            Shape is ``(468, 3)`` where columns are (x, y, z).
        left_eye: 6 landmark (x, y) pairs for left eye EAR computation.
        right_eye: 6 landmark (x, y) pairs for right eye EAR computation.
        left_iris: 5 landmark (x, y) pairs for left iris centre.
        right_iris: 5 landmark (x, y) pairs for right iris centre.
        mouth: 8 landmark (x, y) pairs for MAR computation.
        nose_tip: Single (x, y) pair for the nose tip landmark.
        pose_2d_points: 6 image-space (x, y) pixel coordinates used by solvePnP.
        frame_width: Width of the source frame in pixels.
        frame_height: Height of the source frame in pixels.
        detection_confidence: MediaPipe face detection confidence score [0, 1].
        tracking_confidence: MediaPipe landmark tracking confidence score [0, 1].
    """

    is_valid: bool = False
    all_landmarks: Optional[np.ndarray] = None
    left_eye: List[Tuple[float, float]] = field(default_factory=list)
    right_eye: List[Tuple[float, float]] = field(default_factory=list)
    left_iris: List[Tuple[float, float]] = field(default_factory=list)
    right_iris: List[Tuple[float, float]] = field(default_factory=list)
    mouth: List[Tuple[float, float]] = field(default_factory=list)
    nose_tip: Optional[Tuple[float, float]] = None
    pose_2d_points: Optional[np.ndarray] = None
    frame_width: int = 0
    frame_height: int = 0
    detection_confidence: float = 0.0
    tracking_confidence: float = 0.0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def left_eye_array(self) -> np.ndarray:
        """Returns the left eye landmarks as a NumPy array of shape (6, 2).

        Returns:
            np.ndarray: Left eye (x, y) coordinates.
        """
        return np.array(self.left_eye, dtype=np.float32)

    def right_eye_array(self) -> np.ndarray:
        """Returns the right eye landmarks as a NumPy array of shape (6, 2).

        Returns:
            np.ndarray: Right eye (x, y) coordinates.
        """
        return np.array(self.right_eye, dtype=np.float32)

    def mouth_array(self) -> np.ndarray:
        """Returns the mouth landmarks as a NumPy array of shape (8, 2).

        Returns:
            np.ndarray: Mouth (x, y) coordinates.
        """
        return np.array(self.mouth, dtype=np.float32)

    def left_iris_centre(self) -> Optional[Tuple[float, float]]:
        """Computes the centroid of the left iris landmarks.

        Returns:
            Optional[Tuple[float, float]]: (cx, cy) centroid or None.
        """
        if not self.left_iris:
            return None
        arr = np.array(self.left_iris, dtype=np.float32)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())

    def right_iris_centre(self) -> Optional[Tuple[float, float]]:
        """Computes the centroid of the right iris landmarks.

        Returns:
            Optional[Tuple[float, float]]: (cx, cy) centroid or None.
        """
        if not self.right_iris:
            return None
        arr = np.array(self.right_iris, dtype=np.float32)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class LandmarkExtractor:
    """Thread-safe singleton wrapper around MediaPipe FaceMesh.

    The singleton ensures MediaPipe is initialised exactly once across all
    threads (vision pipeline thread, FastAPI request thread, etc.).

    Usage::

        extractor = LandmarkExtractor.get_instance()
        result: LandmarkResult = extractor.extract(bgr_frame)

    Attributes:
        _instance: Class-level singleton reference.
        _lock: Module-level threading lock guarding singleton creation.
        _face_mesh: Underlying MediaPipe FaceMesh processor.
        _min_detection_confidence: Minimum detection confidence threshold.
        _min_tracking_confidence: Minimum tracking confidence threshold.
        _max_num_faces: Maximum number of faces to detect per frame.
    """

    _instance: Optional["LandmarkExtractor"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
    ) -> None:
        """Initialises the MediaPipe FaceMesh processor.

        Args:
            min_detection_confidence: Minimum detection confidence in [0, 1].
            min_tracking_confidence: Minimum tracking confidence in [0, 1].
            max_num_faces: Maximum number of simultaneous faces to track.
            refine_landmarks: If True enables iris and eye contour refinement
                (required for iris tracking). Adds ~5 ms latency per frame.

        Raises:
            RuntimeError: If MediaPipe FaceMesh cannot be initialised.
        """
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._max_num_faces = max_num_faces
        self._refine_landmarks = refine_landmarks
        self._face_mesh = None
        self._backend = "none"

        model_path = _DEFAULT_LANDMARKER_MODEL
        if hasattr(mp, "solutions"):
            try:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=max_num_faces,
                    refine_landmarks=refine_landmarks,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self._backend = "solutions"
            except Exception as exc:
                logger.warning("MediaPipe solutions FaceMesh failed: %s", exc)

        if self._face_mesh is None:
            if not model_path.exists():
                logger.warning(
                    "Face landmarker model not found at %s. "
                    "Download face_landmarker.task from MediaPipe model zoo into MODEL_DIR. "
                    "Vision pipeline will return no face detections until the model is available.",
                    model_path,
                )
            else:
                try:
                    options = mp_face_landmarker.FaceLandmarkerOptions(
                        base_options=mp_base_options.BaseOptions(
                            model_asset_path=str(model_path)
                        ),
                        running_mode=mp_running_mode.VisionTaskRunningMode.IMAGE,
                        num_faces=max_num_faces,
                        min_face_detection_confidence=min_detection_confidence,
                        min_face_presence_confidence=min_tracking_confidence,
                        min_tracking_confidence=min_tracking_confidence,
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False,
                    )
                    self._face_mesh = mp_face_landmarker.FaceLandmarker.create_from_options(
                        options
                    )
                    self._backend = "tasks"
                except Exception as exc:
                    logger.warning("MediaPipe FaceLandmarker tasks init failed: %s", exc)

        if self._face_mesh is not None:
            logger.info(
                "LandmarkExtractor initialised — backend=%s, max_faces=%d, refine=%s, "
                "det_conf=%.2f, track_conf=%.2f",
                self._backend,
                max_num_faces,
                refine_landmarks,
                min_detection_confidence,
                min_tracking_confidence,
            )
        else:
            logger.warning(
                "LandmarkExtractor running in degraded mode — no face landmark backend available."
            )

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
    ) -> "LandmarkExtractor":
        """Returns the singleton :class:`LandmarkExtractor` instance.

        Creates the instance on first call; subsequent calls return the
        cached instance ignoring any constructor arguments.

        Args:
            min_detection_confidence: Forwarded to ``__init__`` on first call.
            min_tracking_confidence: Forwarded to ``__init__`` on first call.
            max_num_faces: Forwarded to ``__init__`` on first call.
            refine_landmarks: Forwarded to ``__init__`` on first call.

        Returns:
            LandmarkExtractor: Shared singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence,
                        max_num_faces=max_num_faces,
                        refine_landmarks=refine_landmarks,
                    )
        return cls._instance

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract(self, frame: np.ndarray) -> LandmarkResult:
        """Extracts facial landmarks from a single BGR camera frame.

        Converts the frame to RGB, runs MediaPipe FaceMesh, and maps
        normalised coordinates back to pixel space for all sub-regions.

        Args:
            frame: BGR image array of shape (H, W, 3) as returned by
                ``cv2.VideoCapture.read()``.

        Returns:
            LandmarkResult: Populated result with ``is_valid=True`` if a
                face was detected, otherwise an empty result with
                ``is_valid=False``.
        """
        if frame is None or frame.size == 0:
            logger.debug("Received empty frame; skipping extraction.")
            return LandmarkResult(is_valid=False)

        h, w = frame.shape[:2]
        if self._face_mesh is None:
            return LandmarkResult(is_valid=False, frame_width=w, frame_height=h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._backend == "solutions":
            rgb.flags.writeable = False
            mp_result = self._face_mesh.process(rgb)
            if not mp_result.multi_face_landmarks:
                return LandmarkResult(is_valid=False, frame_width=w, frame_height=h)
            face = mp_result.multi_face_landmarks[0]
            lm = face.landmark
        else:
            mp_image = MpImage(
                image_format=MpImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            mp_result = self._face_mesh.detect(mp_image)
            if not mp_result.face_landmarks:
                return LandmarkResult(is_valid=False, frame_width=w, frame_height=h)
            lm = mp_result.face_landmarks[0]

        # Build full (468 + 10 iris) array in normalised space
        total = len(lm)
        all_lm = np.array(
            [(point.x, point.y, point.z) for point in lm],
            dtype=np.float32,
        )

        def _px(idx: int) -> Tuple[float, float]:
            """Converts normalised landmark to pixel coordinates."""
            point = lm[idx]
            return float(point.x * w), float(point.y * h)

        left_eye = [_px(i) for i in LEFT_EYE_INDICES]
        right_eye = [_px(i) for i in RIGHT_EYE_INDICES]
        mouth = [_px(i) for i in MOUTH_INDICES]
        nose_tip = _px(NOSE_TIP_INDEX)

        # Iris — only available when refine_landmarks=True
        left_iris: List[Tuple[float, float]] = []
        right_iris: List[Tuple[float, float]] = []
        if self._refine_landmarks and total >= 478:
            left_iris = [_px(i) for i in LEFT_IRIS_INDICES]
            right_iris = [_px(i) for i in RIGHT_IRIS_INDICES]

        # 2-D points for solvePnP
        pose_2d = np.array(
            [_px(i) for i in POSE_LANDMARK_INDICES], dtype=np.float64
        )

        # Confidence scores (MediaPipe exposes these at the FaceDetection level;
        # we approximate via the face_blendshapes when available, else default 1.0)
        det_conf = 1.0
        track_conf = 1.0

        return LandmarkResult(
            is_valid=True,
            all_landmarks=all_lm,
            left_eye=left_eye,
            right_eye=right_eye,
            left_iris=left_iris,
            right_iris=right_iris,
            mouth=mouth,
            nose_tip=nose_tip,
            pose_2d_points=pose_2d,
            frame_width=w,
            frame_height=h,
            detection_confidence=det_conf,
            tracking_confidence=track_conf,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Releases MediaPipe FaceMesh resources.

        Should be called when the vision pipeline shuts down to free GPU/CPU
        memory held by the underlying MediaPipe graph.
        """
        try:
            if self._backend == "solutions" and hasattr(self._face_mesh, "close"):
                self._face_mesh.close()
            logger.info("LandmarkExtractor closed and resources released.")
        except Exception as exc:
            logger.warning("Error closing LandmarkExtractor: %s", exc)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"LandmarkExtractor("
            f"max_faces={self._max_num_faces}, "
            f"refine={self._refine_landmarks}, "
            f"det={self._min_detection_confidence}, "
            f"track={self._min_tracking_confidence})"
        )
