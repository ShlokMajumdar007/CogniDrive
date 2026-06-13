"""Thread-safe camera capture manager for CogniDrive's Vision Pipeline.

Provides a singleton OpenCV camera reader that runs in a dedicated
background thread, continuously populating a shared latest-frame buffer.
All consumers (landmark extractor, dashboard streaming) read the most
recent frame without blocking the capture loop.

Design goals:
    - **Zero-copy latest frame**: The last captured frame is stored as a
      class attribute; consumers receive a reference — no queue overhead.
    - **Decoupled capture rate**: The capture thread targets a configurable
      FPS independently of downstream processing speed.
    - **Thread safety**: A ``threading.Lock`` guards the shared frame buffer.
    - **Singleton**: Only one camera instance exists system-wide.
    - **Graceful shutdown**: The capture thread exits cleanly when
      :meth:`CameraManager.stop` is called.

Typical usage::

    mgr = CameraManager.get_instance(camera_index=0, target_fps=30)
    mgr.start()
    frame = mgr.read_frame()
    mgr.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera state enum
# ---------------------------------------------------------------------------


class CameraState(str, Enum):
    """Lifecycle state of the :class:`CameraManager`.

    Attributes:
        IDLE: Camera has not been started.
        RUNNING: Capture thread is active and producing frames.
        STOPPED: Camera was started and subsequently stopped.
        ERROR: Camera encountered an unrecoverable error.
    """

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Camera statistics dataclass
# ---------------------------------------------------------------------------


@dataclass
class CameraStats:
    """Runtime statistics for the camera capture thread.

    Attributes:
        state: Current :class:`CameraState`.
        actual_fps: Measured frames captured per second over the last second.
        total_frames_captured: Cumulative frames captured since ``start()``.
        total_frames_dropped: Frames where ``cap.read()`` returned False.
        frame_width: Capture resolution width in pixels.
        frame_height: Capture resolution height in pixels.
        camera_index: OpenCV camera device index.
        uptime_seconds: Seconds since the capture thread started.
    """

    state: CameraState = CameraState.IDLE
    actual_fps: float = 0.0
    total_frames_captured: int = 0
    total_frames_dropped: int = 0
    frame_width: int = 0
    frame_height: int = 0
    camera_index: int = 0
    uptime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Camera Manager
# ---------------------------------------------------------------------------


class CameraManager:
    """Thread-safe singleton OpenCV camera capture manager.

    The manager starts a background daemon thread that continuously reads
    frames from the camera device. The latest frame is stored in
    :attr:`_latest_frame` and protected by :attr:`_frame_lock`.

    Attributes:
        _instance: Class-level singleton reference.
        _singleton_lock: Lock guarding singleton creation.
    """

    _instance: Optional["CameraManager"] = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        camera_index: int = 0,
        target_fps: float = 30.0,
        frame_width: int = 640,
        frame_height: int = 480,
        buffer_size: int = 1,
    ) -> None:
        """Initialises the camera manager configuration.

        Does NOT start the capture thread — call :meth:`start` explicitly.

        Args:
            camera_index: OpenCV camera device index (0 = default webcam).
            target_fps: Desired capture frame rate. The capture thread
                sleeps between frames to approximate this rate.
            frame_width: Requested capture width. May not be honoured by
                all cameras (OpenCV will use the closest supported value).
            frame_height: Requested capture height.
            buffer_size: OpenCV internal frame buffer size. Setting to 1
                minimises latency by discarding stale buffered frames.
        """
        self._camera_index = camera_index
        self._target_fps = target_fps
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._buffer_size = buffer_size

        self._cap: Optional[cv2.VideoCapture] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_lock: threading.Lock = threading.Lock()
        self._stop_event: threading.Event = threading.Event()

        self._latest_frame: Optional[np.ndarray] = None
        self._state: CameraState = CameraState.IDLE

        # Runtime statistics
        self._total_frames_captured: int = 0
        self._total_frames_dropped: int = 0
        self._start_time: Optional[float] = None
        self._fps_counter: int = 0
        self._fps_timer: float = time.monotonic()
        self._actual_fps: float = 0.0

        logger.info(
            "CameraManager configured — index=%d, target_fps=%.1f, "
            "resolution=%dx%d",
            camera_index,
            target_fps,
            frame_width,
            frame_height,
        )

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        camera_index: int = 0,
        target_fps: float = 30.0,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> "CameraManager":
        """Returns the singleton :class:`CameraManager` instance.

        Args:
            camera_index: Forwarded to ``__init__`` on first call.
            target_fps: Forwarded to ``__init__`` on first call.
            frame_width: Forwarded to ``__init__`` on first call.
            frame_height: Forwarded to ``__init__`` on first call.

        Returns:
            CameraManager: Shared singleton instance.
        """
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls(
                        camera_index=camera_index,
                        target_fps=target_fps,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
        return cls._instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Opens the camera device and starts the background capture thread.

        Returns:
            bool: True if the camera opened successfully and the thread
                was started; False if the camera could not be opened.

        Raises:
            RuntimeError: If called while the camera is already running.
        """
        if self._state == CameraState.RUNNING:
            logger.warning("CameraManager.start() called while already RUNNING.")
            return True

        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            logger.error(
                "Failed to open camera device at index %d.", self._camera_index
            )
            self._state = CameraState.ERROR
            return False

        # Apply requested resolution and buffer size
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

        # Read back actual resolution (camera may not support requested size)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != self._frame_width or actual_h != self._frame_height:
            logger.warning(
                "Camera resolution adjusted: requested %dx%d, got %dx%d.",
                self._frame_width, self._frame_height, actual_w, actual_h,
            )
            self._frame_width = actual_w
            self._frame_height = actual_h

        self._stop_event.clear()
        self._start_time = time.monotonic()
        self._state = CameraState.RUNNING

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="CogniDrive-CameraCapture",
            daemon=True,
        )
        self._capture_thread.start()

        logger.info(
            "CameraManager started — camera=%d, resolution=%dx%d, target_fps=%.1f",
            self._camera_index,
            self._frame_width,
            self._frame_height,
            self._target_fps,
        )
        return True

    def stop(self) -> None:
        """Signals the capture thread to exit and releases camera resources.

        Blocks until the capture thread terminates (max 3 seconds).
        """
        if self._state != CameraState.RUNNING:
            return

        self._stop_event.set()

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=3.0)
            if self._capture_thread.is_alive():
                logger.warning("Capture thread did not exit cleanly within 3s.")

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._state = CameraState.STOPPED
        logger.info("CameraManager stopped.")

    def restart(self) -> bool:
        """Stops and restarts the camera capture thread.

        Returns:
            bool: True if restart succeeded.
        """
        logger.info("CameraManager restarting.")
        self.stop()
        return self.start()

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read_frame(self) -> Optional[np.ndarray]:
        """Returns the most recently captured camera frame.

        This is a non-blocking read. Returns ``None`` if no frame has been
        captured yet or if the camera is not running.

        Returns:
            Optional[np.ndarray]: Latest BGR frame array or ``None``.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def read_frame_nocopy(self) -> Optional[np.ndarray]:
        """Returns the latest frame WITHOUT copying (zero-copy read).

        The caller must NOT modify the returned array. Intended for
        read-only consumers such as the MJPEG dashboard stream.

        Returns:
            Optional[np.ndarray]: Latest BGR frame reference or ``None``.
        """
        with self._frame_lock:
            return self._latest_frame

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def stats(self) -> CameraStats:
        """Returns a snapshot of the current camera runtime statistics.

        Returns:
            CameraStats: Current statistics dataclass instance.
        """
        uptime = (
            time.monotonic() - self._start_time
            if self._start_time is not None
            else 0.0
        )
        return CameraStats(
            state=self._state,
            actual_fps=self._actual_fps,
            total_frames_captured=self._total_frames_captured,
            total_frames_dropped=self._total_frames_dropped,
            frame_width=self._frame_width,
            frame_height=self._frame_height,
            camera_index=self._camera_index,
            uptime_seconds=round(uptime, 2),
        )

    @property
    def is_running(self) -> bool:
        """Returns True if the capture thread is currently active."""
        return self._state == CameraState.RUNNING

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Returns (width, height) of the capture resolution."""
        return self._frame_width, self._frame_height

    # ------------------------------------------------------------------
    # Internal capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Background thread target — continuously reads frames from the camera.

        Runs until :attr:`_stop_event` is set or the camera fails. Stores
        each frame in :attr:`_latest_frame` under the frame lock and updates
        FPS statistics every second.
        """
        frame_interval = 1.0 / self._target_fps if self._target_fps > 0 else 0.0
        self._fps_timer = time.monotonic()
        self._fps_counter = 0

        logger.debug("Capture loop started (interval=%.4fs).", frame_interval)

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            if self._cap is None or not self._cap.isOpened():
                logger.error("Camera device closed unexpectedly. Stopping.")
                self._state = CameraState.ERROR
                break

            ret, frame = self._cap.read()

            if not ret or frame is None:
                self._total_frames_dropped += 1
                logger.debug(
                    "Frame read failed (drop #%d).", self._total_frames_dropped
                )
                # Brief sleep before retrying — avoids busy-wait on transient errors
                time.sleep(0.01)
                continue

            with self._frame_lock:
                self._latest_frame = frame

            self._total_frames_captured += 1
            self._fps_counter += 1

            # Update actual FPS every second
            now = time.monotonic()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._actual_fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_timer = now

            # Throttle to target FPS
            processing_time = time.monotonic() - loop_start
            sleep_time = frame_interval - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.debug("Capture loop exited.")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraManager":
        """Starts the camera on context manager entry."""
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Stops the camera on context manager exit."""
        self.stop()

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"CameraManager(index={self._camera_index}, "
            f"state={self._state.value}, "
            f"fps={self._actual_fps:.1f}/{self._target_fps:.1f}, "
            f"captured={self._total_frames_captured})"
        )
