from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional
import numpy as np
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# PERCLOS configuration constants
# ---------------------------------------------------------------------------
PERCLOS_CLOSURE_RATIO: float = 0.80  # eye ≥ 80% closed when EAR < 0.20
PERCLOS_EAR_CLOSED_THRESHOLD: float = 0.20
DEFAULT_WINDOW_SECONDS: int = 60
DEFAULT_FPS: float = 30.0
# ---------------------------------------------------------------------------
# Alert thresholds
# ---------------------------------------------------------------------------
#: PERCLOS above this triggers a DROWSY alert.
PERCLOS_DROWSY_THRESHOLD: float = 0.15
#: PERCLOS above this triggers a SEVERELY_DROWSY alert.
PERCLOS_SEVERE_THRESHOLD: float = 0.30
# ---------------------------------------------------------------------------
# Drowsiness state enum
# ---------------------------------------------------------------------------
class DrowsinessState(str, Enum):
    ALERT = "ALERT"
    DROWSY = "DROWSY"
    SEVERELY_DROWSY = "SEVERELY_DROWSY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class PERCLOSResult:
    perclos: float = 0.0
    state: DrowsinessState = DrowsinessState.INSUFFICIENT_DATA
    window_frames: int = 0
    closed_frames: int = 0
    mean_ear: float = 0.0
    min_ear: float = 0.0
    fatigue_probability: float = 0.0
    is_drowsy: bool = False
# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------
class PERCLOSCalculator:
    def __init__(
        self,
        fps: float = DEFAULT_FPS,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        ear_closed_threshold: float = PERCLOS_EAR_CLOSED_THRESHOLD,
    ) -> None:
        self._fps = fps
        self._window_seconds = window_seconds
        self._ear_closed_threshold = ear_closed_threshold
        self._window_size: int = max(1, int(fps * window_seconds))

        self._ear_window: Deque[float] = deque(maxlen=self._window_size)
        self._closed_count: int = 0
        self._total_frames_processed: int = 0

        logger.info(
            "PERCLOSCalculator initialised — fps=%.1f, window=%ds (%d frames), "
            "closure_threshold=%.2f",
            fps,
            window_seconds,
            self._window_size,
            ear_closed_threshold,
        )

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------
    def update(self, ear_value: float) -> PERCLOSResult:
        self._total_frames_processed += 1
        is_closed = ear_value < self._ear_closed_threshold

        # Evict the oldest sample if the window is full
        if len(self._ear_window) == self._window_size:
            evicted = self._ear_window[0]
            if evicted < self._ear_closed_threshold:
                self._closed_count -= 1

        self._ear_window.append(ear_value)
        if is_closed:
            self._closed_count += 1

        return self._compute_result()

    def _compute_result(self) -> PERCLOSResult:
        window_len = len(self._ear_window)
        if window_len < int(self._fps * 5):
            # Need at least 5 seconds of data for meaningful PERCLOS
            return PERCLOSResult(
                state=DrowsinessState.INSUFFICIENT_DATA,
                window_frames=window_len,
            )
        ear_arr = np.array(self._ear_window, dtype=np.float32)
        perclos = self._closed_count / window_len
        mean_ear = float(ear_arr.mean())
        min_ear = float(ear_arr.min())
        # Classify state
        if perclos >= PERCLOS_SEVERE_THRESHOLD:
            state = DrowsinessState.SEVERELY_DROWSY
        elif perclos >= PERCLOS_DROWSY_THRESHOLD:
            state = DrowsinessState.DROWSY
        else:
            state = DrowsinessState.ALERT
        # Map PERCLOS → fatigue_probability (linear, capped at 1.0)
        fatigue_prob = min(1.0, perclos / PERCLOS_SEVERE_THRESHOLD)
        return PERCLOSResult(
            perclos=round(perclos, 4),
            state=state,
            window_frames=window_len,
            closed_frames=self._closed_count,
            mean_ear=round(mean_ear, 4),
            min_ear=round(min_ear, 4),
            fatigue_probability=round(fatigue_prob, 4),
            is_drowsy=(state in (DrowsinessState.DROWSY, DrowsinessState.SEVERELY_DROWSY)),
        )
    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------
    def update_batch(self, ear_values: List[float]) -> PERCLOSResult:
        result = PERCLOSResult(state=DrowsinessState.INSUFFICIENT_DATA)
        for ear in ear_values:
            result = self.update(ear)
        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def window_fill_ratio(self) -> float:
        return len(self._ear_window) / self._window_size

    @property
    def total_frames_processed(self) -> int:
        return self._total_frames_processed
    @property
    def window_size(self) -> int:
        return self._window_size
    def current_ear_history(self) -> List[float]:
        return list(self._ear_window)
    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._ear_window.clear()
        self._closed_count = 0
        self._total_frames_processed = 0
        logger.info("PERCLOSCalculator reset.")
#----------------------------------------------------------------------------