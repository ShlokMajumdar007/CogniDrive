"""PredictionService — Service layer coordinating driver state and risk predictions.

Acts as the service layer handling real-time camera frame inference, historical 
predictions lookup, and pipeline synchronization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from sqlalchemy.orm import Session

# Project imports with fallback
try:
    from backend.app.dependencies import get_pipeline_manager
    from backend.database.models.driving_metrics import DrivingMetric
except ImportError:
    from app.dependencies import get_pipeline_manager  # type: ignore[no-redef]
    from database.models.driving_metrics import DrivingMetric  # type: ignore[no-redef]

logger = logging.getLogger("CogniDrive.PredictionService")


class PredictionService:
    """Service layer class for prediction and pipeline tasks."""

    def __init__(self, db: Session) -> None:
        """Initializes the service with a database session."""
        self._db = db
        # PipelineManager is a singleton fetched via get_pipeline_manager
        try:
            self._pipeline_manager = get_pipeline_manager()
        except Exception as exc:
            logger.error("Failed to load pipeline manager in PredictionService: %s", exc)
            self._pipeline_manager = None

    def process_realtime_frame(
        self,
        driver_id: int,
        session_id: int,
        frame: np.ndarray,
        frame_number: int,
        frame_time_ms: float,
        telemetry: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Submits a single camera frame to the PipelineManager for full inference.

        Args:
            driver_id: Primary key of the driver.
            session_id: Active session ID.
            frame: Numpy BGR frame array.
            frame_number: Sequential frame index.
            frame_time_ms: Timestamp in milliseconds.
            telemetry: Telemetry inputs.

        Returns:
            Dict containing predictions, metrics, recommendations and detected flags.
        """
        if self._pipeline_manager is None:
            raise RuntimeError("PipelineManager not initialized.")

        # Ensure pipeline is bound to this database session
        self._pipeline_manager.db_session = self._db
        self._pipeline_manager.active_driver_id = driver_id
        self._pipeline_manager.active_session_id = session_id

        return self._pipeline_manager.process_frame(
            frame=frame,
            frame_number=frame_number,
            frame_time_ms=frame_time_ms,
            telemetry=telemetry,
        )

    def get_session_prediction_history(self, session_id: int) -> List[Dict[str, Any]]:
        """Retrieves fine-grained time-series prediction metrics for a session.

        Args:
            session_id: Session ID.

        Returns:
            List of dictionaries containing predictions per frame.
        """
        metrics = (
            self._db.query(DrivingMetric)
            .filter(DrivingMetric.session_id == session_id)
            .order_by(DrivingMetric.frame_number.asc())
            .all()
        )
        return [m.to_dict() for m in metrics]
