"""DashboardService — Service layer coordinating dashboard aggregation and metrics analytics.

Computes session statistics, event counts (fatigue, distraction, lane departures), 
average scores, and compiles aggregated analytics for the HMI cockpit dashboard.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database.models.session_data import SessionData, SessionStatus
from backend.database.models.driving_metrics import DrivingMetric
from backend.database.models.driver_profile import DriverProfile

logger = logging.getLogger("CogniDrive.DashboardService")


class DashboardService:
    """Service layer class for aggregating session data for HMI cockpit dashboard visualization."""

    def __init__(self, db: Session) -> None:
        """Initializes the service with a database session."""
        self._db = db

    def get_session_summary(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Returns aggregated session statistics for the dashboard.

        Compiles averages (attention, stress, risk, cli) and event sums 
        from the DrivingMetric records of a session.
        """
        session = self._db.get(SessionData, session_id)
        if not session:
            return None

        # Aggregate frame-level metrics
        aggregates = (
            self._db.query(
                func.avg(DrivingMetric.attention_score).label("avg_attention"),
                func.avg(DrivingMetric.stress_score).label("avg_stress"),
                func.avg(DrivingMetric.cli).label("avg_cli"),
                func.avg(DrivingMetric.risk_score).label("avg_risk"),
                func.avg(DrivingMetric.speed).label("avg_speed"),
                func.max(DrivingMetric.speed).label("max_speed"),
            )
            .filter(DrivingMetric.session_id == session_id)
            .one_or_none()
        )

        # Update SessionData columns before serialization if session is active
        if session.status == SessionStatus.ACTIVE and aggregates:
            session.avg_attention_score = float(aggregates.avg_attention or 0.0)
            session.avg_stress_score = float(aggregates.avg_stress or 0.0)
            session.avg_cli = float(aggregates.avg_cli or 0.0)
            session.avg_risk_score = float(aggregates.avg_risk or 0.0)
            session.avg_speed = float(aggregates.avg_speed or 0.0)
            session.max_speed = float(aggregates.max_speed or 0.0)
            session.calculate_duration()
            self._db.commit()

        return session.to_dict()

    def get_driver_historical_summary(self, driver_id: int) -> Optional[Dict[str, Any]]:
        """Aggregates historical totals and baselines from the DriverProfile.

        Useful for general statistics display on HMI dashboard loading.
        """
        driver = self._db.get(DriverProfile, driver_id)
        if not driver:
            return None

        summary = driver.to_dict()
        summary["calculated_overall_risk"] = driver.calculate_driver_risk_factor()
        return summary
