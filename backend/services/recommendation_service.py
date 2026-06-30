"""RecommendationService — Service layer coordinating advisory and safety recommendation lifecycle.

Handles active recommendations querying, status changes (acknowledge, dismiss), 
and SQLite persistence operations.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from backend.database.models.recommendations import Recommendation, RecommendationStatus

logger = logging.getLogger("CogniDrive.RecommendationService")


class RecommendationService:
    """Service layer class for managing advisory safety recommendations."""

    def __init__(self, db: Session) -> None:
        """Initializes the service with a database session."""
        self._db = db

    def get_active_recommendations(self, driver_id: int) -> List[Recommendation]:
        """Retrieves all pending or currently displayed recommendations for a driver.

        Excludes expired, acknowledged or dismissed entries.
        """
        now = datetime.now(timezone.utc)
        recs = (
            self._db.query(Recommendation)
            .filter(
                Recommendation.driver_id == driver_id,
                Recommendation.status.in_([
                    RecommendationStatus.PENDING,
                    RecommendationStatus.DISPLAYED,
                ]),
            )
            .all()
        )

        active = []
        for r in recs:
            # Check expiry
            if r.expires_at and r.expires_at.replace(tzinfo=timezone.utc) < now:
                r.expire()
            else:
                r.mark_as_displayed()
                active.append(r)

        self._db.commit()
        return active

    def acknowledge_recommendation(self, recommendation_id: int) -> Optional[Recommendation]:
        """Marks a recommendation as acknowledged by the driver."""
        rec = self._db.get(Recommendation, recommendation_id)
        if rec:
            rec.acknowledge()
            self._db.commit()
            logger.info("Recommendation %d acknowledged.", recommendation_id)
        return rec

    def dismiss_recommendation(self, recommendation_id: int) -> Optional[Recommendation]:
        """Marks a recommendation as dismissed by the driver."""
        rec = self._db.get(Recommendation, recommendation_id)
        if rec:
            rec.dismiss()
            self._db.commit()
            logger.info("Recommendation %d dismissed.", recommendation_id)
        return rec

    def delete_recommendation(self, recommendation_id: int) -> bool:
        """Soft-deletes a recommendation."""
        rec = self._db.get(Recommendation, recommendation_id)
        if rec:
            rec.soft_delete()
            self._db.commit()
            return True
        return False
