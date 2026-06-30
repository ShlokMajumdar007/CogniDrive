"""RecommendationEngine — Personalized Driver Safety Recommendation Generator.

Analyzes active driver biometrics, cognitive state predictions, and risk scores 
against personalized thresholds to produce contextual, actionable safety recommendations.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from backend.database.models.recommendations import RecommendationType, PriorityLevel
from backend.app.constants import Recommendations

logger = logging.getLogger("CogniDrive.RecommendationEngine")


class RecommendationEngine:
    """Analyzes real-time biometrics and triggers safety recommendations.

    Matches current metric values against thresholds to produce tailored suggestions 
    for driver fatigue, distraction, cognitive overload, and high risk conditions.
    """

    def __init__(self) -> None:
        """Initializes the recommendation engine."""
        logger.info("RecommendationEngine initialized.")

    def generate_recommendations(
        self,
        driver_id: int,
        metrics: Dict[str, float],
        thresholds: Dict[str, float],
        confidence_score: float = 1.0,
    ) -> List[Dict[str, any]]:
        """Generates safety recommendations based on current metrics and active thresholds.

        Args:
            driver_id: Primary key of the driver.
            metrics: Current frame/window metric values (e.g. cli, fatigue_probability, etc.).
            thresholds: Personalized or global default thresholds.
            confidence_score: Base confidence score for the recommendation.

        Returns:
            List[Dict]: List of recommendation details ready for DB ingestion/schemas.
        """
        recommendations = []
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(minutes=15)  # Recommendations expire in 15 mins

        # 1. Critical Risk Check
        risk_score = metrics.get("risk_score", 0.0)
        critical_thresh = thresholds.get("risk_score_critical", 0.80)
        high_thresh = thresholds.get("risk_score_high", 0.60)

        if risk_score >= critical_thresh:
            recommendations.append({
                "driver_id": driver_id,
                "recommendation_type": RecommendationType.HIGH_RISK,
                "priority": PriorityLevel.CRITICAL,
                "title": "Critical Collision Risk Detected",
                "message": "Imminent collision or loss of control risk detected. Focus entirely on safety.",
                "explanation": f"Accident risk index ({risk_score:.2f}) has crossed critical threshold ({critical_thresh:.2f}).",
                "recommended_action": "Decelerate immediately, keep eyes on the road, and be prepared to stop.",
                "risk_score": risk_score,
                "confidence_score": confidence_score,
                "trigger_metric": "risk_score",
                "trigger_value": risk_score,
                "baseline_value": critical_thresh,
                "is_personalized": True,
                "expires_at": expiry,
            })
        elif risk_score >= high_thresh:
            recommendations.append({
                "driver_id": driver_id,
                "recommendation_type": RecommendationType.HIGH_RISK,
                "priority": PriorityLevel.HIGH,
                "title": "Elevated Accident Risk",
                "message": "Driving telemetry and biometrics indicate high accident risk.",
                "explanation": f"Accident risk index ({risk_score:.2f}) has crossed high threshold ({high_thresh:.2f}).",
                "recommended_action": "Reduce vehicle speed and check surroundings.",
                "risk_score": risk_score,
                "confidence_score": confidence_score,
                "trigger_metric": "risk_score",
                "trigger_value": risk_score,
                "baseline_value": high_thresh,
                "is_personalized": True,
                "expires_at": expiry,
            })

        # 2. Fatigue / Drowsiness Check
        fatigue_prob = metrics.get("fatigue_probability", 0.0)
        fatigue_thresh = thresholds.get("fatigue_probability", 0.60)
        perclos_val = metrics.get("perclos", 0.0)
        perclos_thresh = thresholds.get("perclos", 0.20)

        if fatigue_prob >= fatigue_thresh or perclos_val >= perclos_thresh:
            # Pull advice from constants
            advice_list = Recommendations.DROWSINESS_ADVICE
            advice = advice_list[0] if advice_list else "Take a safety break."
            
            trigger_metric = "fatigue_probability" if fatigue_prob >= fatigue_thresh else "perclos"
            trigger_val = fatigue_prob if trigger_metric == "fatigue_probability" else perclos_val
            base_val = fatigue_thresh if trigger_metric == "fatigue_probability" else perclos_thresh

            recommendations.append({
                "driver_id": driver_id,
                "recommendation_type": RecommendationType.FATIGUE,
                "priority": PriorityLevel.HIGH if fatigue_prob > 0.75 else PriorityLevel.MEDIUM,
                "title": "Driver Fatigue Detected",
                "message": "Biometrics indicate onset of severe drowsiness or micro-sleep patterns.",
                "explanation": f"Fatigue index ({trigger_val:.2f}) has exceeded safety threshold ({base_val:.2f}).",
                "recommended_action": advice,
                "risk_score": risk_score,
                "confidence_score": confidence_score,
                "trigger_metric": trigger_metric,
                "trigger_value": trigger_val,
                "baseline_value": base_val,
                "is_personalized": True,
                "expires_at": expiry,
            })

        # 3. Distraction Check
        distraction_prob = metrics.get("distraction_probability", 0.0)
        distraction_thresh = thresholds.get("distraction_probability", 0.60)

        if distraction_prob >= distraction_thresh:
            advice_list = Recommendations.DISTRACTION_ADVICE
            advice = advice_list[0] if advice_list else "Keep eyes on the road."

            recommendations.append({
                "driver_id": driver_id,
                "recommendation_type": RecommendationType.DISTRACTION,
                "priority": PriorityLevel.MEDIUM,
                "title": "Driver Distraction Warning",
                "message": "Continuous off-road gaze or distracted head pose detected.",
                "explanation": f"Distraction probability ({distraction_prob:.2f}) is above baseline limit ({distraction_thresh:.2f}).",
                "recommended_action": advice,
                "risk_score": risk_score,
                "confidence_score": confidence_score,
                "trigger_metric": "distraction_probability",
                "trigger_value": distraction_prob,
                "baseline_value": distraction_thresh,
                "is_personalized": True,
                "expires_at": expiry,
            })

        # 4. Cognitive Overload Check
        cli = metrics.get("cli", 0.0)
        cli_thresh = thresholds.get("cli", 70.0)

        if cli >= cli_thresh:
            advice_list = Recommendations.COGNITIVE_ADVICE
            advice = advice_list[0] if advice_list else "Reduce dashboard distractions."

            recommendations.append({
                "driver_id": driver_id,
                "recommendation_type": RecommendationType.STRESS,
                "priority": PriorityLevel.MEDIUM,
                "title": "High Cognitive Overload",
                "message": "High mental workload and physiological stress metrics detected.",
                "explanation": f"Cognitive Load Index ({cli:.1f}) has crossed overload limit ({cli_thresh:.1f}).",
                "recommended_action": advice,
                "risk_score": risk_score,
                "confidence_score": confidence_score,
                "trigger_metric": "cli",
                "trigger_value": cli,
                "baseline_value": cli_thresh,
                "is_personalized": True,
                "expires_at": expiry,
            })

        return recommendations
