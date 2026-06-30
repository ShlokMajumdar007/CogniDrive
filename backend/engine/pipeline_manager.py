"""PipelineManager — Real-time frame processing and ML inference pipeline orchestrator.

Integrates MediaPipe landmark extraction, metric windowing, feature builder, 
normalizer, cognitive, risk, anomaly engines, personalization updates, and database logging.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session

# Project imports
from backend.vision.landmark_extractor import LandmarkExtractor, LandmarkResult
from backend.vision.ear import compute_ear, BlinkTracker
from backend.vision.mar import compute_mar, YawnTracker
from backend.vision.gaze import GazeEstimator, DistractionTracker
from backend.vision.head_pose import HeadPoseEstimator
from backend.vision.perclos import PERCLOSCalculator
from backend.features.feature_vector import FeatureVectorBuilder, RawSignals
from backend.features.normalizer import FeatureNormalizer, NormalizerStats
from backend.features.windowing import SignalWindowManager
from backend.ml.inference.cognitive_model import CognitiveResult
from backend.ml.inference.risk_model import RiskResult
from backend.ml.anomaly_detection.anomaly_engine import AnomalyEngine, AnomalyResult
from backend.ml.recommendation.recommendation_engine import RecommendationEngine
from backend.digital_twin.personalization import PersonalizationEngine, TrackedSignal
from backend.digital_twin.threshold_manager import ThresholdManager
from backend.engine.cognitive_load_engine import CognitiveLoadEngine
from backend.engine.accident_risk_engine import AccidentRiskEngine
from backend.engine.state_classifier import StateClassifier
from backend.database.models.driving_metrics import DrivingMetric, DriverState
from backend.database.models.session_data import SessionData
from backend.database.models.driver_profile import DriverProfile
from backend.database.models.recommendations import Recommendation

logger = logging.getLogger("CogniDrive.PipelineManager")


class PipelineManager:
    """Orchestrates real-time frame ingestion, vision features, ML scoring, and DT personalization."""

    def __init__(self, camera_manager: Any, settings: Any) -> None:
        """Initializes pipeline components and engines."""
        self.camera_manager = camera_manager
        self.settings = settings

        # Vision processing
        self.landmark_extractor = LandmarkExtractor.get_instance()
        self.blink_tracker = BlinkTracker()
        self.yawn_tracker = YawnTracker()
        self.gaze_estimator = GazeEstimator()
        self.distraction_tracker = DistractionTracker()
        self.head_pose_estimator = HeadPoseEstimator(
            frame_width=settings.FRAME_WIDTH,
            frame_height=settings.FRAME_HEIGHT,
        )
        self.perclos_calculator = PERCLOSCalculator(fps=30.0, window_seconds=60)
        self.window_manager = SignalWindowManager(window_seconds=5.0, fps=30.0)

        # ML & Engines
        self.feature_builder = FeatureVectorBuilder()
        self.feature_normalizer = FeatureNormalizer()
        self.cognitive_engine = CognitiveLoadEngine()
        self.risk_engine = AccidentRiskEngine()
        self.anomaly_engine = AnomalyEngine.get_instance()
        self.state_classifier = StateClassifier()
        self.recommendation_engine = RecommendationEngine()

        # Digital Twin Personalization
        self.personalization_engine = PersonalizationEngine()
        self.threshold_manager = ThresholdManager(storage_dir=Path(settings.MODEL_DIR) / "driver_profiles")

        # Session tracking state
        self.active_driver_id: Optional[int] = None
        self.active_session_id: Optional[int] = None
        self.db_session: Optional[Session] = None

        # Previous frame state caches for sequential features [15-18]
        self.prev_attention: float = 100.0
        self.prev_stress: float = 0.0
        self.prev_cli: float = 0.0
        self.prev_risk: float = 0.0

        # Last processed prediction — used by dashboard SSE stream.
        # Always a dict; initialised to an empty sentinel so consumers can
        # call getattr(pipeline, "last_prediction", None) safely.
        self.last_prediction: Optional[Dict[str, Any]] = None

        logger.info("PipelineManager successfully initialized.")

    def start_session(self, driver_id: int, session_id: int, db_session: Session) -> None:
        """Configures the pipeline for an active driver session."""
        self.active_driver_id = driver_id
        self.active_session_id = session_id
        self.db_session = db_session

        # Reset trackers
        self.blink_tracker.reset()
        self.yawn_tracker.reset()
        self.distraction_tracker.reset()
        self.perclos_calculator.reset()
        self.window_manager.reset()

        # Hydrate personalization baselines if available in driver profile
        driver = db_session.get(DriverProfile, driver_id)
        if driver:
            baseline_file = Path(self.settings.MODEL_DIR) / f"driver_{driver_id}_baseline.json"
            if baseline_file.exists():
                try:
                    self.feature_normalizer.load_from_file(baseline_file)
                    logger.info("Loaded custom normalizer baseline for driver %d from file.", driver_id)
                except Exception as exc:
                    logger.warning("Failed to load driver baseline file: %s. Using population stats.", exc)
                    self.feature_normalizer.load_population_stats()
            else:
                self.feature_normalizer.load_population_stats()

            self.threshold_manager.load_from_disk(driver_id)
        else:
            self.feature_normalizer.load_population_stats()

        logger.info("PipelineManager: Session started for driver %d, session %d.", driver_id, session_id)

    def end_session(self) -> None:
        """Closes the current session, saving personalization profiles and baselines."""
        if self.active_driver_id and self.db_session:
            driver_id = self.active_driver_id
            logger.info("PipelineManager: Ending session for driver %d.", driver_id)

            self.personalization_engine.end_session(driver_id)

            adaptive = self.personalization_engine.compute_all_thresholds(driver_id)
            self.threshold_manager.update(driver_id, adaptive)

            baseline_file = Path(self.settings.MODEL_DIR) / f"driver_{driver_id}_baseline.json"
            try:
                self.feature_normalizer.save_to_file(baseline_file)
                logger.info("Saved updated baseline profile for driver %d.", driver_id)
            except Exception as exc:
                logger.error("Failed to save driver baseline file: %s", exc)

        self.active_driver_id = None
        self.active_session_id = None
        self.db_session = None

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        frame_time_ms: float,
        telemetry: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Main pipeline loop.

        Runs landmark extraction, metric computation, ML models, state classification,
        personalization updates, recommendation checks, and DB persistence.

        Always updates self.last_prediction before returning so the SSE dashboard
        stream always has a fresh snapshot even when no face is detected.
        """
        if self.active_driver_id is None or self.db_session is None:
            raise RuntimeError("PipelineManager.process_frame: No active session.")

        driver_id = self.active_driver_id
        session_id = self.active_session_id

        # Defaults for vehicle telemetry
        tel = telemetry or {}
        speed = tel.get("speed", 0.0)
        acceleration = tel.get("acceleration", 0.0)
        steering_angle = tel.get("steering_angle", 0.0)
        brake_pressure = tel.get("brake_pressure", 0.0)
        lane_offset = tel.get("lane_offset", 0.0)
        indicator_state = int(tel.get("indicator_state", 0))

        # 1. Landmark Extraction
        lm_result: LandmarkResult = self.landmark_extractor.extract(frame)

        if not lm_result.is_valid:
            # Face not tracked — return degraded state but still update last_prediction
            result: Dict[str, Any] = {
                "face_detected": False,
                "frame_number": frame_number,
                "driver_state": DriverState.NORMAL.value
                    if isinstance(DriverState.NORMAL, DriverState)
                    else str(DriverState.NORMAL),
                "attention_score": self.prev_attention,
                "stress_score": self.prev_stress,
                "cli": self.prev_cli,
                "risk_score": self.prev_risk,
                "recommendations": [],
            }
            self.last_prediction = result
            return result

        # 2. Extract Biometrics
        ear_res = compute_ear(
            lm_result.left_eye,
            lm_result.right_eye,
            closed_frame_count=self.blink_tracker.consec_closed,
        )
        self.blink_tracker.update(ear_res)

        mar_res = compute_mar(lm_result.mouth)
        self.yawn_tracker.update(mar_res)

        gaze_res = self.gaze_estimator.estimate(
            lm_result.left_eye,
            lm_result.right_eye,
            lm_result.left_iris,
            lm_result.right_iris,
        )
        self.distraction_tracker.update(gaze_res)

        head_pose = self.head_pose_estimator.estimate(lm_result.pose_2d_points)
        perclos_res = self.perclos_calculator.update(ear_res.mean_ear)

        # 3. Update Rolling Statistics Window Manager
        self.window_manager.update(
            ear_left=ear_res.left_ear,
            ear_right=ear_res.right_ear,
            mar=mar_res.mar,
            perclos=perclos_res.perclos,
            gaze_horizontal=gaze_res.horizontal_ratio,
            gaze_vertical=gaze_res.vertical_ratio,
            head_pitch=head_pose.pitch,
            head_yaw=head_pose.yaw,
            head_roll=head_pose.roll,
        )

        # 4. Construct Feature Vector
        raw_signals = RawSignals(
            ear_left=ear_res.left_ear,
            ear_right=ear_res.right_ear,
            mar=mar_res.mar,
            perclos=perclos_res.perclos,
            fatigue_probability=perclos_res.fatigue_probability,
            blink_rate_bpm=self.blink_tracker.blinks_per_minute(),
            yawn_rate_per_hour=self.yawn_tracker.yawns_per_minute() * 60.0,
            gaze_horizontal=gaze_res.horizontal_ratio,
            gaze_vertical=gaze_res.vertical_ratio,
            gaze_off_road=gaze_res.is_off_road,
            head_pitch=head_pose.pitch,
            head_yaw=head_pose.yaw,
            head_roll=head_pose.roll,
            head_distracted=head_pose.is_distracted,
            attention_score=self.prev_attention,
            stress_score=self.prev_stress,
            cli=self.prev_cli,
            risk_score=self.prev_risk,
            blink_consec_frames=self.blink_tracker.consec_closed,
            yawn_consec_frames=self.yawn_tracker.consec_yawning,
        )

        feature_vector = self.feature_builder.build(raw_signals)
        norm_vector = self.feature_normalizer.normalize(feature_vector)

        self.feature_normalizer.update_online(feature_vector)

        # 5. Model Inferences
        cog_res = self.cognitive_engine.estimate_cognitive_load(norm_vector.to_numpy())
        risk_res = self.risk_engine.estimate_accident_risk(norm_vector.to_numpy(), cog_res)
        anomaly_res = self.anomaly_engine.predict(norm_vector.to_numpy())

        # Always update cached predictions so last_prediction is never stale
        self.prev_attention = cog_res.attention_score
        self.prev_stress = cog_res.stress_score
        self.prev_cli = cog_res.cli
        self.prev_risk = risk_res.risk_score

        # 6. DT Thresholds and State Classification
        thresh_map = {}
        for signal in TrackedSignal:
            thresh_map[signal.value] = self.threshold_manager.get(driver_id, signal)

        thresh_map["risk_score_high"] = 0.60
        thresh_map["fatigue_probability"] = 0.60
        thresh_map["distraction_probability"] = 0.60
        thresh_map["cli"] = 70.0

        driver_state = self.state_classifier.classify_state(
            risk_score=risk_res.risk_score,
            fatigue_probability=perclos_res.fatigue_probability,
            distraction_probability=gaze_res.confidence * (1.0 if gaze_res.is_off_road else 0.0),
            cli=cog_res.cli,
            is_anomaly=anomaly_res.is_anomaly,
            thresholds=thresh_map,
        )

        # 7. Personalization Observation
        self.personalization_engine.observe(
            driver_id=driver_id,
            signal_values={
                TrackedSignal.EAR_MEAN: ear_res.mean_ear,
                TrackedSignal.MAR: mar_res.mar,
                TrackedSignal.PERCLOS: perclos_res.perclos,
                TrackedSignal.BLINK_RATE_BPM: self.blink_tracker.blinks_per_minute(),
                TrackedSignal.GAZE_HORIZONTAL: gaze_res.horizontal_ratio,
                TrackedSignal.GAZE_VERTICAL: gaze_res.vertical_ratio,
                TrackedSignal.HEAD_PITCH: head_pose.pitch,
                TrackedSignal.HEAD_YAW: head_pose.yaw,
                TrackedSignal.HEAD_ROLL: head_pose.roll,
            },
        )

        # 8. Safety Recommendations
        metrics_eval = {
            "risk_score": risk_res.risk_score,
            "fatigue_probability": perclos_res.fatigue_probability,
            "perclos": perclos_res.perclos,
            "distraction_probability": 1.0 if gaze_res.is_off_road else 0.0,
            "cli": cog_res.cli,
        }
        recs_data = self.recommendation_engine.generate_recommendations(
            driver_id=driver_id,
            metrics=metrics_eval,
            thresholds=thresh_map,
            confidence_score=1.0,
        )

        recommendations_models = []
        for r_dict in recs_data:
            rec = Recommendation(
                driver_id=driver_id,
                recommendation_type=r_dict["recommendation_type"],
                priority=r_dict["priority"],
                title=r_dict["title"],
                message=r_dict["message"],
                explanation=r_dict["explanation"],
                recommended_action=r_dict["recommended_action"],
                risk_score=r_dict["risk_score"],
                confidence_score=r_dict["confidence_score"],
                trigger_metric=r_dict["trigger_metric"],
                trigger_value=r_dict["trigger_value"],
                baseline_value=r_dict["baseline_value"],
                is_personalized=r_dict["is_personalized"],
                expires_at=r_dict["expires_at"],
            )
            self.db_session.add(rec)
            recommendations_models.append(rec)

        # 9. Persist Frame Metrics
        db_metric = DrivingMetric(
            session_id=session_id,
            frame_number=frame_number,
            frame_time_ms=frame_time_ms,
            ear=ear_res.mean_ear,
            mar=mar_res.mar,
            perclos=perclos_res.perclos,
            blink_rate=self.blink_tracker.blinks_per_minute(),
            head_pitch=head_pose.pitch,
            head_yaw=head_pose.yaw,
            head_roll=head_pose.roll,
            gaze_x=gaze_res.horizontal_ratio,
            gaze_y=gaze_res.vertical_ratio,
            yawning_probability=1.0 if mar_res.is_yawning else 0.0,
            speed=speed,
            acceleration=acceleration,
            steering_angle=steering_angle,
            brake_pressure=brake_pressure,
            lane_offset=lane_offset,
            indicator_state=indicator_state,
            attention_score=cog_res.attention_score,
            cli=cog_res.cli,
            stress_score=cog_res.stress_score,
            fatigue_probability=perclos_res.fatigue_probability,
            distraction_probability=1.0 if gaze_res.is_off_road else 0.0,
            aggression_score=anomaly_res.anomaly_score,
            risk_score=risk_res.risk_score,
            driver_state=driver_state,
        )
        self.db_session.add(db_metric)

        if frame_number % 300 == 0:
            try:
                self.db_session.commit()
                self.feature_normalizer.commit_online_stats(min_samples=100)
            except Exception as exc:
                logger.error("Failed to commit metrics payload: %s", exc)
                self.db_session.rollback()

        # Serialise driver_state to string for JSON compatibility
        driver_state_value = (
            driver_state.value if isinstance(driver_state, DriverState) else str(driver_state)
        )

        # Build output dict and cache it for the SSE dashboard stream
        result = {
            "face_detected": True,
            "frame_number": frame_number,
            "driver_state": driver_state_value,
            "attention_score": cog_res.attention_score,
            "stress_score": cog_res.stress_score,
            "cli": cog_res.cli,
            "risk_score": risk_res.risk_score,
            "anomaly_score": anomaly_res.anomaly_score,
            "is_anomaly": anomaly_res.is_anomaly,
            "recommendations": [r.to_dict() for r in recommendations_models],
            "biometrics": {
                "ear": ear_res.mean_ear,
                "mar": mar_res.mar,
                "perclos": perclos_res.perclos,
                "blink_rate": self.blink_tracker.blinks_per_minute(),
                "head_pitch": head_pose.pitch,
                "head_yaw": head_pose.yaw,
                "head_roll": head_pose.roll,
                "gaze_x": gaze_res.horizontal_ratio,
                "gaze_y": gaze_res.vertical_ratio,
            },
        }

        # Always update last_prediction so the SSE stream can read it
        self.last_prediction = result

        return result
