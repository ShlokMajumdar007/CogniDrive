from __future__ import annotations
import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
try:
    import joblib
except ImportError as exc:
    raise ImportError(
        "joblib is required to train and persist CogniDrive ML models. "
        "Install it with `pip install joblib`."
    ) from exc
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required to train CogniDrive ML models. "
        "Install it with `pip install scikit-learn`."
    ) from exc
try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError(
        "lightgbm is required to train the CogniDrive cognitive load model. "
        "Install it with `pip install lightgbm`."
    ) from exc
try:
    from backend.features.feature_vector import FEATURE_DIM, FEATURE_NAMES
    from backend.app.config import get_settings
    from backend.app.constants import MLConstants
    from backend.ml.training.train_embeddings import SyntheticDriverDataGenerator
except ImportError:
    from features.feature_vector import FEATURE_DIM, FEATURE_NAMES
    from app.config import get_settings
    from app.constants import MLConstants
    from ml.training.train_embeddings import SyntheticDriverDataGenerator
def _setup_logger() -> logging.Logger:
    """Configures and returns the module logger."""
    logger = logging.getLogger("CogniDrive.Training.Cognitive")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
logger = _setup_logger()
DEFAULT_N_SAMPLES: int = 30_000
DEFAULT_N_ESTIMATORS: int = 200
DEFAULT_MAX_DEPTH: int = 6
DEFAULT_LEARNING_RATE: float = 0.05
DEFAULT_TEST_SIZE: float = 0.15
DEFAULT_RANDOM_STATE: int = 42

TARGET_NAMES: Tuple[str, str, str] = ("attention_score", "stress_score", "cli")
_CLI_ATTENTION_WEIGHT: float = 0.60
_CLI_STRESS_WEIGHT: float = 0.40
_LABEL_NOISE_STD: float = 4.0
_IDX_EAR_MEAN: int = 2
_IDX_PERCLOS: int = 4
_IDX_FATIGUE_PROB: int = 5
_IDX_GAZE_OFF_ROAD: int = 10
_IDX_HEAD_DISTRACTED: int = 14
_IDX_PREV_STRESS: int = 16
_BASELINE_EAR: float = 0.28
def _validate_feature_indices(feature_dim: int) -> None:
    indices = {
        "EAR_MEAN": _IDX_EAR_MEAN,
        "PERCLOS": _IDX_PERCLOS,
        "FATIGUE_PROB": _IDX_FATIGUE_PROB,
        "GAZE_OFF_ROAD": _IDX_GAZE_OFF_ROAD,
        "HEAD_DISTRACTED": _IDX_HEAD_DISTRACTED,
        "PREV_STRESS": _IDX_PREV_STRESS,
    }
    for name, idx in indices.items():
        if idx >= feature_dim or idx < 0:
            raise ValueError(
                f"Feature index {name}={idx} out of range [0, {feature_dim})"
            )
@dataclass
class CognitiveLabelGenerator:
    noise_std: float = _LABEL_NOISE_STD
    random_state: int = DEFAULT_RANDOM_STATE
    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_state)
    def generate(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected X of shape (n_samples, {FEATURE_DIM}), got {X.shape}."
            )
        _validate_feature_indices(FEATURE_DIM)
        ear_mean = X[:, _IDX_EAR_MEAN]
        perclos = X[:, _IDX_PERCLOS]
        fatigue_prob = X[:, _IDX_FATIGUE_PROB]
        off_road = X[:, _IDX_GAZE_OFF_ROAD]
        head_distracted = X[:, _IDX_HEAD_DISTRACTED]
        prev_stress_norm = X[:, _IDX_PREV_STRESS]
        ear_deficit = np.clip((_BASELINE_EAR - ear_mean) / _BASELINE_EAR, 0.0, None)
        attention_raw = (
            1.0
            - 0.40 * off_road
            - 0.25 * head_distracted
            - 0.25 * ear_deficit
            - 0.10 * perclos
        )
        attention = np.clip(attention_raw * 100.0, 0.0, 100.0)
        stress_raw = (
            0.50 * fatigue_prob
            + 0.30 * perclos
            + 0.20 * prev_stress_norm
        )
        stress = np.clip(stress_raw * 100.0, 0.0, 100.0)
        cli = np.clip(
            _CLI_ATTENTION_WEIGHT * (100.0 - attention) + _CLI_STRESS_WEIGHT * stress,
            0.0,
            100.0,
        )

        labels = np.stack([attention, stress, cli], axis=1)
        noise = self._rng.normal(loc=0.0, scale=self.noise_std, size=labels[:, :2].shape)
        noisy_att_stress = np.clip(labels[:, :2] + noise, 0.0, 100.0)
        noisy_cli = np.clip(
            _CLI_ATTENTION_WEIGHT * (100.0 - noisy_att_stress[:, 0])
            + _CLI_STRESS_WEIGHT * noisy_att_stress[:, 1],
            0.0,
            100.0,
        )

        labels = np.column_stack([noisy_att_stress, noisy_cli]).astype(np.float32)
        return labels
class CognitiveTrainer:
    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        random_state: int = DEFAULT_RANDOM_STATE,
        test_size: float = DEFAULT_TEST_SIZE,
    ) -> None:

        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}.")
        if n_estimators <= 0:
            raise ValueError(f"n_estimators must be positive, got {n_estimators}.")
        if max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {max_depth}.")
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}.")
        if not (0.0 < test_size < 1.0):
            raise ValueError(f"test_size must be in (0, 1), got {test_size}.")
        self.feature_dim = feature_dim
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.test_size = test_size
        self._pipeline: Optional[Pipeline] = None
        self._val_mae: Optional[Dict[str, float]] = None
    def generate_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        feature_generator = SyntheticDriverDataGenerator(
            feature_dim=self.feature_dim, random_state=self.random_state
        )
        X = feature_generator.generate(n_samples)

        label_generator = CognitiveLabelGenerator(random_state=self.random_state)
        y = label_generator.generate(X)

        return X, y
    def fit(self, X: np.ndarray, y: np.ndarray) -> "CognitiveTrainer":
        if X.ndim != 2 or X.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected X of shape (n_samples, {self.feature_dim}), got {X.shape}."
            )
        if y.ndim != 2 or y.shape[1] != 3:
            raise ValueError(f"Expected y of shape (n_samples, 3), got {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y sample counts must match: {X.shape[0]} != {y.shape[0]}."
            )
        if X.shape[0] < 2:
            raise ValueError("At least 2 samples are required to fit the model.")
        try:
            _validate_feature_indices(self.feature_dim)
        except ValueError as exc:
            raise ValueError(f"Feature configuration error: {exc}") from exc

        logger.info(
            "Starting cognitive load model training | samples=%d, features=%d, "
            "targets=%s, estimators=%d, depth=%d, lr=%.4f",
            X.shape[0],
            self.feature_dim,
            TARGET_NAMES,
            self.n_estimators,
            self.max_depth,
            self.learning_rate,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )

        base_estimator = LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1,
        )
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", MultiOutputRegressor(estimator=base_estimator, n_jobs=1)),
        ])

        start = time.time()
        try:
            pipeline.fit(X_train, y_train)
        except Exception as exc:
            raise RuntimeError(f"Failed to train cognitive load model: {exc}") from exc
        elapsed = time.time() - start
        logger.info("Cognitive model training completed | elapsed=%.2fs", elapsed)
        y_pred = pipeline.predict(X_val)
        if y_pred.ndim != 2 or y_pred.shape[1] != 3:
            raise RuntimeError(
                f"Pipeline output shape mismatch: expected (n, 3), got {y_pred.shape}"
            )
        self._val_mae = {
            name: float(mean_absolute_error(y_val[:, i], y_pred[:, i]))
            for i, name in enumerate(TARGET_NAMES)
        }
        logger.info(
            "Validation MAE | attention=%.3f, stress=%.3f, cli=%.3f",
            self._val_mae["attention_score"],
            self._val_mae["stress_score"],
            self._val_mae["cli"],
        )

        self._pipeline = pipeline
        return self
    def save(self, output_path: Path) -> Path:
        if self._pipeline is None:
            raise RuntimeError("Cannot save: model must be fitted first (call fit() before save()).")

        output_path = Path(output_path)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._pipeline, output_path)
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info("Successfully saved cognitive model to %s | size=%.2f MB", 
                       output_path, file_size_mb)
        except (OSError, IOError) as exc:
            raise RuntimeError(f"Failed to persist model to {output_path}: {exc}") from exc

        return output_path

    @property
    def pipeline(self) -> Optional[Pipeline]:
        return self._pipeline

    @property
    def val_mae(self) -> Optional[Dict[str, float]]:
        return self._val_mae

    def smoke_test(self, n_probe: int = 4) -> bool:
        if self._pipeline is None:
            raise RuntimeError("Cannot run smoke test: model must be fitted first.")

        generator = SyntheticDriverDataGenerator(
            feature_dim=self.feature_dim, random_state=self.random_state + 1
        )
        probe = generator.generate(n_probe)
        preds = self._pipeline.predict(probe)
        if preds.shape != (n_probe, 3):
            logger.error(
                "Smoke test FAILED | Shape check: expected %s, got %s", 
                (n_probe, 3), preds.shape
            )
            return False
        if not np.all(np.isfinite(preds)):
            non_finite_count = np.sum(~np.isfinite(preds))
            logger.error("Smoke test FAILED | %d non-finite predictions", non_finite_count)
            return False
        margin = 1e-3
        out_of_range = (preds < -margin) | (preds > 100.0 + margin)
        if np.any(out_of_range):
            n_violating = np.sum(out_of_range)
            max_violation = np.max(preds) if np.any(preds > 100.0) else np.min(preds)
            logger.error(
                "Smoke test FAILED | %d predictions out of [0,100] range, max=%.3f", 
                n_violating, max_violation
            )
            return False
        attention, stress, cli = preds[:, 0], preds[:, 1], preds[:, 2]
        expected_cli = _CLI_ATTENTION_WEIGHT * (100.0 - attention) + _CLI_STRESS_WEIGHT * stress
        cli_deviation = float(np.mean(np.abs(cli - expected_cli)))

        logger.info(
            "Smoke test PASSED | shape=%s, range=[%.1f, %.1f], cli_error=%.3f",
            preds.shape,
            np.min(preds),
            np.max(preds),
            cli_deviation,
        )
        return True

_CACHED_SETTINGS = None

def resolve_output_path(explicit_path: Optional[str]) -> Path:

    if explicit_path:
        return Path(explicit_path)
    try:
        global _CACHED_SETTINGS
        if _CACHED_SETTINGS is None:
            _CACHED_SETTINGS = get_settings()
        model_path = Path(_CACHED_SETTINGS.MODEL_DIR) / MLConstants.COGNITIVE_MODEL_NAME.value
        return model_path
    except Exception as exc:  # pragma: no cover - defensive fallback
        fallback_path = Path("backend/ml/models_saved/cognitive_load_xgb.joblib")
        logger.warning(
            "Could not resolve path from settings (%s); using fallback: %s", 
            exc, fallback_path
        )
        return fallback_path

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CogniDrive cognitive load model (Attention/Stress/CLI), offline.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES,
        help=f"Number of synthetic training samples (default: {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS,
        help=f"LightGBM boosting rounds per target (default: {DEFAULT_N_ESTIMATORS}).",
    )
    parser.add_argument(
        "--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
        help=f"LightGBM max tree depth (default: {DEFAULT_MAX_DEPTH}).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
        help=f"LightGBM learning rate (default: {DEFAULT_LEARNING_RATE}).",
    )
    parser.add_argument(
        "--test-size", type=float, default=DEFAULT_TEST_SIZE,
        help=f"Validation split fraction (default: {DEFAULT_TEST_SIZE}).",
    )
    parser.add_argument(
        "--random-state", type=int, default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE}).",
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help=(
            "Destination .joblib path. Defaults to "
            "<MODEL_DIR>/cognitive_load_xgb.joblib (see app.config.Settings)."
        ),
    )
    return parser.parse_args(argv)
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logger.info("=" * 70)
    logger.info("CogniDrive ML Training Pipeline: Cognitive Load Model")
    logger.info("=" * 70)
    try:
        trainer = CognitiveTrainer(
            feature_dim=FEATURE_DIM,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
            test_size=args.test_size,
        )
        X, y = trainer.generate_data(args.n_samples)
        trainer.fit(X, y)
        if not trainer.smoke_test():
            return 1
        output_path = resolve_output_path(args.output_path)
        saved_path = trainer.save(output_path)
        logger.info("=" * 70)
        logger.info("✓ Training completed successfully")
        logger.info("Model metrics: %s", trainer.val_mae)
        logger.info("Saved to: %s", saved_path)
        logger.info("=" * 70)
        return 0
    except Exception as exc:
        logger.error("✗ Training failed: %s", exc, exc_info=True)
        return 1
if __name__ == "__main__":
    sys.exit(main())