from __future__ import annotations
import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
try:
    import joblib
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "joblib is required to train and persist CogniDrive ML models. "
        "Install it with `pip install joblib`."
    ) from exc
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required to train CogniDrive ML models. "
        "Install it with `pip install scikit-learn`."
    ) from exc
try:
    from lightgbm import LGBMClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lightgbm is required to train the CogniDrive accident risk model. "
        "Install it with `pip install lightgbm`."
    ) from exc
# ---------------------------------------------------------------------------
# Project imports with fallback for direct script execution.
# ---------------------------------------------------------------------------
try:
    from backend.features.feature_vector import FEATURE_DIM
    from backend.app.config import get_settings
    from backend.app.constants import MLConstants
    from backend.ml.training.train_embeddings import SyntheticDriverDataGenerator
except ImportError:
    from features.feature_vector import FEATURE_DIM  # type: ignore[no-redef]
    from app.config import get_settings  # type: ignore[no-redef]
    from app.constants import MLConstants  # type: ignore[no-redef]
    from ml.training.train_embeddings import SyntheticDriverDataGenerator  # type: ignore[no-redef]
logger = logging.getLogger("CogniDrive.Training.Risk")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_N_SAMPLES: int = 40_000
DEFAULT_N_ESTIMATORS: int = 250
DEFAULT_MAX_DEPTH: int = 7
DEFAULT_LEARNING_RATE: float = 0.05
DEFAULT_TEST_SIZE: float = 0.15
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_EVENT_RATE: float = 0.12
_RISK_HIGH_THRESHOLD: float = 0.60
_IDX_PERCLOS: int = 4
_IDX_FATIGUE_PROB: int = 5
_IDX_GAZE_OFF_ROAD: int = 10
_IDX_HEAD_DISTRACTED: int = 14
_IDX_CLI_NORM: int = 17
_IDX_PREV_RISK: int = 18
_W_FATIGUE: float = 0.30
_W_PERCLOS: float = 0.25
_W_OFF_ROAD: float = 0.20
_W_HEAD_DISTRACTED: float = 0.15
_W_CLI: float = 0.10


# ---------------------------------------------------------------------------
# Synthetic event simulation (discrete high-risk injections)
# ---------------------------------------------------------------------------
@dataclass
class RiskEventSimulator:
    event_rate: float = DEFAULT_EVENT_RATE
    random_state: int = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        """Validates configuration and initialises the random generator."""
        if not (0.0 < self.event_rate < 1.0):
            raise ValueError(f"event_rate must be in (0, 1), got {self.event_rate}.")
        self._rng = np.random.default_rng(self.random_state)

    def inject(self, X: np.ndarray) -> np.ndarray:
        """Injects high-risk events into a copy of the feature matrix.

        Args:
            X: Base synthetic feature matrix of shape
                ``(n_samples, FEATURE_DIM)``.

        Returns:
            np.ndarray: New feature matrix (copy of ``X``) with a subset of
            rows perturbed to represent discrete high-risk driving events.

        Raises:
            ValueError: If ``X`` does not have ``FEATURE_DIM`` columns.
        """
        if X.ndim != 2 or X.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected X of shape (n_samples, {FEATURE_DIM}), got {X.shape}."
            )

        X_event = X.copy()
        n_samples = X.shape[0]
        n_events = int(round(n_samples * self.event_rate))
        event_idx = self._rng.choice(n_samples, size=n_events, replace=False)

        # Three event archetypes, each pushing a different risk pathway to
        # an extreme, modelling distinct real-world near-miss scenarios.
        event_types = self._rng.integers(low=0, high=3, size=n_events)

        for local_i, global_i in enumerate(event_idx):
            kind = event_types[local_i]
            if kind == 0:
                # Micro-sleep / severe fatigue event.
                X_event[global_i, _IDX_PERCLOS] = self._rng.uniform(0.65, 1.0)
                X_event[global_i, _IDX_FATIGUE_PROB] = self._rng.uniform(0.70, 1.0)
            elif kind == 1:
                # Sudden distraction event (gaze + head off-road simultaneously).
                X_event[global_i, _IDX_GAZE_OFF_ROAD] = 1.0
                X_event[global_i, _IDX_HEAD_DISTRACTED] = 1.0
            else:
                # Cognitive overload spike (high CLI carried over from prior frame).
                X_event[global_i, _IDX_CLI_NORM] = self._rng.uniform(0.75, 1.0)
                X_event[global_i, _IDX_PREV_RISK] = self._rng.uniform(0.55, 0.85)

        logger.info(
            "Injected %d high-risk events (%.1f%%) across %d synthetic samples.",
            n_events,
            100.0 * self.event_rate,
            n_samples,
        )
        return X_event


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------


def compute_risk_labels(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2 or X.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected X of shape (n_samples, {FEATURE_DIM}), got {X.shape}."
        )

    fatigue_prob = X[:, _IDX_FATIGUE_PROB]
    perclos = X[:, _IDX_PERCLOS]
    off_road = X[:, _IDX_GAZE_OFF_ROAD]
    head_distracted = X[:, _IDX_HEAD_DISTRACTED]
    cli_norm = X[:, _IDX_CLI_NORM]
    prev_risk = X[:, _IDX_PREV_RISK]

    risk_raw = (
        _W_FATIGUE * fatigue_prob
        + _W_PERCLOS * perclos
        + _W_OFF_ROAD * off_road
        + _W_HEAD_DISTRACTED * head_distracted
        + _W_CLI * cli_norm
    )
    risk_score = np.clip(0.80 * risk_raw + 0.20 * prev_risk, 0.0, 1.0)

    labels = (risk_score >= _RISK_HIGH_THRESHOLD).astype(np.int64)
    return labels
# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class RiskTrainer:
    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        random_state: int = DEFAULT_RANDOM_STATE,
        test_size: float = DEFAULT_TEST_SIZE,
        event_rate: float = DEFAULT_EVENT_RATE,
    ) -> None:
        """Initialises the accident risk trainer.

        Args:
            feature_dim: Dimensionality of input feature vectors.
            n_estimators: Number of boosting rounds for the LightGBM
                classifier.
            max_depth: Maximum tree depth for the LightGBM classifier.
            learning_rate: Learning rate (shrinkage) for LightGBM.
            random_state: Seed for data generation, splitting, and model
                initialisation.
            test_size: Fraction (0, 1) of samples reserved for evaluation.
            event_rate: Target fraction of synthetic high-risk event
                injections (see :class:`RiskEventSimulator`).

        Raises:
            ValueError: If any numeric argument is out of valid range.
        """
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
        if not (0.0 < event_rate < 1.0):
            raise ValueError(f"event_rate must be in (0, 1), got {event_rate}.")

        self.feature_dim = feature_dim
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.test_size = test_size
        self.event_rate = event_rate

        self._pipeline: Optional[Pipeline] = None
        self._metrics: Optional[dict] = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def generate_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        feature_generator = SyntheticDriverDataGenerator(
            feature_dim=self.feature_dim, random_state=self.random_state
        )
        X_base = feature_generator.generate(n_samples)

        event_simulator = RiskEventSimulator(
            event_rate=self.event_rate, random_state=self.random_state
        )
        X = event_simulator.inject(X_base)

        y = compute_risk_labels(X)

        positive_rate = float(np.mean(y))
        logger.info(
            "Generated risk training data: n_samples=%d, positive_rate=%.2f%%",
            n_samples,
            100.0 * positive_rate,
        )

        return X, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 2 or X.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected X of shape (n_samples, {self.feature_dim}), got {X.shape}."
            )
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(
                f"Expected y of shape ({X.shape[0]},), got {y.shape}."
            )
        if X.shape[0] < 2:
            raise ValueError("At least 2 samples are required to fit the model.")
        if len(np.unique(y)) < 2:
            raise ValueError(
                "y must contain both classes (0 and 1) to train a binary classifier; "
                "consider increasing event_rate or n_samples."
            )

        logger.info(
            "Starting accident risk model training: n_samples=%d, feature_dim=%d, "
            "n_estimators=%d, max_depth=%d, lr=%.4f",
            X.shape[0],
            self.feature_dim,
            self.n_estimators,
            self.max_depth,
            self.learning_rate,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            shuffle=True, stratify=y,
        )

        # Address class imbalance via LightGBM's built-in scale_pos_weight,
        # computed from the training split's actual class ratio.
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        scale_pos_weight = float(n_neg) / float(max(n_pos, 1))

        classifier = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary",
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1,
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])

        start = time.time()
        try:
            pipeline.fit(X_train, y_train)
        except Exception as exc:
            logger.error("Accident risk model training failed: %s", exc)
            raise RuntimeError("Failed to train accident risk model.") from exc
        elapsed = time.time() - start

        logger.info(
            "Accident risk model training completed in %.2fs "
            "(scale_pos_weight=%.3f).",
            elapsed,
            scale_pos_weight,
        )

        self._metrics = self._evaluate(pipeline, X_val, y_val)
        logger.info(
            "Validation — ROC-AUC=%.4f, PR-AUC=%.4f, precision=%.4f, "
            "recall=%.4f, f1=%.4f",
            self._metrics["roc_auc"],
            self._metrics["pr_auc"],
            self._metrics["precision"],
            self._metrics["recall"],
            self._metrics["f1"],
        )
        logger.info("Confusion matrix [[TN, FP], [FN, TP]]: %s", self._metrics["confusion_matrix"])

        self._pipeline = pipeline
        return self

    @staticmethod
    def _evaluate(pipeline: Pipeline, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """Computes holdout classification metrics.

        Args:
            pipeline: Fitted classification pipeline exposing
                ``predict_proba``.
            X_val: Validation feature matrix.
            y_val: Validation binary labels.

        Returns:
            dict: Metrics including ``roc_auc``, ``pr_auc``, ``precision``,
            ``recall``, ``f1``, and ``confusion_matrix`` (as a nested list).
        """
        proba = pipeline.predict_proba(X_val)[:, 1]
        preds = (proba >= _RISK_HIGH_THRESHOLD).astype(np.int64)

        roc_auc = float(roc_auc_score(y_val, proba))
        pr_auc = float(average_precision_score(y_val, proba))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, preds, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_val, preds).tolist()

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_path: Path) -> Path:
        """Serialises the trained pipeline to disk via joblib.

        Args:
            output_path: Destination ``.joblib`` file path.

        Returns:
            Path: The resolved output path the model was written to.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet, or the
                write fails.
        """
        if self._pipeline is None:
            raise RuntimeError("Cannot save: call fit() before save().")

        output_path = Path(output_path)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._pipeline, output_path)
            logger.info("Saved accident risk model pipeline to %s", output_path)
        except OSError as exc:
            logger.error("Failed to save risk model to %s: %s", output_path, exc)
            raise RuntimeError(f"Failed to write model artifact to {output_path}") from exc

        return output_path

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def pipeline(self) -> Optional[Pipeline]:
        """Returns the fitted pipeline, or ``None`` if unfit."""
        return self._pipeline

    @property
    def metrics(self) -> Optional[dict]:
        """Returns holdout evaluation metrics, or ``None`` if unfit."""
        return self._metrics

    def smoke_test(self, n_probe: int = 4) -> bool:
        """Runs a quick sanity check on the fitted pipeline.

        Verifies that ``predict_proba`` produces a correctly-shaped,
        finite, valid-probability output matching the
        :class:`~backend.ml.inference.risk_model.RiskModel` contract:
        ``model.predict_proba(x)`` -> shape ``(n, 2)``, rows summing to 1.

        Args:
            n_probe: Number of probe samples to score.

        Returns:
            bool: True if all sanity checks pass.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self._pipeline is None:
            raise RuntimeError("Cannot smoke-test: call fit() before smoke_test().")

        feature_generator = SyntheticDriverDataGenerator(
            feature_dim=self.feature_dim, random_state=self.random_state + 1
        )
        probe = feature_generator.generate(n_probe)
        proba = self._pipeline.predict_proba(probe)

        if proba.shape != (n_probe, 2):
            logger.error(
                "Smoke test failed: expected shape %s, got %s", (n_probe, 2), proba.shape
            )
            return False

        if not np.all(np.isfinite(proba)):
            logger.error("Smoke test failed: non-finite probabilities.")
            return False

        row_sums = proba.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            logger.error(
                "Smoke test failed: predict_proba rows do not sum to 1: %s", row_sums
            )
            return False

        logger.info(
            "Smoke test passed: predict_proba shape=%s, sample positive-class "
            "probabilities=%s",
            proba.shape,
            np.round(proba[:, 1], 4),
        )
        return True


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_output_path(explicit_path: Optional[str]) -> Path:
    """Resolves the output path for the trained accident risk model.

    Resolution order:
        1. ``explicit_path`` if provided.
        2. ``<settings.MODEL_DIR>/<MLConstants.RISK_MODEL_NAME>``.
        3. ``backend/ml/models_saved/accident_risk_lgb.joblib`` as a final
           hardcoded fallback.

    Args:
        explicit_path: Optional CLI-provided output path.

    Returns:
        Path: Resolved output path (not yet created on disk).
    """
    if explicit_path:
        return Path(explicit_path)

    try:
        settings = get_settings()
        return Path(settings.MODEL_DIR) / MLConstants.RISK_MODEL_NAME.value
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "Could not resolve output path from settings (%s); using default "
            "backend/ml/models_saved/accident_risk_lgb.joblib", exc
        )
        return Path("backend/ml/models_saved/accident_risk_lgb.joblib")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for the accident risk training script.

    Args:
        argv: Optional argument list (used for testing). Defaults to
            ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train the CogniDrive accident risk classifier, offline.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=DEFAULT_N_SAMPLES,
        help=f"Number of synthetic training samples (default: {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS,
        help=f"LightGBM boosting rounds (default: {DEFAULT_N_ESTIMATORS}).",
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
        "--event-rate", type=float, default=DEFAULT_EVENT_RATE,
        help=f"Target fraction of injected high-risk events (default: {DEFAULT_EVENT_RATE}).",
    )
    parser.add_argument(
        "--random-state", type=int, default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE}).",
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help=(
            "Destination .joblib path. Defaults to "
            "<MODEL_DIR>/accident_risk_lgb.joblib (see app.config.Settings)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Runs the full accident risk model training pipeline.

    Steps:
        1. Generate synthetic 21-D driver feature data.
        2. Inject discrete high-risk events (micro-sleep, distraction,
           cognitive overload spikes).
        3. Derive binary risk labels via the shared domain-knowledge
           heuristic, thresholded at the fallback model's HIGH boundary.
        4. Fit a ``StandardScaler -> LGBMClassifier`` pipeline with
           class-imbalance correction.
        5. Evaluate on a stratified holdout split (ROC-AUC, PR-AUC,
           precision/recall/F1, confusion matrix).
        6. Run a smoke test on the fitted pipeline.
        7. Persist the pipeline to disk via joblib.

    Args:
        argv: Optional argument list (used for testing).

    Returns:
        int: Process exit code (``0`` on success, ``1`` on failure).
    """
    args = parse_args(argv)

    logger.info("=" * 70)
    logger.info("CogniDrive — Accident Risk Model Training")
    logger.info("=" * 70)

    try:
        trainer = RiskTrainer(
            feature_dim=FEATURE_DIM,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
            test_size=args.test_size,
            event_rate=args.event_rate,
        )

        X, y = trainer.generate_data(args.n_samples)
        trainer.fit(X, y)

        if not trainer.smoke_test():
            logger.error("Smoke test failed — refusing to persist a broken model.")
            return 1

        output_path = resolve_output_path(args.output_path)
        trainer.save(output_path)

        logger.info("Training complete. metrics=%s", trainer.metrics)
        return 0

    except Exception as exc:
        logger.error("Accident risk training pipeline failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())