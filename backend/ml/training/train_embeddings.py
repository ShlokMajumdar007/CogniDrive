from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import joblib
except ImportError as exc:  # pragma: no cover - joblib is a hard dependency
    raise ImportError(
        "joblib is required to train and persist CogniDrive ML models. "
        "Install it with `pip install joblib`."
    ) from exc

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
except ImportError as exc:  # pragma: no cover - scikit-learn is a hard dependency
    raise ImportError(
        "scikit-learn is required to train CogniDrive ML models. "
        "Install it with `pip install scikit-learn`."
    ) from exc

# ---------------------------------------------------------------------------
# Project imports with fallback to support both `python -m backend....` and
# direct script execution from within the `backend/` directory.
# ---------------------------------------------------------------------------

from backend.features.feature_vector import FEATURE_DIM, FEATURE_NAMES
from backend.app.config import get_model_path
from backend.app.constants import MLConstants
from backend.ml.inference.encoder_transformer import EncoderTransformer as _EncoderTransformer


logger = logging.getLogger("CogniDrive.Training.Embeddings")
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

DEFAULT_EMBEDDING_DIM: int = 128
DEFAULT_N_SAMPLES: int = 20_000
DEFAULT_MAX_ITER: int = 300
DEFAULT_TEST_SIZE: float = 0.15
DEFAULT_RANDOM_STATE: int = 42

# Driver archetype centroids over the 21-D feature space. Each archetype
# represents a coherent behavioural persona used to seed synthetic samples
# so the embedding space learns *separable* clusters from the outset
# (this directly supports the cold-start clustering in
# backend/ml/recommendation/clustering.py).
#
# Indices follow FEATURE_NAMES ordering in backend/features/feature_vector.py:
#   [0]  ear_left            [7]  yawn_rate_norm        [14] head_distracted
#   [1]  ear_right           [8]  gaze_horizontal        [15] attention_score_norm
#   [2]  ear_mean            [9]  gaze_vertical          [16] stress_score_norm
#   [3]  mar                 [10] gaze_off_road          [17] cli_norm
#   [4]  perclos             [11] head_pitch_norm        [18] risk_score
#   [5]  fatigue_probability [12] head_yaw_norm          [19] blink_consec_norm
#   [6]  blink_rate_norm     [13] head_roll_norm         [20] yawn_consec_norm
_ARCHETYPE_CENTROIDS: Tuple[Tuple[float, ...], ...] = (
    # ALERT_COMMUTER: high EAR, low PERCLOS, low everything-bad
    (0.30, 0.30, 0.30, 0.10, 0.05, 0.05, 0.45, 0.05, 0.50, 0.50, 0.05, 0.05, 0.05, 0.05, 0.05, 0.90, 0.15, 0.10, 0.05, 0.05, 0.02),
    # NIGHT_SHIFT_FATIGUED: low EAR, high PERCLOS/fatigue, high blink consec
    (0.16, 0.16, 0.16, 0.30, 0.35, 0.55, 0.65, 0.30, 0.50, 0.45, 0.10, 0.10, 0.08, 0.08, 0.10, 0.45, 0.45, 0.55, 0.35, 0.50, 0.30),
    # AGGRESSIVE_URBAN: normal EAR, high gaze movement, elevated stress/cli/risk
    (0.27, 0.27, 0.27, 0.15, 0.08, 0.08, 0.55, 0.10, 0.65, 0.60, 0.20, 0.15, 0.20, 0.15, 0.20, 0.65, 0.65, 0.60, 0.45, 0.10, 0.05),
    # DISTRACTED_NEWBIE: frequent off-road gaze and head distraction
    (0.26, 0.26, 0.26, 0.20, 0.10, 0.10, 0.50, 0.15, 0.80, 0.75, 0.65, 0.30, 0.40, 0.25, 0.70, 0.40, 0.40, 0.45, 0.30, 0.15, 0.10),
    # ANXIOUS_COGNITIVE_OVERLOAD: high stress/cli, moderate fatigue
    (0.22, 0.22, 0.22, 0.25, 0.15, 0.20, 0.50, 0.20, 0.55, 0.55, 0.15, 0.12, 0.12, 0.10, 0.20, 0.55, 0.80, 0.80, 0.40, 0.15, 0.10),
    # HIGH_RISK_MICROSLEEP: severe PERCLOS/fatigue and risk
    (0.10, 0.10, 0.10, 0.35, 0.60, 0.85, 0.75, 0.40, 0.50, 0.45, 0.15, 0.20, 0.10, 0.10, 0.15, 0.20, 0.55, 0.75, 0.85, 0.80, 0.50),
)

# Per-feature noise standard deviations (kept proportionally small so
# archetype clusters remain well-separated while still spanning realistic
# intra-driver variance).
_NOISE_STD: float = 0.045

# Hard bounds matching FeatureVectorBuilder normalisation ranges.
_FEATURE_BOUNDS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.5), (0.0, 0.5), (0.0, 0.5),  # ear_left, ear_right, ear_mean
    (0.0, 1.5),                           # mar
    (0.0, 1.0), (0.0, 1.0),               # perclos, fatigue_probability
    (0.0, 1.0), (0.0, 1.0),               # blink_rate_norm, yawn_rate_norm
    (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),    # gaze_h, gaze_v, gaze_off_road
    (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),  # head pitch/yaw/roll
    (0.0, 1.0),                            # head_distracted
    (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),     # attention, stress, cli
    (0.0, 1.0),                            # risk_score
    (0.0, 1.0), (0.0, 1.0),                # blink_consec, yawn_consec
)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


@dataclass
class SyntheticDriverDataGenerator:
    """Generates synthetic 21-D feature vectors for offline embedding training.

    Samples are drawn from a mixture of behavioural archetypes (see
    :data:`_ARCHETYPE_CENTROIDS`) with additive Gaussian noise, then clipped
    to the same bounds enforced by
    :class:`backend.features.feature_vector.FeatureVectorBuilder`. This keeps
    synthetic training data statistically consistent with the real-time
    inference pipeline's inputs.

    Attributes:
        feature_dim: Dimensionality of generated feature vectors. Must equal
            :data:`backend.features.feature_vector.FEATURE_DIM`.
        random_state: Seed for the internal NumPy random generator, ensuring
            fully reproducible offline training runs.
        noise_std: Standard deviation of the additive Gaussian noise applied
            to each archetype centroid.
    """

    feature_dim: int = FEATURE_DIM
    random_state: int = DEFAULT_RANDOM_STATE
    noise_std: float = _NOISE_STD

    def __post_init__(self) -> None:
        """Validates configuration and initialises the random generator."""
        if self.feature_dim != FEATURE_DIM:
            raise ValueError(
                f"SyntheticDriverDataGenerator.feature_dim ({self.feature_dim}) "
                f"must match FEATURE_DIM ({FEATURE_DIM})."
            )

        centroid_dims = {len(c) for c in _ARCHETYPE_CENTROIDS}
        if centroid_dims != {self.feature_dim}:
            raise ValueError(
                f"Archetype centroid dimensions {centroid_dims} do not match "
                f"feature_dim ({self.feature_dim})."
            )

        self._rng = np.random.default_rng(self.random_state)
        self._centroids = np.array(_ARCHETYPE_CENTROIDS, dtype=np.float64)
        self._bounds = np.array(_FEATURE_BOUNDS, dtype=np.float64)

        logger.debug(
            "SyntheticDriverDataGenerator initialised with %d archetypes, "
            "feature_dim=%d, noise_std=%.4f",
            len(self._centroids),
            self.feature_dim,
            self.noise_std,
        )

    def generate(self, n_samples: int) -> np.ndarray:
        """Generates ``n_samples`` synthetic 21-D feature vectors.

        Samples are drawn uniformly across the configured archetypes, with
        Gaussian perturbation and clipping to physiologically valid ranges.

        Args:
            n_samples: Number of synthetic samples to generate. Must be
                strictly positive.

        Returns:
            np.ndarray: Array of shape ``(n_samples, feature_dim)``, dtype
            float32.

        Raises:
            ValueError: If ``n_samples`` is not a positive integer.
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples}.")

        n_archetypes = len(self._centroids)
        archetype_idx = self._rng.integers(low=0, high=n_archetypes, size=n_samples)

        base = self._centroids[archetype_idx]
        noise = self._rng.normal(loc=0.0, scale=self.noise_std, size=base.shape)
        samples = base + noise

        # Clip to valid per-feature ranges (matches FeatureVectorBuilder bounds).
        lower = self._bounds[:, 0]
        upper = self._bounds[:, 1]
        samples = np.clip(samples, lower, upper)

        # Binary-ish features (gaze_off_road, head_distracted) are rounded
        # with a soft threshold to keep some intermediate "uncertain" values
        # for embedding richness, while staying within [0, 1].
        for binary_idx in (10, 14):
            samples[:, binary_idx] = np.clip(samples[:, binary_idx], 0.0, 1.0)

        logger.info(
            "Generated %d synthetic samples across %d driver archetypes.",
            n_samples,
            n_archetypes,
        )
        return samples.astype(np.float32)



# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class EmbeddingTrainer:
    """Trains and persists the Driver Digital Twin embedding encoder.

    Orchestrates synthetic data generation, autoencoder training, holdout
    reconstruction evaluation, and serialisation of the final
    ``StandardScaler -> _EncoderTransformer`` pipeline.

    Attributes:
        feature_dim: Input feature dimensionality (21).
        embedding_dim: Output embedding dimensionality (default 128).
        random_state: Random seed for reproducibility.
        max_iter: Maximum training iterations for the autoencoder MLP.
        test_size: Fraction of samples held out for reconstruction evaluation.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        random_state: int = DEFAULT_RANDOM_STATE,
        max_iter: int = DEFAULT_MAX_ITER,
        test_size: float = DEFAULT_TEST_SIZE,
    ) -> None:
        """Initialises the embedding trainer.

        Args:
            feature_dim: Dimensionality of input feature vectors.
            embedding_dim: Target embedding (hidden layer) dimensionality.
            random_state: Seed used for data generation, train/test split,
                and MLP weight initialisation.
            max_iter: Maximum number of training iterations for the
                autoencoder MLP.
            test_size: Fraction (0, 1) of samples reserved for holdout
                reconstruction-error evaluation.

        Raises:
            ValueError: If any dimensional or fractional argument is invalid.
        """
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")
        if not (0.0 < test_size < 1.0):
            raise ValueError(f"test_size must be in (0, 1), got {test_size}.")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}.")

        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.max_iter = max_iter
        self.test_size = test_size

        self._pipeline: Optional[Pipeline] = None
        self._train_mse: Optional[float] = None
        self._val_mse: Optional[float] = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def generate_data(self, n_samples: int) -> np.ndarray:
        """Generates the synthetic training dataset.

        Args:
            n_samples: Total number of synthetic samples to generate before
                the train/validation split.

        Returns:
            np.ndarray: Array of shape ``(n_samples, feature_dim)``.
        """
        generator = SyntheticDriverDataGenerator(
            feature_dim=self.feature_dim,
            random_state=self.random_state,
        )
        return generator.generate(n_samples)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "EmbeddingTrainer":
        """Fits the autoencoder pipeline on the provided feature matrix.

        Splits ``X`` into train/validation subsets, fits a
        ``StandardScaler`` followed by an ``MLPRegressor`` autoencoder
        (``feature_dim -> embedding_dim -> feature_dim``), extracts the
        encoder half via :class:`_EncoderTransformer`, and records
        reconstruction MSE on both subsets for diagnostics.

        Args:
            X: Training feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            EmbeddingTrainer: ``self``, for method chaining.

        Raises:
            ValueError: If ``X`` does not have the expected number of columns
                or contains fewer than 2 samples.
            RuntimeError: If autoencoder training fails.
        """
        if X.ndim != 2 or X.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected X of shape (n_samples, {self.feature_dim}), got {X.shape}."
            )
        if X.shape[0] < 2:
            raise ValueError("At least 2 samples are required to fit the encoder.")

        logger.info(
            "Starting embedding autoencoder training: n_samples=%d, "
            "feature_dim=%d -> embedding_dim=%d, max_iter=%d",
            X.shape[0],
            self.feature_dim,
            self.embedding_dim,
            self.max_iter,
        )

        X_train, X_val = train_test_split(
            X, test_size=self.test_size, random_state=self.random_state, shuffle=True
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        autoencoder = MLPRegressor(
            hidden_layer_sizes=(self.embedding_dim,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=self.max_iter,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1,
            random_state=self.random_state,
            shuffle=True,
        )

        start = time.time()
        try:
            autoencoder.fit(X_train_scaled, X_train_scaled)
        except Exception as exc:
            logger.error("Autoencoder training failed: %s", exc)
            raise RuntimeError("Failed to train embedding autoencoder.") from exc
        elapsed = time.time() - start

        logger.info(
            "Autoencoder training completed in %.2fs (%d iterations, "
            "converged=%s).",
            elapsed,
            autoencoder.n_iter_,
            autoencoder.n_iter_ < self.max_iter,
        )

        encoder = _EncoderTransformer.from_mlp(autoencoder)

        # Reconstruction MSE diagnostics (train + holdout).
        recon_train = autoencoder.predict(X_train_scaled)
        recon_val = autoencoder.predict(X_val_scaled)
        self._train_mse = float(mean_squared_error(X_train_scaled, recon_train))
        self._val_mse = float(mean_squared_error(X_val_scaled, recon_val))

        logger.info(
            "Reconstruction MSE — train=%.6f, validation=%.6f",
            self._train_mse,
            self._val_mse,
        )

        self._pipeline = Pipeline([
            ("scaler", scaler),
            ("encoder", encoder),
        ])

        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_path: Path) -> Path:
        
        if self._pipeline is None:
            raise RuntimeError("Cannot save: call fit() before save().")

        output_path = Path(output_path)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._pipeline, output_path)
            logger.info("Saved embedding encoder pipeline to %s", output_path)
        except OSError as exc:
            logger.error("Failed to save embedding encoder to %s: %s", output_path, exc)
            raise RuntimeError(f"Failed to write model artifact to {output_path}") from exc

        return output_path


    @property
    def pipeline(self) -> Optional[Pipeline]:
        return self._pipeline

    @property
    def train_mse(self) -> Optional[float]:

        return self._train_mse

    @property
    def val_mse(self) -> Optional[float]:
        
        return self._val_mse

    def smoke_test(self, n_probe: int = 4) -> bool:
 
        if self._pipeline is None:
            raise RuntimeError("Cannot smoke-test: call fit() before smoke_test().")

        generator = SyntheticDriverDataGenerator(
            feature_dim=self.feature_dim, random_state=self.random_state + 1
        )
        probe = generator.generate(n_probe)
        embeddings = self._pipeline.transform(probe)

        if embeddings.shape != (n_probe, self.embedding_dim):
            logger.error(
                "Smoke test failed: expected shape %s, got %s",
                (n_probe, self.embedding_dim),
                embeddings.shape,
            )
            return False

        if not np.all(np.isfinite(embeddings)):
            logger.error("Smoke test failed: non-finite values in embeddings.")
            return False

        a, b = embeddings[0], embeddings[1]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cosine = float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0
        if not np.isfinite(cosine):
            logger.error("Smoke test failed: cosine similarity is not finite.")
            return False

        logger.info(
            "Smoke test passed: embeddings shape=%s, sample cosine similarity=%.4f",
            embeddings.shape,
            cosine,
        )
        return True



def resolve_output_path(explicit_path: Optional[str]) -> Path:
    
    if explicit_path:
        return Path(explicit_path)

    try:
        return get_model_path(MLConstants.EMBEDDING_MODEL_NAME)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "Could not resolve output path from settings (%s); using default "
            "backend/ml/models_saved/driver_face_encoder.joblib", exc
        )
        return Path("backend/ml/models_saved/driver_face_encoder.joblib")




def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
 
    parser = argparse.ArgumentParser(
        description="Train the CogniDrive Driver Digital Twin embedding encoder (offline).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of synthetic training samples (default: {DEFAULT_N_SAMPLES}).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help=f"Output embedding dimensionality (default: {DEFAULT_EMBEDDING_DIM}).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help=f"Maximum autoencoder training iterations (default: {DEFAULT_MAX_ITER}).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Validation split fraction (default: {DEFAULT_TEST_SIZE}).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE}).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Destination .joblib path. Defaults to "
            "<MODEL_DIR>/driver_face_encoder.joblib (see app.config.Settings)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
 
    args = parse_args(argv)

    logger.info("=" * 70)
    logger.info("CogniDrive — Driver Digital Twin Embedding Encoder Training")
    logger.info("=" * 70)

    try:
        trainer = EmbeddingTrainer(
            feature_dim=FEATURE_DIM,
            embedding_dim=args.embedding_dim,
            random_state=args.random_state,
            max_iter=args.max_iter,
            test_size=args.test_size,
        )

        X = trainer.generate_data(args.n_samples)
        trainer.fit(X)

        if not trainer.smoke_test():
            logger.error("Smoke test failed — refusing to persist a broken model.")
            return 1

        output_path = resolve_output_path(args.output_path)
        trainer.save(output_path)

        logger.info(
            "Training complete. train_mse=%.6f, val_mse=%.6f, "
            "embedding_dim=%d, feature_names=%s",
            trainer.train_mse,
            trainer.val_mse,
            args.embedding_dim,
            FEATURE_NAMES[:3] + ["..."],
        )
        return 0

    except Exception as exc:
        logger.error("Embedding training pipeline failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())