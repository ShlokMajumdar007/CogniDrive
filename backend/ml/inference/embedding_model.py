"""Driver Embedding Model — Digital Twin vector generation.

Generates a 128-dimensional embedding vector that represents a driver's
unique biometric signature. This embedding is stored in the Digital Twin
and used for:

    - Personalised threshold adaptation (distance from driver baseline).
    - Similar driver search (cosine similarity across the driver corpus).
    - Clustering (K-Means on the 128-D space to identify risk cohorts).
    - Drift detection (embedding distance over time → behavioural change).

Architecture:
    A lightweight autoencoder trained on sequences of 21-D feature vectors.
    At inference time, only the encoder half is used. The encoder is
    implemented as a scikit-learn ``Pipeline`` (StandardScaler + PCA + MLP)
    and serialised with ``joblib``.

    If no trained model file is found at the configured path the model
    falls back to a deterministic PCA-based projection that still produces
    a valid 128-D vector from population statistics — guaranteeing the
    pipeline never hard-fails on new deployments.

Typical usage::

    model = EmbeddingModel.get_instance()
    embedding = model.encode(feature_vector)  # → np.ndarray (128,)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — graceful degradation if not installed
# ---------------------------------------------------------------------------

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False
    logger.warning("joblib not installed — EmbeddingModel will use fallback PCA.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM: int = 128
FEATURE_DIM: int = 21

#: Default path where the trained encoder pipeline is stored.
DEFAULT_MODEL_PATH: Path = Path("backend/ml/models/embedding_encoder.joblib")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingResult:
    """Result produced by :class:`EmbeddingModel`.

    Attributes:
        embedding: 128-dimensional float32 embedding vector.
        is_fallback: True when the fallback PCA projection was used.
        model_version: String identifier of the model that produced this result.
        l2_norm: L2 norm of the embedding (useful for anomaly detection).
    """

    embedding: np.ndarray
    is_fallback: bool = False
    model_version: str = "1.0.0"
    l2_norm: float = 0.0

    def to_list(self) -> List[float]:
        """Returns the embedding as a Python list.

        Returns:
            List[float]: 128-element float list.
        """
        return self.embedding.tolist()

    def cosine_similarity(self, other: "EmbeddingResult") -> float:
        """Computes cosine similarity with another embedding.

        Args:
            other: Another :class:`EmbeddingResult`.

        Returns:
            float: Cosine similarity in [-1, 1].
        """
        a, b = self.embedding, other.embedding
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-8:
            return 0.0
        return float(np.dot(a, b) / denom)

    def euclidean_distance(self, other: "EmbeddingResult") -> float:
        """Computes Euclidean distance from another embedding.

        Args:
            other: Another :class:`EmbeddingResult`.

        Returns:
            float: L2 distance between the two embeddings.
        """
        return float(np.linalg.norm(self.embedding - other.embedding))


# ---------------------------------------------------------------------------
# Fallback PCA encoder
# ---------------------------------------------------------------------------


class _FallbackPCAEncoder:
    """Deterministic PCA-based projection used when the trained model is absent.

    Projects the 21-D feature vector into 128-D space using a fixed random
    projection matrix seeded from a constant, ensuring reproducibility
    across restarts. The output is L2-normalised to unit sphere.
    """

    def __init__(self) -> None:
        rng = np.random.default_rng(seed=42)
        self._W = rng.standard_normal((FEATURE_DIM, EMBEDDING_DIM)).astype(np.float32)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Projects input to 128-D and L2-normalises.

        Args:
            x: Input array of shape (21,).

        Returns:
            np.ndarray: Normalised 128-D embedding.
        """
        proj = x.astype(np.float32) @ self._W
        norm = np.linalg.norm(proj)
        if norm < 1e-8:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        return (proj / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """Thread-safe singleton that encodes 21-D feature vectors into 128-D embeddings.

    Loads a pre-trained ``joblib`` encoder pipeline on first use. Falls back
    to the deterministic PCA projection if the model file is not found.

    Attributes:
        _instance: Class-level singleton reference.
        _lock: Threading lock guarding singleton creation.
        _encoder: Loaded sklearn Pipeline or :class:`_FallbackPCAEncoder`.
        _is_fallback: True when the fallback encoder is active.
        _model_version: String version tag for the loaded model.
    """

    _instance: Optional["EmbeddingModel"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> None:
        """Initialises the model by loading from disk or using the fallback.

        Args:
            model_path: Path to the ``joblib``-serialised encoder pipeline.
        """
        self._model_path = model_path
        self._encoder, self._is_fallback = self._load_encoder(model_path)
        self._model_version: str = "1.0.0-fallback" if self._is_fallback else "1.0.0"
        logger.info(
            "EmbeddingModel initialised — fallback=%s, version=%s",
            self._is_fallback,
            self._model_version,
        )

    @staticmethod
    def _load_encoder(path: Path):  # type: ignore[return]
        """Attempts to load the sklearn pipeline from disk.

        Args:
            path: Joblib file path.

        Returns:
            Tuple of (encoder, is_fallback).
        """
        if _JOBLIB_AVAILABLE and path.exists():
            try:
                encoder = joblib.load(path)
                logger.info("EmbeddingModel loaded encoder from %s.", path)
                return encoder, False
            except Exception as exc:
                logger.warning(
                    "Failed to load encoder from %s: %s. Using fallback.", path, exc
                )
        else:
            logger.info(
                "Encoder not found at %s. Using deterministic fallback.", path
            )
        return _FallbackPCAEncoder(), True

    @classmethod
    def get_instance(
        cls, model_path: Path = DEFAULT_MODEL_PATH
    ) -> "EmbeddingModel":
        """Returns the singleton :class:`EmbeddingModel` instance.

        Args:
            model_path: Forwarded to ``__init__`` on first call.

        Returns:
            EmbeddingModel: Shared singleton.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_path=model_path)
        return cls._instance

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def encode(self, feature_vector: np.ndarray) -> EmbeddingResult:
        """Encodes a single 21-D feature vector into a 128-D embedding.

        Args:
            feature_vector: NumPy array of shape (21,), dtype float32.

        Returns:
            EmbeddingResult: 128-D embedding with metadata.
        """
        x = feature_vector.reshape(1, -1)

        try:
            if self._is_fallback:
                embedding = self._encoder.encode(x.squeeze(0))
            else:
                embedding = self._encoder.transform(x).squeeze(0).astype(np.float32)
        except Exception as exc:
            logger.error("EmbeddingModel.encode failed: %s. Returning zeros.", exc)
            embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        l2 = float(np.linalg.norm(embedding))
        return EmbeddingResult(
            embedding=embedding,
            is_fallback=self._is_fallback,
            model_version=self._model_version,
            l2_norm=l2,
        )

    def encode_batch(self, feature_matrix: np.ndarray) -> List[EmbeddingResult]:
        """Encodes a (N, 21) feature matrix into N embedding results.

        Args:
            feature_matrix: NumPy array of shape (N, 21), dtype float32.

        Returns:
            List[EmbeddingResult]: N embedding results in input order.
        """
        results = []
        for row in feature_matrix:
            results.append(self.encode(row))
        return results

    def encode_mean_embedding(
        self, feature_matrix: np.ndarray
    ) -> EmbeddingResult:
        """Encodes a session's feature matrix and returns the mean embedding.

        Useful for producing a single Digital Twin embedding that summarises
        an entire session.

        Args:
            feature_matrix: NumPy array of shape (N, 21), dtype float32.

        Returns:
            EmbeddingResult: Mean embedding across all session frames.
        """
        results = self.encode_batch(feature_matrix)
        if not results:
            return EmbeddingResult(
                embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                is_fallback=self._is_fallback,
                model_version=self._model_version,
            )
        stacked = np.stack([r.embedding for r in results], axis=0)
        mean_emb = stacked.mean(axis=0).astype(np.float32)
        # Renormalise the mean embedding to the unit sphere
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-8:
            mean_emb = mean_emb / norm
        return EmbeddingResult(
            embedding=mean_emb,
            is_fallback=self._is_fallback,
            model_version=self._model_version,
            l2_norm=float(np.linalg.norm(mean_emb)),
        )

    @property
    def is_fallback(self) -> bool:
        """True when the deterministic fallback encoder is active."""
        return self._is_fallback

    @property
    def model_version(self) -> str:
        """String version tag of the loaded model."""
        return self._model_version
