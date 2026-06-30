"""Recommendation Engine — behavioural similarity search.

Implements cosine-similarity-based nearest-neighbour search over Driver
Digital Twin embeddings (see
:mod:`backend.ml.digital_twin.driver_embedding`), the foundation the rest
of the recommendation engine (:mod:`clustering`,
:mod:`recommendation_engine`) builds on for cold-start initialisation and
"drivers like you" style behavioural insights.

Mathematical foundation
------------------------
Given a query embedding ``q`` and a corpus of driver embeddings
``{e_1, ..., e_n}``, cosine similarity is:

.. math::

    \\text{sim}(q, e_i) = \\frac{q \\cdot e_i}{\\lVert q \\rVert \\, \\lVert e_i \\rVert}

This module batches the computation as a single matrix-vector product
against an L2-normalised embedding matrix for O(n*d) top-k search — no
external vector database or ANN library is required at CogniDrive's scale
(expected corpus size: tens to low hundreds of drivers on a single edge
device), so brute-force search with NumPy is both sufficient and avoids
adding a heavyweight dependency.

All computation is local NumPy; nothing in this module performs network I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from backend.ml.digital_twin.driver_embedding import EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-8

#: Default number of nearest neighbours returned by similarity search.
DEFAULT_TOP_K: int = 5


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SimilarityResult:
    """A single nearest-neighbour match from a similarity search.

    Attributes:
        driver_id: Primary key of the matched driver.
        similarity: Cosine similarity to the query embedding, ``[-1, 1]``.
        rank: 1-indexed rank of this result within the returned top-k list
            (``1`` = most similar).
    """

    driver_id: int
    similarity: float
    rank: int


# ---------------------------------------------------------------------------
# Similarity Engine
# ---------------------------------------------------------------------------


class SimilarityEngine:
    """Brute-force cosine-similarity nearest-neighbour search over driver embeddings.

    Maintains an in-memory corpus of ``{driver_id: embedding}`` and serves
    top-k similarity queries against it. The corpus is intentionally kept
    as a simple dictionary plus a lazily-rebuilt cached matrix (rather than
    a persistent index structure) since CogniDrive's edge deployment target
    has at most a handful to a few hundred driver profiles — well within
    brute-force NumPy's comfortable range, and far simpler to reason about
    / keep 100% offline than introducing an ANN library (e.g. FAISS, which
    is overkill at this corpus size and adds a non-trivial native
    dependency to cross-compile for ARM edge targets like Jetson Nano).

    Attributes:
        embedding_dim: Expected dimensionality of all embeddings in the
            corpus.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM) -> None:
        """Initialises an empty similarity search corpus.

        Args:
            embedding_dim: Expected embedding dimensionality. Must match
                :data:`backend.ml.digital_twin.driver_embedding.EMBEDDING_DIM`.

        Raises:
            ValueError: If ``embedding_dim`` is not positive.
        """
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")

        self.embedding_dim = embedding_dim
        self._corpus: Dict[int, np.ndarray] = {}
        self._matrix_cache: Optional[np.ndarray] = None
        self._matrix_norms_cache: Optional[np.ndarray] = None
        self._driver_id_order: List[int] = []
        self._dirty = True

        logger.info(
            "SimilarityEngine initialised (embedding_dim=%d).", embedding_dim
        )

    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------

    def add_or_update(self, driver_id: int, embedding: np.ndarray) -> None:
        """Adds a new driver embedding to the corpus, or updates an existing one.

        Args:
            driver_id: Primary key of the driver profile.
            embedding: The driver's current embedding, shape
                ``(embedding_dim,)``.

        Raises:
            ValueError: If ``embedding`` has the wrong shape, contains
                non-finite values, or ``driver_id`` is invalid.
        """
        if driver_id <= 0:
            raise ValueError(f"driver_id must be a positive integer, got {driver_id}.")
        if embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"embedding must have shape ({self.embedding_dim},), got {embedding.shape}."
            )
        if not np.all(np.isfinite(embedding)):
            raise ValueError("embedding contains non-finite values.")

        self._corpus[driver_id] = embedding.astype(np.float32).copy()
        self._dirty = True

        logger.debug("Corpus updated for driver_id=%d (corpus size=%d).", driver_id, len(self._corpus))

    def remove(self, driver_id: int) -> bool:
        """Removes a driver's embedding from the corpus.

        Args:
            driver_id: Primary key of the driver profile to remove.

        Returns:
            bool: True if the driver was present and removed.
        """
        existed = self._corpus.pop(driver_id, None) is not None
        if existed:
            self._dirty = True
        return existed

    def corpus_size(self) -> int:
        """Returns the number of driver embeddings currently in the corpus.

        Returns:
            int: Corpus size.
        """
        return len(self._corpus)

    def get_embedding(self, driver_id: int) -> Optional[np.ndarray]:
        """Retrieves a driver's embedding from the corpus, if present.

        Args:
            driver_id: Primary key of the driver profile.

        Returns:
            Optional[np.ndarray]: Copy of the embedding, or ``None`` if not
            in the corpus.
        """
        emb = self._corpus.get(driver_id)
        return emb.copy() if emb is not None else None

    # ------------------------------------------------------------------
    # Matrix cache (rebuilt lazily on corpus mutation)
    # ------------------------------------------------------------------

    def _rebuild_matrix_cache(self) -> None:
        """Rebuilds the cached embedding matrix and L2-norm vector.

        This is the only O(n*d) operation triggered by corpus mutation;
        subsequent queries against an unchanged corpus reuse the cache and
        cost only O(n*d) for the similarity dot product itself (no
        redundant norm recomputation).
        """
        if not self._corpus:
            self._matrix_cache = np.zeros((0, self.embedding_dim), dtype=np.float32)
            self._matrix_norms_cache = np.zeros((0,), dtype=np.float32)
            self._driver_id_order = []
            self._dirty = False
            return

        self._driver_id_order = list(self._corpus.keys())
        self._matrix_cache = np.stack(
            [self._corpus[did] for did in self._driver_id_order], axis=0
        ).astype(np.float32)
        self._matrix_norms_cache = np.linalg.norm(self._matrix_cache, axis=1).astype(np.float32)
        self._dirty = False

        logger.debug(
            "Rebuilt similarity matrix cache: corpus_size=%d.", len(self._driver_id_order)
        )

    def _ensure_fresh_cache(self) -> None:
        """Rebuilds the matrix cache if the corpus has been mutated since the last build."""
        if self._dirty:
            self._rebuild_matrix_cache()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        exclude_driver_id: Optional[int] = None,
    ) -> List[SimilarityResult]:
        """Finds the top-k most similar driver embeddings to a query embedding.

        Args:
            embedding: Query embedding, shape ``(embedding_dim,)``.
            top_k: Maximum number of results to return.
            exclude_driver_id: If provided, this driver is excluded from
                results (typically used to exclude the querying driver
                themselves when searching "drivers like me").

        Returns:
            List[SimilarityResult]: Up to ``top_k`` results, sorted by
            descending similarity, with ``rank`` populated 1-indexed.
            Returns an empty list if the corpus is empty (after exclusion).

        Raises:
            ValueError: If ``embedding`` has the wrong shape, contains
                non-finite values, or ``top_k`` is not positive.
        """
        if embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"embedding must have shape ({self.embedding_dim},), got {embedding.shape}."
            )
        if not np.all(np.isfinite(embedding)):
            raise ValueError("embedding contains non-finite values.")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")

        self._ensure_fresh_cache()

        if self._matrix_cache is None or self._matrix_cache.shape[0] == 0:
            return []

        query_norm = float(np.linalg.norm(embedding))
        if query_norm < _EPS:
            logger.warning("Query embedding has near-zero norm; returning no matches.")
            return []

        dots = self._matrix_cache @ embedding.astype(np.float32)
        denom = self._matrix_norms_cache * query_norm + _EPS
        similarities = dots / denom
        similarities = np.clip(similarities, -1.0, 1.0)

        candidates: List[Tuple[int, float]] = [
            (self._driver_id_order[i], float(similarities[i]))
            for i in range(len(self._driver_id_order))
            if self._driver_id_order[i] != exclude_driver_id
        ]
        candidates.sort(key=lambda pair: pair[1], reverse=True)

        top = candidates[:top_k]
        results = [
            SimilarityResult(driver_id=did, similarity=sim, rank=i + 1)
            for i, (did, sim) in enumerate(top)
        ]

        logger.info(
            "Similarity query returned %d results (top_k=%d, corpus_size=%d, "
            "excluded=%s).",
            len(results),
            top_k,
            len(self._driver_id_order),
            exclude_driver_id,
        )
        return results

    def query_by_driver_id(
        self, driver_id: int, top_k: int = DEFAULT_TOP_K
    ) -> List[SimilarityResult]:
        """Finds the top-k drivers most similar to an existing corpus member.

        Convenience wrapper around :meth:`query` that looks up the query
        driver's own embedding and automatically excludes them from their
        own results.

        Args:
            driver_id: Primary key of the driver to find neighbours for.
            top_k: Maximum number of results to return.

        Returns:
            List[SimilarityResult]: Up to ``top_k`` nearest neighbours,
            excluding ``driver_id`` itself.

        Raises:
            KeyError: If ``driver_id`` is not present in the corpus.
        """
        embedding = self._corpus.get(driver_id)
        if embedding is None:
            raise KeyError(f"driver_id={driver_id} is not present in the similarity corpus.")

        return self.query(embedding, top_k=top_k, exclude_driver_id=driver_id)

    @staticmethod
    def pairwise_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Computes cosine similarity between two arbitrary embedding vectors.

        Thin convenience wrapper exposed at the engine level (in addition
        to :meth:`backend.ml.digital_twin.driver_embedding.DriverEmbeddingManager.cosine_similarity`)
        so callers that only import :mod:`similarity_engine` don't need a
        second import for one-off comparisons.

        Args:
            a: First embedding vector.
            b: Second embedding vector, must match ``a``'s shape.

        Returns:
            float: Cosine similarity in ``[-1, 1]``. Returns ``0.0`` if
            either vector has near-zero norm.

        Raises:
            ValueError: If ``a`` and ``b`` have mismatched shapes.
        """
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}.")

        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < _EPS or norm_b < _EPS:
            return 0.0

        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        return max(-1.0, min(1.0, similarity))

    def similarity_matrix(self) -> Tuple[np.ndarray, List[int]]:
        """Computes the full pairwise cosine similarity matrix for the corpus.

        Useful for dashboard heatmaps or as a precursor step to clustering
        diagnostics. Cost is O(n^2) in corpus size, so this is intended for
        the expected small-to-moderate edge-device corpus sizes, not
        large-scale fleets.

        Returns:
            Tuple[np.ndarray, List[int]]: ``(matrix, driver_id_order)``
            where ``matrix`` has shape ``(n, n)`` and ``matrix[i, j]`` is
            the cosine similarity between ``driver_id_order[i]`` and
            ``driver_id_order[j]``.
        """
        self._ensure_fresh_cache()

        if self._matrix_cache is None or self._matrix_cache.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float32), []

        norms = self._matrix_norms_cache.reshape(-1, 1)
        normalized = self._matrix_cache / (norms + _EPS)
        sim_matrix = normalized @ normalized.T
        sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

        return sim_matrix.astype(np.float32), list(self._driver_id_order)
