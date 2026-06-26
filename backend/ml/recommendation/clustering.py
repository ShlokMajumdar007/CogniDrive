"""DriverClusteringEngine — Behavioral Persona Discovery and Cohort Assignment.

Groups drivers into behavioral clusters (personas) using the 128-D behavioral
embeddings produced by EmbeddingModel. Clustering enables:

    - Cold-start initialization: new drivers inherit thresholds from their cluster.
    - Similar driver retrieval: find drivers with similar behavioral patterns.
    - Personalized recommendations: calibrate advice based on cluster norms.
    - Risk cohort analysis: identify high-risk driver segments.

Algorithm pipeline:
    1. Collect behavioral embeddings from all enrolled drivers.
    2. Apply PCA to reduce to 32 dimensions (noise reduction, speed).
    3. Fit KMeans (K=5 default) on the reduced space.
    4. Assign new drivers to the nearest centroid.
    5. Use NearestNeighbors for similar-driver retrieval.

Cluster personas (default K=5):
    - Cluster 0: High-attention, low-stress "focused" drivers.
    - Cluster 1: Moderate attention, elevated stress "pressured" drivers.
    - Cluster 2: Fatigue-prone, low blink-rate "tired" drivers.
    - Cluster 3: Distraction-prone, high gaze-off-road drivers.
    - Cluster 4: Aggressive, high-risk drivers.

Note: Cluster semantics are emergent — the above are illustrative labels
assigned post-hoc based on cluster centroid feature analysis.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize as sk_normalize
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger_pre = logging.getLogger("CogniDrive.DriverClusteringEngine")
    logger_pre.critical(
        "scikit-learn not available. DriverClusteringEngine will run in fallback mode."
    )

logger = logging.getLogger("CogniDrive.DriverClusteringEngine")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_N_CLUSTERS: int = 5
DEFAULT_PCA_COMPONENTS: int = 32
DEFAULT_N_NEIGHBORS: int = 5
DEFAULT_MODEL_DIR: Path = Path("backend/ml/models_saved")

CLUSTER_PERSONA_NAMES: Dict[int, str] = {
    0: "focused",
    1: "pressured",
    2: "fatigued",
    3: "distracted",
    4: "aggressive",
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClusterAssignmentResult:
    """Result of assigning a driver to a behavioral cluster.

    Attributes:
        driver_id: Primary key of the assigned driver.
        cluster_id: Assigned cluster index.
        persona_name: Human-readable persona label.
        distance_to_centroid: Euclidean distance to the centroid in PCA space.
        similarity_to_centroid: Cosine similarity to the centroid.
        cluster_size: Number of drivers in this cluster.
    """
    driver_id: int
    cluster_id: int
    persona_name: str
    distance_to_centroid: float
    similarity_to_centroid: float
    cluster_size: int = 0


@dataclass
class SimilarDriverResult:
    """A single similar-driver search result.

    Attributes:
        driver_id: Primary key of the similar driver.
        distance: Distance in PCA embedding space.
        similarity_rank: Rank position (1 = most similar).
    """
    driver_id: int
    distance: float
    similarity_rank: int


# ---------------------------------------------------------------------------
# DriverClusteringEngine
# ---------------------------------------------------------------------------


class DriverClusteringEngine:
    """PCA + KMeans clustering engine for driver behavioral persona discovery.

    Thread-safe. Supports online cluster assignment for new drivers without
    re-fitting the full model. The model is retrained periodically via
    ``fit()`` when enough new drivers are enrolled.

    Attributes:
        _pca: Fitted scikit-learn PCA instance.
        _kmeans: Fitted scikit-learn KMeans instance.
        _nn: Fitted NearestNeighbors for similar-driver retrieval.
        _is_fitted: True if the model has been fitted.
        _driver_ids: List of driver IDs in the same order as the corpus.
        _pca_embeddings: PCA-reduced embedding matrix (N, PCA_COMPONENTS).
        _cluster_labels: Cluster assignments for each driver.
        _lock: Threading lock for thread-safe inference.
    """

    def __init__(
        self,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        pca_components: int = DEFAULT_PCA_COMPONENTS,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
    ) -> None:
        """Initialises the clustering engine.

        Args:
            n_clusters: Number of KMeans clusters.
            pca_components: PCA output dimensionality.
            n_neighbors: KNN search radius.
        """
        self._n_clusters = n_clusters
        self._pca_components = pca_components
        self._n_neighbors = n_neighbors

        self._pca: Optional[Any] = None
        self._kmeans: Optional[Any] = None
        self._nn: Optional[Any] = None

        self._is_fitted: bool = False
        self._driver_ids: List[int] = []
        self._pca_embeddings: Optional[np.ndarray] = None
        self._cluster_labels: Optional[np.ndarray] = None
        self._cluster_sizes: Dict[int, int] = {}

        self._lock: threading.Lock = threading.Lock()

        if not _SKLEARN_AVAILABLE:
            logger.error(
                "DriverClusteringEngine: scikit-learn not available. "
                "Clustering features will be disabled."
            )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        driver_ids: List[int],
        embeddings: np.ndarray,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Fits the PCA → KMeans pipeline on the driver embedding corpus.

        Requires at least ``n_clusters`` drivers to fit. Falls back to
        deterministic assignment if fewer drivers are available.

        Args:
            driver_ids: List of driver primary keys, length N.
            embeddings: ``(N, 128)`` float32 behavioral embedding matrix.
            random_state: Random seed for reproducibility.

        Returns:
            Dict[str, Any]: Fit summary with inertia, cluster sizes, etc.

        Raises:
            ValueError: If embeddings shape is incompatible.
        """
        if not _SKLEARN_AVAILABLE:
            logger.error("DriverClusteringEngine.fit: scikit-learn unavailable.")
            return {"fitted": False, "reason": "scikit-learn not installed"}

        if len(driver_ids) != embeddings.shape[0]:
            raise ValueError(
                f"driver_ids length ({len(driver_ids)}) must match "
                f"embeddings rows ({embeddings.shape[0]})."
            )

        n_drivers = len(driver_ids)
        effective_clusters = min(self._n_clusters, n_drivers)

        if n_drivers < 2:
            logger.warning(
                "DriverClusteringEngine.fit: Too few drivers (%d) to cluster.", n_drivers
            )
            return {"fitted": False, "reason": f"Too few drivers: {n_drivers}"}

        with self._lock:
            # L2 normalize before PCA
            X = sk_normalize(embeddings.astype(np.float32), norm="l2")

            # PCA reduction
            effective_pca = min(self._pca_components, n_drivers, X.shape[1])
            self._pca = PCA(n_components=effective_pca, random_state=random_state)
            X_pca = self._pca.fit_transform(X)

            # KMeans
            self._kmeans = KMeans(
                n_clusters=effective_clusters,
                random_state=random_state,
                n_init=10,
                max_iter=300,
            )
            labels = self._kmeans.fit_predict(X_pca)

            # NearestNeighbors for similar-driver retrieval
            k_nn = min(self._n_neighbors + 1, n_drivers)
            self._nn = NearestNeighbors(
                n_neighbors=k_nn, metric="euclidean", algorithm="ball_tree"
            )
            self._nn.fit(X_pca)

            # Store state
            self._driver_ids = list(driver_ids)
            self._pca_embeddings = X_pca
            self._cluster_labels = labels
            self._is_fitted = True

            # Compute cluster sizes
            unique, counts = np.unique(labels, return_counts=True)
            self._cluster_sizes = {int(k): int(v) for k, v in zip(unique, counts)}

        inertia = float(self._kmeans.inertia_)
        logger.info(
            "DriverClusteringEngine.fit: Fitted K=%d clusters on %d drivers "
            "(pca=%d, inertia=%.2f).",
            effective_clusters,
            n_drivers,
            effective_pca,
            inertia,
        )

        return {
            "fitted": True,
            "n_drivers": n_drivers,
            "n_clusters": effective_clusters,
            "pca_components": effective_pca,
            "inertia": inertia,
            "cluster_sizes": self._cluster_sizes,
            "fitted_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Cluster assignment
    # ------------------------------------------------------------------

    def assign_cluster(
        self, driver_id: int, embedding: np.ndarray
    ) -> ClusterAssignmentResult:
        """Assigns a driver to the nearest cluster centroid.

        Works in online mode — does not require re-fitting the full model.

        Args:
            driver_id: Primary key of the DriverProfile.
            embedding: 128-D behavioral embedding, shape (128,).

        Returns:
            ClusterAssignmentResult: Cluster assignment and distance metrics.
        """
        if not self._is_fitted:
            return ClusterAssignmentResult(
                driver_id=driver_id,
                cluster_id=0,
                persona_name=CLUSTER_PERSONA_NAMES[0],
                distance_to_centroid=0.0,
                similarity_to_centroid=1.0,
            )

        with self._lock:
            X = sk_normalize(embedding.reshape(1, -1).astype(np.float32), norm="l2")
            X_pca = self._pca.transform(X)
            cluster_id = int(self._kmeans.predict(X_pca)[0])
            centroid = self._kmeans.cluster_centers_[cluster_id]
            distance = float(np.linalg.norm(X_pca[0] - centroid))

            # Cosine similarity to centroid
            xp = X_pca[0]
            c = centroid
            norm_x = float(np.linalg.norm(xp))
            norm_c = float(np.linalg.norm(c))
            if norm_x > 1e-8 and norm_c > 1e-8:
                sim = float(np.clip(np.dot(xp, c) / (norm_x * norm_c), -1.0, 1.0))
            else:
                sim = 0.0

        persona = CLUSTER_PERSONA_NAMES.get(cluster_id, f"cluster_{cluster_id}")

        return ClusterAssignmentResult(
            driver_id=driver_id,
            cluster_id=cluster_id,
            persona_name=persona,
            distance_to_centroid=round(distance, 6),
            similarity_to_centroid=round(sim, 6),
            cluster_size=self._cluster_sizes.get(cluster_id, 0),
        )

    # ------------------------------------------------------------------
    # Similar driver retrieval
    # ------------------------------------------------------------------

    def find_similar_drivers(
        self,
        embedding: np.ndarray,
        k: int = DEFAULT_N_NEIGHBORS,
        exclude_driver_id: Optional[int] = None,
    ) -> List[SimilarDriverResult]:
        """Finds the K most behaviorally similar drivers in the corpus.

        Args:
            embedding: Query driver's 128-D behavioral embedding.
            k: Number of similar drivers to return.
            exclude_driver_id: Optionally exclude a specific driver (e.g. self).

        Returns:
            List[SimilarDriverResult]: K similar drivers sorted by proximity.
        """
        if not self._is_fitted or self._nn is None:
            return []

        with self._lock:
            X = sk_normalize(embedding.reshape(1, -1).astype(np.float32), norm="l2")
            X_pca = self._pca.transform(X)

            k_query = min(k + 1, len(self._driver_ids))
            distances, indices = self._nn.kneighbors(X_pca, n_neighbors=k_query)

            results = []
            rank = 1
            for dist, idx in zip(distances[0], indices[0]):
                driver_id = self._driver_ids[idx]
                if driver_id == exclude_driver_id:
                    continue
                results.append(
                    SimilarDriverResult(
                        driver_id=driver_id,
                        distance=round(float(dist), 6),
                        similarity_rank=rank,
                    )
                )
                rank += 1
                if len(results) >= k:
                    break

        return results

    # ------------------------------------------------------------------
    # Cluster analytics
    # ------------------------------------------------------------------

    def get_cluster_analytics(self) -> Dict[str, Any]:
        """Returns aggregate cluster analytics.

        Returns:
            Dict[str, Any]: Cluster sizes, persona names, fit status.
        """
        if not self._is_fitted:
            return {"fitted": False, "n_clusters": 0, "clusters": []}

        clusters = []
        for cluster_id, size in sorted(self._cluster_sizes.items()):
            clusters.append({
                "cluster_id": cluster_id,
                "persona_name": CLUSTER_PERSONA_NAMES.get(cluster_id, f"cluster_{cluster_id}"),
                "size": size,
            })

        return {
            "fitted": True,
            "n_clusters": self._n_clusters,
            "n_drivers": len(self._driver_ids),
            "pca_components": self._pca_components,
            "clusters": clusters,
        }

    # ------------------------------------------------------------------
    # Cold-start initialization
    # ------------------------------------------------------------------

    def get_cold_start_thresholds(
        self,
        embedding: np.ndarray,
        cluster_threshold_map: Dict[int, Dict[str, float]],
    ) -> Dict[str, float]:
        """Returns thresholds for a new driver based on their cluster assignment.

        Args:
            embedding: New driver's behavioral embedding.
            cluster_threshold_map: Dict mapping cluster_id → threshold dict.

        Returns:
            Dict[str, float]: Threshold map for the assigned cluster.
        """
        result = self.assign_cluster(driver_id=0, embedding=embedding)
        cluster_thresholds = cluster_threshold_map.get(
            result.cluster_id, cluster_threshold_map.get(0, {})
        )
        logger.info(
            "DriverClusteringEngine.get_cold_start_thresholds: "
            "New driver → cluster=%d (%s), %d thresholds.",
            result.cluster_id,
            result.persona_name,
            len(cluster_thresholds),
        )
        return cluster_thresholds

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_dir: Path = DEFAULT_MODEL_DIR) -> bool:
        """Saves the fitted clustering model to disk.

        Args:
            model_dir: Directory where model files are saved.

        Returns:
            bool: True on success.
        """
        if not _JOBLIB_AVAILABLE or not self._is_fitted:
            return False
        try:
            model_dir = Path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._pca, model_dir / "clustering_pca.joblib")
            joblib.dump(self._kmeans, model_dir / "clustering_kmeans.joblib")
            joblib.dump(self._nn, model_dir / "clustering_nn.joblib")
            joblib.dump(
                {
                    "driver_ids": self._driver_ids,
                    "pca_embeddings": self._pca_embeddings,
                    "cluster_labels": self._cluster_labels,
                    "cluster_sizes": self._cluster_sizes,
                    "n_clusters": self._n_clusters,
                    "pca_components": self._pca_components,
                },
                model_dir / "clustering_state.joblib",
            )
            logger.info("DriverClusteringEngine.save: Saved to %s.", model_dir)
            return True
        except Exception as exc:
            logger.error("DriverClusteringEngine.save: Failed — %s.", exc)
            return False

    def load(self, model_dir: Path = DEFAULT_MODEL_DIR) -> bool:
        """Loads a previously saved clustering model from disk.

        Args:
            model_dir: Directory containing saved model files.

        Returns:
            bool: True if all model files were loaded successfully.
        """
        if not _JOBLIB_AVAILABLE:
            return False
        try:
            model_dir = Path(model_dir)
            self._pca = joblib.load(model_dir / "clustering_pca.joblib")
            self._kmeans = joblib.load(model_dir / "clustering_kmeans.joblib")
            self._nn = joblib.load(model_dir / "clustering_nn.joblib")
            state = joblib.load(model_dir / "clustering_state.joblib")
            self._driver_ids = state["driver_ids"]
            self._pca_embeddings = state["pca_embeddings"]
            self._cluster_labels = state["cluster_labels"]
            self._cluster_sizes = state["cluster_sizes"]
            self._n_clusters = state["n_clusters"]
            self._pca_components = state["pca_components"]
            self._is_fitted = True
            logger.info(
                "DriverClusteringEngine.load: Loaded model from %s "
                "(%d drivers, %d clusters).",
                model_dir,
                len(self._driver_ids),
                self._n_clusters,
            )
            return True
        except Exception as exc:
            logger.error("DriverClusteringEngine.load: Failed — %s.", exc)
            return False
