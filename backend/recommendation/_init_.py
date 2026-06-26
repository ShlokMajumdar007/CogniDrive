"""CogniDrive Recommendation Engine package.

Implements driver behavioural similarity search, PCA/K-Means archetype
clustering, and time-series trend analysis over Driver Digital Twin
embeddings and historical session data — entirely offline, analogous to a
classical recommender system but applied to driving-safety personalisation
rather than content recommendation.

Modules:
    similarity_engine: Cosine-similarity-based nearest-neighbour search over
        driver embeddings.
    clustering: PCA dimensionality reduction + K-Means archetype clustering
        for cold-start initialisation.
    recommendation_engine: Top-level orchestrator combining similarity,
        clustering, and time-series trend analysis into driver-facing
        recommendations.
"""

from __future__ import annotations

__all__: list = []
