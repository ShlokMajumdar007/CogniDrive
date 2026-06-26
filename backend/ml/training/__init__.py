from __future__ import annotations

try:
    from backend.ml.digital_twin.driver_embedding import (
        DriverEmbeddingManager,
        EMBEDDING_DIM,
    )
except ImportError:
    from ml.digital_twin.driver_embedding import (  # type: ignore[no-redef]
        DriverEmbeddingManager,
        EMBEDDING_DIM,
    )

__all__ = [
    "DriverEmbeddingManager",
    "EMBEDDING_DIM",
]

