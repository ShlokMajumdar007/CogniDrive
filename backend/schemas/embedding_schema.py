from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, computed_field


class EmbeddingStatistics(BaseModel):
    """Statistical summary values calculated over an embedding vector."""

    vector_norm: float = Field(..., description="L2 norm of the vector")
    vector_mean: float = Field(..., description="Mean value of the vector coordinates")
    vector_std: float = Field(..., description="Standard deviation of the vector coordinates")
    non_zero_elements: int = Field(..., description="Number of non-zero elements in the vector")
    sparsity: float = Field(..., description="Sparsity index, portion of zero elements (0.0 to 1.0)")

    model_config = ConfigDict(from_attributes=True)


class EmbeddingCreate(BaseModel):
    """Schema for validating driver embedding creation payloads."""

    driver_id: int = Field(..., description="Associated driver profile primary key")
    embedding_vector: List[float] = Field(..., description="Flat coordinates list of the embedding")
    embedding_dimension: int = Field(default=32, description="Target dimension count")
    calibration_samples: int = Field(default=0, ge=0, description="Number of frame samples used in calibration")
    embedding_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall quality score")

    @field_validator("embedding_dimension")
    @classmethod
    def validate_dimension(cls, v: int) -> int:
        """Validates that embedding dimension bounds are respected."""
        if v <= 0:
            raise ValueError("Embedding dimension must be greater than 0.")
        if v > 1024:
            raise ValueError("Embedding dimension exceeds maximum limit of 1024.")
        return v

    @model_validator(mode="after")
    def validate_vector_integrity(self) -> "EmbeddingCreate":
        """Enforces matching dimensions, empty bounds, and finite vector coordinate values."""
        vector = self.embedding_vector
        dimension = self.embedding_dimension

        if not vector:
            raise ValueError("Embedding vector cannot be empty.")

        if len(vector) != dimension:
            raise ValueError(
                f"Embedding vector coordinate length ({len(vector)}) does not match "
                f"embedding_dimension ({dimension})."
            )

        # Ensure all coordinate values are finite (no NaN or Inf values permitted)
        for idx, val in enumerate(vector):
            if not np.isfinite(val):
                raise ValueError(
                    f"Invalid vector coordinate detected at index {idx}: {val}. "
                    "All coordinates must be finite (no NaN or Infinity allowed)."
                )

        return self


class EmbeddingUpdate(BaseModel):
    """Schema for validating driver embedding updates.

    All fields are optional to support partial updates.
    """

    embedding_vector: Optional[List[float]] = Field(default=None)
    embedding_dimension: Optional[int] = Field(default=None)
    calibration_samples: Optional[int] = Field(default=None, ge=0)
    embedding_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    last_updated: Optional[datetime] = Field(default=None)

    @field_validator("embedding_dimension")
    @classmethod
    def validate_dimension(cls, v: Optional[int]) -> Optional[int]:
        """Validates that update dimension bounds are respected."""
        if v is not None:
            if v <= 0:
                raise ValueError("Embedding dimension must be greater than 0.")
            if v > 1024:
                raise ValueError("Embedding dimension exceeds maximum limit of 1024.")
        return v

    @model_validator(mode="after")
    def validate_vector_integrity_update(self) -> "EmbeddingUpdate":
        """Ensures updated dimensions match vector length and coordinate integrity remains intact."""
        vector = self.embedding_vector
        dimension = self.embedding_dimension

        if vector is not None:
            if not vector:
                raise ValueError("Embedding vector cannot be empty.")

            # Validate against dimension if both are updated
            if dimension is not None and len(vector) != dimension:
                raise ValueError(
                    f"Updated vector length ({len(vector)}) does not match "
                    f"updated embedding_dimension ({dimension})."
                )

            # Ensure coordinate values remain finite
            for idx, val in enumerate(vector):
                if not np.isfinite(val):
                    raise ValueError(
                        f"Invalid updated vector coordinate detected at index {idx}: {val}. "
                        "All coordinates must be finite."
                    )

        return self


class EmbeddingResponse(BaseModel):
    """Schema returned by APIs representing a driver's face mesh biometrics embedding."""

    id: int
    driver_id: int
    embedding_vector: List[float]
    embedding_dimension: int
    version: int
    calibration_samples: int
    embedding_quality_score: float
    last_updated: datetime

    model_config = ConfigDict(from_attributes=True)

    @computed_field
    @property
    def statistics(self) -> EmbeddingStatistics:
        """Calculates statistical summary properties dynamically from the coordinate vector."""
        arr = self.to_numpy(self.embedding_vector)
        length = len(arr)

        if length == 0:
            return EmbeddingStatistics(
                vector_norm=0.0,
                vector_mean=0.0,
                vector_std=0.0,
                non_zero_elements=0,
                sparsity=0.0,
            )

        norm = float(np.linalg.norm(arr))
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        non_zero = int(np.count_nonzero(arr))
        sparsity = float(1.0 - (non_zero / length))

        return EmbeddingStatistics(
            vector_norm=norm,
            vector_mean=mean,
            vector_std=std,
            non_zero_elements=non_zero,
            sparsity=sparsity,
        )

    @staticmethod
    def to_numpy(vector: List[float]) -> np.ndarray:
        """Converts a standard Python float list into a NumPy float32 array.

        Args:
            vector: The python float coordinate list.

        Returns:
            np.ndarray: Native NumPy array of the coordinates.
        """
        return np.array(vector, dtype=np.float32)

    @staticmethod
    def from_numpy(arr: np.ndarray) -> List[float]:
        """Converts a NumPy array back to a standard Python float list.

        Args:
            arr: The NumPy coordinate array.

        Returns:
            List[float]: The coordinate list.
        """
        return arr.tolist()

    @staticmethod
    def normalize(vector: List[float]) -> List[float]:
        """Applies L2 normalization on a coordinate vector.

        Formula:
            v_norm = v / ||v||

        Args:
            vector: Coordinate list.

        Returns:
            List[float]: Standardized L2-normalized vector list.
        """
        arr = EmbeddingResponse.to_numpy(vector)
        norm = np.linalg.norm(arr)

        if norm == 0.0 or not np.isfinite(norm):
            # Return zero vector if normalization is impossible
            return [0.0] * len(vector)

        return (arr / norm).tolist()

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Computes the cosine similarity between two coordinate lists.

        Formula:
            similarity = dot(a, b) / (||a|| * ||b||)

        Args:
            a: First vector list.
            b: Second vector list.

        Returns:
            float: Similarity metric clamped between 0.0 and 1.0.
        """
        if len(a) != len(b) or not a:
            return 0.0

        arr_a = EmbeddingResponse.to_numpy(a)
        arr_b = EmbeddingResponse.to_numpy(b)

        norm_a = np.linalg.norm(arr_a)
        norm_b = np.linalg.norm(arr_b)

        if norm_a == 0.0 or norm_b == 0.0 or not np.isfinite(norm_a) or not np.isfinite(norm_b):
            return 0.0

        dot_product = np.dot(arr_a, arr_b)
        similarity = dot_product / (norm_a * norm_b)

        # Cosine similarity technically boundaries [-1.0, 1.0], clamp 0.0 to 1.0 for driver matching
        return float(max(0.0, min(1.0, similarity)))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Pydantic response schema to a standard Python dictionary."""
        return self.model_dump()
