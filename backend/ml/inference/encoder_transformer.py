"""Encoder transformer extracted from the trained embedding autoencoder."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPRegressor


class EncoderTransformer(BaseEstimator, TransformerMixin):
    """Wraps the hidden layer of a trained autoencoder as a sklearn transformer."""

    def __init__(
        self,
        coef_: Optional[np.ndarray] = None,
        intercept_: Optional[np.ndarray] = None,
        activation: str = "relu",
    ) -> None:
        self.coef_ = coef_
        self.intercept_ = intercept_
        self.activation = activation

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "EncoderTransformer":
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError(
                "EncoderTransformer requires coef_ and intercept_ before use."
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("EncoderTransformer is not fitted: missing weights.")

        z = X @ self.coef_ + self.intercept_

        if self.activation == "relu":
            z = np.maximum(z, 0.0)
        elif self.activation == "tanh":
            z = np.tanh(z)
        elif self.activation == "logistic":
            z = 1.0 / (1.0 + np.exp(-z))

        return z.astype(np.float32)

    @property
    def embedding_dim_(self) -> int:
        if self.coef_ is None:
            return 0
        return int(self.coef_.shape[1])

    @classmethod
    def from_mlp(cls, mlp: MLPRegressor) -> "EncoderTransformer":
        if not mlp.coefs_ or not mlp.intercepts_:
            raise ValueError("MLPRegressor has no learned weights to extract.")

        return cls(
            coef_=mlp.coefs_[0].astype(np.float32),
            intercept_=mlp.intercepts_[0].astype(np.float32),
            activation=mlp.activation,
        )


# Legacy alias used in older serialized pipelines.
_EncoderTransformer = EncoderTransformer
