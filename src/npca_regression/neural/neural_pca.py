from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ._utils import as_2d_float_array, resolve_n_components
from .activations import ActivationStats, estimate_activation_stats, neural_activation
from .linalg import orthogonalize_rows, row_orthogonality_error
from .whitening import PCAWhitening

Array = np.ndarray


@dataclass
class NeuralPCAConfig:
    """Configuration for fixed-point neural NPCA.

    This implements the stabilized fixed-point update from the uploaded neural
    NPCA prototype, but as a reusable estimator rather than as a demo script.
    """

    n_components: Optional[int] = None
    whitening: str = "pca"
    epsilon: float = 1e-5
    block_size: int = 32
    step_size: float = 0.1
    max_iter: int = 1000
    tol: float = 1e-6
    random_state: Optional[int] = 0
    verbose: bool = False


class NeuralPCA:
    """Neural nonlinear PCA estimator for sample-major data.

    The model first whitens the data, then learns an orthonormal matrix ``W`` by
    repeatedly applying a nonlinear activation and symmetric orthogonalization.

    Input shape is ``(n_samples, n_features)``.
    """

    def __init__(self, config: Optional[NeuralPCAConfig] = None) -> None:
        self.config = config or NeuralPCAConfig()
        self.whitener_: Optional[PCAWhitening] = None
        self.W_: Optional[Array] = None
        self.separating_matrix_: Optional[Array] = None
        self.activation_stats_: Optional[ActivationStats] = None
        self.history_: list[Dict[str, float]] = []
        self.n_features_in_: Optional[int] = None
        self.n_components_: Optional[int] = None

    @property
    def components_(self) -> Array:
        """Rows of the learned separating matrix in the original input space."""
        if self.separating_matrix_ is None:
            raise AttributeError("Call fit before accessing components_.")
        return self.separating_matrix_

    def fit(self, X: Array) -> "NeuralPCA":
        X = as_2d_float_array(X)
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("NeuralPCA requires at least two samples.")

        n_components = resolve_n_components(self.config.n_components, n_features)
        block_size = int(self.config.block_size)
        if block_size < 1:
            raise ValueError("block_size must be at least 1.")
        if self.config.max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if self.config.step_size <= 0:
            raise ValueError("step_size must be positive.")

        self.n_features_in_ = int(n_features)
        self.n_components_ = int(n_components)
        self.whitener_ = PCAWhitening(
            n_components=n_components,
            method=self.config.whitening,
            epsilon=self.config.epsilon,
        )
        V = self.whitener_.fit_transform(X)  # (n_samples, n_components)

        rng = np.random.default_rng(self.config.random_state)
        W0 = rng.normal(size=(n_components, n_components))
        W = orthogonalize_rows(W0)

        self.history_ = []
        num_blocks = max(1, int(np.ceil(n_samples / block_size)))

        for iteration in range(1, int(self.config.max_iter) + 1):
            W_prev = W.copy()
            grad = np.zeros_like(W)

            for start in range(0, n_samples, block_size):
                stop = min(start + block_size, n_samples)
                V_b = V[start:stop]
                Y_b = V_b @ W.T
                Phi_b = neural_activation(Y_b)
                grad += (Phi_b.T @ V_b) / float(max(stop - start, 1))

            grad *= float(self.config.step_size) / float(num_blocks)
            W = orthogonalize_rows(W + grad)

            delta = float(np.max(np.abs(W - W_prev)))
            record = {
                "iteration": float(iteration),
                "delta": delta,
                "orthogonality_error": row_orthogonality_error(W),
            }
            self.history_.append(record)
            if self.config.verbose:
                print(record)
            if delta < float(self.config.tol):
                break

        self.W_ = W
        self.separating_matrix_ = W @ self.whitener_.whitening_matrix_
        Y = V @ W.T
        self.activation_stats_ = estimate_activation_stats(Y)
        return self

    def transform(self, X: Array) -> Array:
        if self.W_ is None or self.whitener_ is None:
            raise RuntimeError("Call fit before transform.")
        X = as_2d_float_array(X)
        V = self.whitener_.transform(X)
        return V @ self.W_.T

    def fit_transform(self, X: Array) -> Array:
        return self.fit(X).transform(X)

    def reconstruct_whitened(self, X: Array) -> Tuple[Array, Array]:
        """Return whitened data and its nonlinear reconstruction."""
        if self.W_ is None or self.whitener_ is None or self.activation_stats_ is None:
            raise RuntimeError("Call fit before reconstruct_whitened.")
        V = self.whitener_.transform(as_2d_float_array(X))
        Y = V @ self.W_.T
        Phi = neural_activation(
            Y,
            second_moment=self.activation_stats_.second_moment,
            kurtosis_sign=self.activation_stats_.kurtosis_sign,
        )
        V_hat = Phi @ self.W_
        return V, V_hat

    def reconstruct(self, X: Array) -> Array:
        if self.whitener_ is None:
            raise RuntimeError("Call fit before reconstruct.")
        _, V_hat = self.reconstruct_whitened(X)
        return self.whitener_.inverse_transform(V_hat)

    def reconstruction_error(self, X: Array) -> Array:
        """Squared whitened-space reconstruction error for each sample."""
        V, V_hat = self.reconstruct_whitened(X)
        return np.sum((V - V_hat) ** 2, axis=1)

    def score(self, X: Array) -> float:
        return -float(np.mean(self.reconstruction_error(X)))
