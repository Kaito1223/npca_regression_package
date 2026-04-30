from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._utils import as_2d_float_array, resolve_n_components

Array = np.ndarray


@dataclass
class PCAWhitening:
    """PCA/ZCA whitening for sample-major data.

    Input convention is ``(n_samples, n_features)``. Compatibility helper
    functions at the bottom keep the uploaded prototype's
    ``(n_features, n_samples)`` convention available.
    """

    n_components: Optional[int] = None
    method: str = "pca"
    epsilon: float = 1e-5

    mean_: Optional[Array] = None
    components_: Optional[Array] = None
    eigenvalues_: Optional[Array] = None
    whitening_matrix_: Optional[Array] = None
    dewhitening_matrix_: Optional[Array] = None
    n_features_in_: Optional[int] = None
    n_components_: Optional[int] = None

    def fit(self, X: Array) -> "PCAWhitening":
        X = as_2d_float_array(X)
        n_samples, n_features = X.shape
        n_components = resolve_n_components(self.n_components, n_features)

        method = str(self.method).lower()
        if method not in {"pca", "zca"}:
            raise ValueError("method must be 'pca' or 'zca'.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.n_features_in_ = int(n_features)
        self.n_components_ = int(n_components)
        self.mean_ = X.mean(axis=0, keepdims=True)
        Xc = X - self.mean_

        cov = (Xc.T @ Xc) / float(max(n_samples, 1))
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        eigvals_n = eigvals[:n_components]
        eigvecs_n = eigvecs[:, :n_components]
        inv_sqrt = 1.0 / np.sqrt(eigvals_n + float(self.epsilon))
        sqrt_vals = np.sqrt(eigvals_n + float(self.epsilon))

        if method == "pca" or n_components < n_features:
            whitening_matrix = np.diag(inv_sqrt) @ eigvecs_n.T
            dewhitening_matrix = eigvecs_n @ np.diag(sqrt_vals)
        else:
            whitening_matrix = eigvecs_n @ np.diag(inv_sqrt) @ eigvecs_n.T
            dewhitening_matrix = eigvecs_n @ np.diag(sqrt_vals) @ eigvecs_n.T

        self.components_ = eigvecs_n.T.copy()
        self.eigenvalues_ = eigvals_n.copy()
        self.whitening_matrix_ = whitening_matrix.copy()
        self.dewhitening_matrix_ = dewhitening_matrix.copy()
        return self

    def transform(self, X: Array) -> Array:
        if self.mean_ is None or self.whitening_matrix_ is None:
            raise RuntimeError("Call fit before transform.")
        X = as_2d_float_array(X)
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features.")
        return (X - self.mean_) @ self.whitening_matrix_.T

    def fit_transform(self, X: Array) -> Array:
        return self.fit(X).transform(X)

    def inverse_transform(self, V: Array) -> Array:
        if self.mean_ is None or self.dewhitening_matrix_ is None:
            raise RuntimeError("Call fit before inverse_transform.")
        V = as_2d_float_array(V, name="V")
        if self.n_components_ is not None and V.shape[1] != self.n_components_:
            raise ValueError(f"V must have {self.n_components_} components.")
        return V @ self.dewhitening_matrix_.T + self.mean_


# Compatibility helpers for the uploaded prototype's feature-major convention.
def pca_whitening(data: Array, n_components: Optional[int] = None, epsilon: float = 1e-5):
    """PCA whitening for data with shape ``(n_features, n_samples)``."""
    whitener = PCAWhitening(n_components=n_components, method="pca", epsilon=epsilon)
    X_white = whitener.fit_transform(np.asarray(data, dtype=float).T).T
    return X_white, whitener.whitening_matrix_


def zca_whitening(data: Array, n_components: Optional[int] = None, epsilon: float = 1e-5):
    """ZCA whitening for data with shape ``(n_features, n_samples)``."""
    whitener = PCAWhitening(n_components=n_components, method="zca", epsilon=epsilon)
    X_white = whitener.fit_transform(np.asarray(data, dtype=float).T).T
    return X_white, whitener.whitening_matrix_


def whitening(data: Array, name: str = "pca"):
    name = str(name).lower()
    if name == "pca":
        return pca_whitening(data)
    if name == "zca":
        return zca_whitening(data)
    raise ValueError(f"Unknown whitening method: {name}")
