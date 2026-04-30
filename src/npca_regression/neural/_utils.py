from __future__ import annotations

from typing import Optional

import numpy as np

Array = np.ndarray


def as_2d_float_array(X: Array, *, name: str = "X") -> Array:
    """Convert input to a finite 2D float array."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape (n_samples, n_features).")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains non-finite values.")
    return X


def as_1d_float_array(y: Array, *, name: str = "y") -> Array:
    """Convert input to a finite 1D float array."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{name} contains non-finite values.")
    return y


def rmse(y_true: Array, y_pred: Array) -> float:
    y_true = as_1d_float_array(y_true, name="y_true")
    y_pred = as_1d_float_array(y_pred, name="y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def resolve_n_components(n_components: Optional[int], n_features: int) -> int:
    if n_components is None:
        return int(n_features)
    n = int(n_components)
    if not 1 <= n <= int(n_features):
        raise ValueError("n_components must be between 1 and n_features.")
    return n
