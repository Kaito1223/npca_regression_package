from __future__ import annotations

import numpy as np

Array = np.ndarray


def _as_matching_1d_arrays(y_true: Array, y_pred: Array) -> tuple[Array, Array]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    return y_true, y_pred


def mse(y_true: Array, y_pred: Array) -> np.float64:
    y_true, y_pred = _as_matching_1d_arrays(y_true, y_pred)
    return np.float64(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: Array, y_pred: Array) -> np.float64:
    return np.float64(np.sqrt(mse(y_true, y_pred)))


def sad(y_true: Array, y_pred: Array) -> np.float64:
    """
    Sum of absolute deviations:

        SAD = sum_i |y_i - yhat_i|
    """
    y_true, y_pred = _as_matching_1d_arrays(y_true, y_pred)
    return np.float64(np.sum(np.abs(y_true - y_pred)))


def mae(y_true: Array, y_pred: Array) -> np.float64:
    """
    Mean absolute error. This is optional, but useful because SAD depends on test-set size.
    """
    y_true, y_pred = _as_matching_1d_arrays(y_true, y_pred)
    return np.float64(np.mean(np.abs(y_true - y_pred)))


__all__ = ["mse", "rmse", "sad", "mae"]