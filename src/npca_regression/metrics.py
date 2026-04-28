from __future__ import annotations

import numpy as np

Array = np.ndarray

def mse(y_true: Array, y_pred: Array) -> np.float64:
    return np.float64(np.mean((y_true - y_pred) ** 2))

def rmse(y_true: Array, y_pred: Array) -> np.float64:
    return np.sqrt(mse(y_true, y_pred))

__all__ = ["mse", "rmse"]
