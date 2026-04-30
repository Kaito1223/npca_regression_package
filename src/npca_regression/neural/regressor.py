from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ._utils import as_1d_float_array, as_2d_float_array
from .neural_pca import NeuralPCA, NeuralPCAConfig

try:
    from scipy.optimize import minimize_scalar
except Exception:  # pragma: no cover
    minimize_scalar = None

Array = np.ndarray


@dataclass
class NeuralNPCARegressionConfig:
    """Configuration for neural-NPCA regression.

    The model fits neural NPCA on joint points ``z=[x,y]``. Prediction fixes
    ``x`` and searches for the scalar ``y`` that minimizes the neural NPCA
    reconstruction residual in whitened feature space.
    """

    neural_pca: NeuralPCAConfig = field(default_factory=NeuralPCAConfig)
    y_bounds: Optional[Tuple[float, float]] = None
    y_margin_fraction: float = 0.15
    grid_size: int = 61
    refine: bool = True
    optimizer_xatol: float = 1e-6


class NeuralNPCARegressor:
    """NPCA regression using neural nonlinear PCA instead of kernel PCA.

    This is the neural analogue of the package's feature-space NPCA regression:
    it learns a principal structure in the joint space ``Z=[X,y]`` and predicts
    a missing target by minimizing distance to that structure.
    """

    def __init__(self, config: Optional[NeuralNPCARegressionConfig] = None) -> None:
        self.config = config or NeuralNPCARegressionConfig()
        self.model_: Optional[NeuralPCA] = None
        self.y_min_: Optional[float] = None
        self.y_max_: Optional[float] = None
        self.y_bounds_: Optional[Tuple[float, float]] = None
        self.n_features_in_: Optional[int] = None
        self.history_: list[Dict[str, float]] = []

    def fit(self, X: Array, y: Array) -> "NeuralNPCARegressor":
        X = as_2d_float_array(X)
        y = as_1d_float_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        Z = np.column_stack([X, y])
        return self.fit_joint(Z)

    def fit_joint(self, Z: Array) -> "NeuralNPCARegressor":
        Z = as_2d_float_array(Z, name="Z")
        if Z.shape[1] < 2:
            raise ValueError("Z must contain at least one feature column and one target column.")

        self.n_features_in_ = int(Z.shape[1] - 1)
        y_train = Z[:, -1]
        self.y_min_ = float(np.min(y_train))
        self.y_max_ = float(np.max(y_train))
        self.y_bounds_ = self._resolve_y_bounds()

        self.model_ = NeuralPCA(self.config.neural_pca).fit(Z)
        self.history_ = [dict(item) for item in self.model_.history_]
        return self

    def predict(self, X: Array, return_details: bool = False):
        if self.model_ is None:
            raise RuntimeError("Call fit before predict.")
        X = as_2d_float_array(X)
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} columns.")

        preds = np.empty(X.shape[0], dtype=float)
        losses = np.empty(X.shape[0], dtype=float)
        for i, x in enumerate(X):
            y_hat, loss = self._predict_one(x)
            preds[i] = y_hat
            losses[i] = loss

        if return_details:
            return preds, {"objective": losses, "y_bounds": self.y_bounds_}
        return preds

    def reconstruction_error(self, X: Array, y: Array) -> Array:
        """Residual for completed points ``[X, y]``."""
        if self.model_ is None:
            raise RuntimeError("Call fit before reconstruction_error.")
        X = as_2d_float_array(X)
        y = as_1d_float_array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        Z = np.column_stack([X, y])
        return self.model_.reconstruction_error(Z)

    def score_candidate(self, x: Array, y_value: float) -> float:
        if self.model_ is None:
            raise RuntimeError("Call fit before score_candidate.")
        x = np.asarray(x, dtype=float).reshape(1, -1)
        z = np.column_stack([x, np.array([float(y_value)])])
        return float(self.model_.reconstruction_error(z)[0])

    def score(self, X: Array, y: Array) -> float:
        y = as_1d_float_array(y)
        y_pred = self.predict(X)
        return -float(np.mean((y_pred - y) ** 2))

    def _predict_one(self, x: Array) -> Tuple[float, float]:
        lo, hi = self._require_bounds()
        grid_size = max(3, int(self.config.grid_size))
        grid = np.linspace(lo, hi, grid_size)
        values = np.array([self.score_candidate(x, yy) for yy in grid], dtype=float)
        best_idx = int(np.argmin(values))
        best_y = float(grid[best_idx])
        best_value = float(values[best_idx])

        if self.config.refine and minimize_scalar is not None:
            left = float(grid[max(0, best_idx - 1)])
            right = float(grid[min(grid_size - 1, best_idx + 1)])
            if right > left:
                result = minimize_scalar(
                    lambda yy: self.score_candidate(x, float(yy)),
                    bounds=(left, right),
                    method="bounded",
                    options={"xatol": float(self.config.optimizer_xatol)},
                )
                if result.success and float(result.fun) <= best_value:
                    best_y = float(result.x)
                    best_value = float(result.fun)

        return best_y, best_value

    def _resolve_y_bounds(self) -> Tuple[float, float]:
        if self.config.y_bounds is not None:
            lo, hi = map(float, self.config.y_bounds)
            if not lo < hi:
                raise ValueError("y_bounds must satisfy lower < upper.")
            return lo, hi

        if self.y_min_ is None or self.y_max_ is None:
            raise RuntimeError("Training target range is unavailable.")
        width = self.y_max_ - self.y_min_
        if width <= 0:
            width = max(abs(self.y_min_), 1.0)
        margin = float(self.config.y_margin_fraction) * width
        return float(self.y_min_ - margin), float(self.y_max_ + margin)

    def _require_bounds(self) -> Tuple[float, float]:
        if self.y_bounds_ is None:
            self.y_bounds_ = self._resolve_y_bounds()
        return self.y_bounds_
