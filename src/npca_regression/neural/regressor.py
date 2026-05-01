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

    The model fits neural NPCA on joint points z=[x,y]. Prediction fixes x and
    optimizes the scalar y that minimizes the neural NPCA reconstruction residual.

    prediction_optimizer:
        "grid":
            Coarse grid search over y, with optional scipy scalar refinement.

        "torch":
            Direct PyTorch gradient descent over y.

        "grid_then_torch":
            Grid search first, then PyTorch gradient descent initialized from
            the best grid value.
    """

    neural_pca: NeuralPCAConfig = field(default_factory=NeuralPCAConfig)

    y_bounds: Optional[Tuple[float, float]] = None
    y_margin_fraction: float = 0.15

    prediction_optimizer: str = "grid"

    # Grid/scipy optimizer settings.
    grid_size: int = 61
    refine: bool = True
    optimizer_xatol: float = 1e-6

    # PyTorch optimizer settings.
    torch_learning_rate: float = 0.05
    torch_steps: int = 200
    torch_tolerance: float = 1e-7
    torch_restarts: int = 1
    torch_init_noise: float = 0.1
    torch_device: str = "cpu"


class NeuralNPCARegressor:
    """NPCA regression using neural nonlinear PCA.

    This is the neural analogue of feature-space NPCA regression. It learns a
    principal structure in the joint space Z=[X,y] and predicts a missing target
    by minimizing distance to that structure.
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
        X = as_2d_float_array(X, name="X")
        y = as_1d_float_array(y, name="y")

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

        X = as_2d_float_array(X, name="X")

        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} columns.")

        optimizer_name = str(self.config.prediction_optimizer).lower()

        if optimizer_name == "grid":
            preds, losses = self._predict_batch_grid(X)

        elif optimizer_name == "torch":
            preds, losses = self._predict_batch_torch(X)

        elif optimizer_name == "grid_then_torch":
            grid_preds, _ = self._predict_batch_grid(X)
            preds, losses = self._predict_batch_torch(X, y_init=grid_preds)

        else:
            raise ValueError(
                "prediction_optimizer must be one of: "
                "'grid', 'torch', 'grid_then_torch'."
            )

        if return_details:
            return preds, {
                "objective": losses,
                "y_bounds": self.y_bounds_,
                "prediction_optimizer": optimizer_name,
            }

        return preds

    def reconstruction_error(self, X: Array, y: Array) -> Array:
        """Residual for completed points [X, y]."""
        if self.model_ is None:
            raise RuntimeError("Call fit before reconstruction_error.")

        X = as_2d_float_array(X, name="X")
        y = as_1d_float_array(y, name="y")

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
        y = as_1d_float_array(y, name="y")
        y_pred = self.predict(X)

        return -float(np.mean((y_pred - y) ** 2))

    def _predict_batch_grid(self, X: Array) -> Tuple[Array, Array]:
        preds = np.empty(X.shape[0], dtype=float)
        losses = np.empty(X.shape[0], dtype=float)

        for i, x in enumerate(X):
            y_hat, loss = self._predict_one_grid(x)
            preds[i] = y_hat
            losses[i] = loss

        return preds, losses

    def _predict_one_grid(self, x: Array) -> Tuple[float, float]:
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

    def _predict_batch_torch(
        self,
        X: Array,
        y_init: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Optimize y with PyTorch autograd.

        This keeps x fixed and treats y as the only trainable variable.
        """
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "PyTorch is required for prediction_optimizer='torch' or "
                "'grid_then_torch'. Install torch or use prediction_optimizer='grid'."
            ) from exc

        lo, hi = self._require_bounds()

        X = as_2d_float_array(X, name="X")
        n_samples = X.shape[0]

        device = torch.device(str(self.config.torch_device))
        dtype = torch.float64

        if y_init is None:
            y0 = np.full(n_samples, 0.5 * (lo + hi), dtype=float)
        else:
            y0 = np.asarray(y_init, dtype=float).reshape(-1)
            if y0.shape[0] != n_samples:
                raise ValueError("y_init must have one value per test sample.")
            y0 = np.clip(y0, lo, hi)

        best_y = np.zeros(n_samples, dtype=float)
        best_loss = np.full(n_samples, np.inf, dtype=float)

        restarts = max(1, int(self.config.torch_restarts))

        for restart in range(restarts):
            y_start = y0.copy()

            if restart > 0:
                rng = np.random.default_rng(restart)
                y_start = y_start + float(self.config.torch_init_noise) * rng.normal(size=n_samples)
                y_start = np.clip(y_start, lo, hi)

            Y = torch.tensor(
                y_start.reshape(-1, 1),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )

            optimizer = torch.optim.Adam(
                [Y],
                lr=float(self.config.torch_learning_rate),
            )

            previous_loss_value: Optional[float] = None

            for _ in range(int(self.config.torch_steps)):
                optimizer.zero_grad(set_to_none=True)

                losses = self._torch_reconstruction_error(X, Y)
                loss = losses.sum()

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    Y.clamp_(lo, hi)

                loss_value = float(loss.detach().cpu().item())

                if previous_loss_value is not None:
                    improvement = abs(previous_loss_value - loss_value)
                    if improvement <= float(self.config.torch_tolerance):
                        break

                previous_loss_value = loss_value

            with torch.no_grad():
                final_losses = self._torch_reconstruction_error(X, Y)
                y_np = Y.detach().cpu().numpy().reshape(-1)
                loss_np = final_losses.detach().cpu().numpy().reshape(-1)

            improved = loss_np < best_loss
            best_loss[improved] = loss_np[improved]
            best_y[improved] = y_np[improved]

        return best_y, best_loss

    def _torch_reconstruction_error(self, X_np: Array, Y_torch):
        """Torch version of the neural reconstruction residual.

        Computes:

            || v - W^T g(Wv) ||^2

        for completed points z=[x,y].
        """
        import torch

        if self.model_ is None:
            raise RuntimeError("Call fit before prediction.")
        if self.model_.W_ is None:
            raise RuntimeError("NeuralPCA model is not fitted.")
        if self.model_.whitener_ is None:
            raise RuntimeError("Whitening model is not fitted.")
        if self.model_.activation_stats_ is None:
            raise RuntimeError("Activation statistics are unavailable.")
        if self.model_.whitener_.mean_ is None:
            raise RuntimeError("Whitening mean is unavailable.")
        if self.model_.whitener_.whitening_matrix_ is None:
            raise RuntimeError("Whitening matrix is unavailable.")

        device = Y_torch.device
        dtype = torch.float64

        X = torch.as_tensor(X_np, dtype=dtype, device=device)

        mean = torch.as_tensor(
            self.model_.whitener_.mean_,
            dtype=dtype,
            device=device,
        )

        whitening_matrix = torch.as_tensor(
            self.model_.whitener_.whitening_matrix_,
            dtype=dtype,
            device=device,
        )

        W = torch.as_tensor(
            self.model_.W_,
            dtype=dtype,
            device=device,
        )

        second_moment = torch.as_tensor(
            self.model_.activation_stats_.second_moment,
            dtype=dtype,
            device=device,
        )

        kurtosis_sign = torch.as_tensor(
            self.model_.activation_stats_.kurtosis_sign,
            dtype=dtype,
            device=device,
        )

        Z = torch.cat([X, Y_torch], dim=1)

        V = (Z - mean) @ whitening_matrix.T
        S = V @ W.T

        Phi = kurtosis_sign * (S**3 - 3.0 * S * second_moment)

        V_hat = Phi @ W

        residual = torch.sum((V - V_hat) ** 2, dim=1)

        return residual

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