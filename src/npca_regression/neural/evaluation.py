from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ._utils import as_1d_float_array, as_2d_float_array, rmse
from .neural_pca import NeuralPCAConfig
from .regressor import NeuralNPCARegressionConfig, NeuralNPCARegressor


Array = np.ndarray


def evaluate_neural_npca_regression(
    Z_train: Array,
    X_test: Array,
    y_test: Optional[Array] = None,
    *,
    config: Optional[NeuralNPCARegressionConfig] = None,
    n_components: Optional[int] = None,
    whitening: str = "pca",
    epsilon: float = 1e-5,
    block_size: int = 32,
    step_size: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-6,
    random_state: Optional[int] = 0,
    y_bounds: Optional[Tuple[float, float]] = None,
    y_margin_fraction: float = 0.15,
    prediction_optimizer: str = "grid",
    grid_size: int = 61,
    refine: bool = True,
    optimizer_xatol: float = 1e-6,
    torch_learning_rate: float = 0.05,
    torch_steps: int = 200,
    torch_tolerance: float = 1e-7,
    torch_restarts: int = 1,
    torch_init_noise: float = 0.1,
    torch_device: str = "cpu",
) -> tuple[pd.DataFrame, list[Dict[str, Any]]]:
    """Fit neural NPCA regression and return KPCA-style benchmark outputs.

    Returns
    -------
    df_results:
        One-row DataFrame with columns compatible with the feature-space NPCA
        output: kernel, params, m, RMSE_yhat_vs_y, root_mean_feature.

    results:
        List containing raw predictions and the fitted estimator.
    """
    Z_train = as_2d_float_array(Z_train, name="Z_train")
    X_test = as_2d_float_array(X_test, name="X_test")

    if Z_train.shape[1] < 2:
        raise ValueError("Z_train must contain at least one feature and one target column.")

    if config is None:
        config = NeuralNPCARegressionConfig(
            neural_pca=NeuralPCAConfig(
                n_components=n_components,
                whitening=whitening,
                epsilon=epsilon,
                block_size=block_size,
                step_size=step_size,
                max_iter=max_iter,
                tol=tol,
                random_state=random_state,
            ),
            y_bounds=y_bounds,
            y_margin_fraction=y_margin_fraction,
            prediction_optimizer=prediction_optimizer,
            grid_size=grid_size,
            refine=refine,
            optimizer_xatol=optimizer_xatol,
            torch_learning_rate=torch_learning_rate,
            torch_steps=torch_steps,
            torch_tolerance=torch_tolerance,
            torch_restarts=torch_restarts,
            torch_init_noise=torch_init_noise,
            torch_device=torch_device,
        )

    X_train = Z_train[:, :-1]
    y_train = Z_train[:, -1]

    estimator = NeuralNPCARegressor(config=config).fit(X_train, y_train)
    y_pred, details = estimator.predict(X_test, return_details=True)

    rmse_y = None

    if y_test is not None:
        rmse_y = rmse(as_1d_float_array(y_test, name="y_test"), y_pred)

    residuals = np.asarray(details["objective"], dtype=float)
    root_mean_feature = float(np.sqrt(np.maximum(np.mean(residuals), 0.0)))

    n_comp = estimator.model_.n_components_ if estimator.model_ is not None else None

    param_raw: Dict[str, Any] = {
        "n_components": n_comp,
        "whitening": config.neural_pca.whitening,
        "epsilon": config.neural_pca.epsilon,
        "block_size": config.neural_pca.block_size,
        "step_size": config.neural_pca.step_size,
        "max_iter": config.neural_pca.max_iter,
        "tol": config.neural_pca.tol,
        "random_state": config.neural_pca.random_state,
        "prediction_optimizer": config.prediction_optimizer,
        "grid_size": config.grid_size,
        "refine": config.refine,
        "optimizer_xatol": config.optimizer_xatol,
        "torch_learning_rate": config.torch_learning_rate,
        "torch_steps": config.torch_steps,
        "torch_tolerance": config.torch_tolerance,
        "torch_restarts": config.torch_restarts,
        "torch_init_noise": config.torch_init_noise,
        "torch_device": config.torch_device,
        "y_bounds": estimator.y_bounds_,
    }

    optimizer_name = str(config.prediction_optimizer).lower()

    if optimizer_name in {"torch", "grid_then_torch"}:
        optimizer_label = (
            f"{optimizer_name}, torch_lr={config.torch_learning_rate:g}, "
            f"torch_steps={config.torch_steps}, restarts={config.torch_restarts}"
        )
    else:
        optimizer_label = (
            f"{optimizer_name}, grid_size={config.grid_size}, refine={config.refine}"
        )

    param_label = (
        f"n_components={n_comp}, whitening={config.neural_pca.whitening}, "
        f"block_size={config.neural_pca.block_size}, "
        f"step_size={config.neural_pca.step_size:g}, "
        f"{optimizer_label}"
    )

    df_results = pd.DataFrame(
        [
            {
                "kernel": "neural-npca",
                "params": param_label,
                "m": n_comp,
                "RMSE_yhat_vs_y": rmse_y,
                "root_mean_feature": root_mean_feature,
            }
        ]
    )

    results = [
        {
            "kernel": "neural-npca",
            "params": param_raw,
            "params_label": param_label,
            "pred": y_pred,
            "details": details,
            "model": estimator,
        }
    ]

    return df_results, results