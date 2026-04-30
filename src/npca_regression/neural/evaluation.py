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
    grid_size: int = 61,
    refine: bool = True,
) -> tuple[pd.DataFrame, list[Dict[str, Any]]]:
    """Fit neural NPCA regression and return KPCA-style benchmark outputs.

    Returns
    -------
    df_results:
        One-row DataFrame with columns compatible with the feature-space NPCA
        output: ``kernel``, ``params``, ``m``, ``RMSE_yhat_vs_y``, and
        ``root_mean_feature``.
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
            grid_size=grid_size,
            refine=refine,
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
        "block_size": config.neural_pca.block_size,
        "step_size": config.neural_pca.step_size,
        "max_iter": config.neural_pca.max_iter,
        "tol": config.neural_pca.tol,
        "random_state": config.neural_pca.random_state,
        "grid_size": config.grid_size,
        "refine": config.refine,
        "y_bounds": estimator.y_bounds_,
    }
    param_label = (
        f"n_components={n_comp}, whitening={config.neural_pca.whitening}, "
        f"block_size={config.neural_pca.block_size}, "
        f"step_size={config.neural_pca.step_size:g}, "
        f"grid_size={config.grid_size}, refine={config.refine}"
    )

    df_results = pd.DataFrame([
        {
            "kernel": "neural-npca",
            "params": param_label,
            "m": n_comp,
            "RMSE_yhat_vs_y": rmse_y,
            "root_mean_feature": root_mean_feature,
        }
    ])

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
