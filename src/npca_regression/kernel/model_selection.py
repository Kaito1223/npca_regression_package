from __future__ import annotations

import numpy as np
from typing import Optional

from .regression import fit_models, predict

Array = np.ndarray

def evaluate_kernels_with_torch_argmin(
    Z_train: Array,
    X_test: Array,
    y_test: Optional[Array] = None,
    method: str = "kernel",
    evr_target: Optional[np.float64] = None,
    m: Optional[int] = None,
    scale_data: bool = True,
    k: int = 0,
    lambda_knn: np.float64 = 0.0,
    batch_size: Optional[int] = None,
    lr: np.float64 = 0.05,
    steps: int = 300,
    torch_device: Optional[str] = None,
    restarts: int = 1,
    init_perturb: np.float64 = 0.5,
    prediction_optimizer: str = "torch",
    grid_size: int = 101,
    grid_refine: bool = True,
    y_bounds: Optional[tuple[float, float]] = None,
    y_margin_fraction: np.float64 = 0.15,
    **kwargs,
):
    models = fit_models(
        Z_train,
        method=method,
        evr_target=evr_target,
        m=m,
        scale_data=scale_data,
        **kwargs,
    )
    return predict(
        models,
        X_test,
        y_test=y_test,
        k=k,
        lambda_knn=lambda_knn,
        batch_size=batch_size,
        lr=lr,
        steps=steps,
        torch_device=torch_device,
        restarts=restarts,
        init_perturb=init_perturb,
        prediction_optimizer=prediction_optimizer,
        grid_size=grid_size,
        grid_refine=grid_refine,
        y_bounds=y_bounds,
        y_margin_fraction=y_margin_fraction,
    )

__all__ = ["evaluate_kernels_with_torch_argmin"]
