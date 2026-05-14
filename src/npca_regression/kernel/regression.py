from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Union
from sklearn.preprocessing import StandardScaler

from . import pca as poly_pca
from .pca import KPCAResult, KernelConfig, MomentPolynomialPCA
from .projectors import (
    TorchProjector_Kernel,
    TorchProjector_Moment,
    kpca_train_rmse_feature,
)
from ..metrics import rmse, sad, mae
Array = np.ndarray

def fit_models(Z_train: Array,
               method: str = 'kernel',
               evr_target: Optional[np.float64] = None,
               m: Optional[int] = None,
               scale_data: bool = True,
               **kwargs) -> List[Union[KPCAResult, MomentPolynomialPCA]]:
    """
    Master function to fit models
    """
    if scale_data:
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_tr = Z_train[:, :-1]
        y_tr = Z_train[:, -1].reshape(-1, 1)
        
        X_tr_sc = x_scaler.fit_transform(X_tr)
        y_tr_sc = y_scaler.fit_transform(y_tr).ravel()
        Z_train_fit = np.column_stack([X_tr_sc, y_tr_sc])
    else:
        x_scaler = None
        y_scaler = None
        Z_train_fit = Z_train

    if method == 'kernel':
        print("--- Fitting Kernel PCA model ---")
        kernel_choice = kwargs.get('kernel_choice', 'all')
        poly_degrees = kwargs.get('poly_degrees', 'all')
        rbf_sigmas = kwargs.get('rbf_sigmas', 'all')
        constant = kwargs.get('constant', 0.0)
        
        if poly_degrees == "all": poly_degrees = [1, 2, 3, 4]
        if rbf_sigmas == "all": rbf_sigmas = [0.25, 0.5, 1.0, 2.0]
        configs: List[KernelConfig] = []
        if kernel_choice in ("poly", "all"):
            for d in poly_degrees:
                configs.append(KernelConfig(kind="poly", params={"degree": int(d), "c0": constant}))
        if kernel_choice in ("rbf", "all"):
            for s in rbf_sigmas:
                configs.append(KernelConfig(kind="rbf", params={"sigma": np.float64(s)}))

        models = []
        print(f"Fitting {len(configs)} models...")
        for cfg in configs:
            print(f"  Fitting {cfg.kind} with params: {cfg.params}")
            model = poly_pca.kpca_fit(Z_train_fit, cfg, m=m, evr_target=evr_target)
            
            model.x_scaler = x_scaler
            model.y_scaler = y_scaler

            rmse_train_phi = kpca_train_rmse_feature(model)
            model.rmse_feature_train = rmse_train_phi
            print(f"    train RMSE_feature (phi-space): {rmse_train_phi:.6g}")

            models.append(model)
        print("Model fitting complete.")
        return models
    
    elif method == 'moment':
        print("--- Fitting Moment PCA Pipeline ---")
        degrees = kwargs.get('degrees', [1, 2, 3])
        consts = kwargs.get('consts', [0.0, 1.0])

        models = []
        print(f"Fitting {len(degrees) * len(consts)} moment-based models...")
        for d in degrees:
            for c in consts:
                print(f"  Fitting degree={d}, const={c}")
                model = poly_pca.MomentPolynomialPCA(degree=d, const=c, n_components=m, evr_target=evr_target)
                model.fit(Z_train_fit)
                
                model.x_scaler = x_scaler
                model.y_scaler = y_scaler
                
                models.append(model)
        print("Model fitting complete.")
        return models
    else:
        raise ValueError(f"Unknown method: {method}")

def predict(
    models,
    X_test,
    y_test=None,
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
    y_margin_fraction: np.float64 = 0.15,) -> pd.DataFrame:

    rows = []
    results = []

    for model in models:
        y_pred: Array
        projector: Union[TorchProjector_Kernel, TorchProjector_Moment]
        if isinstance(model, KPCAResult):
            projector = TorchProjector_Kernel(model, k=k, lambda_knn=lambda_knn, device=torch_device)
            kernel_name = model.kernel_cfg.kind
            param_raw = dict(model.kernel_cfg.params)

            if kernel_name == "rbf" and "sigma" in param_raw:
                param_label = f"sigma={float(param_raw['sigma']):.1f}"
            elif kernel_name == "poly":
                degree = param_raw.get("degree", "?")
                c0 = float(param_raw.get("c0", 0.0))
                param_label = f"degree={degree}, c0={c0:g}"
            else:
                param_label = str(param_raw)
            print(f"Predicting with: {kernel_name}, params: {param_label}, "f"k={k}, lambda={lambda_knn}, device={projector.device}, "f"lr={lr}, steps={steps}, restarts={restarts}")

        elif isinstance(model, MomentPolynomialPCA):
            projector = TorchProjector_Moment(model, k=k, lambda_knn=lambda_knn, device=torch_device)
            kernel_name = "poly-moment"
            param_raw = {"degree": model.degree, "const": model.const}
            param_label = f"degree={model.degree}, const={model.const:g}"
            print(f"Predicting with: moment, params: {param_label}, k={k}, lambda={lambda_knn}, device={projector.device}, "f"lr={lr}, steps={steps}, restarts={restarts}")
        else:
            raise ValueError(f"Unknown model type for prediction. Model types are: {type(model)}")
        
        if batch_size is None or batch_size >= len(X_test):
            y_pred = projector.predict_y_batch(X_test, lr=lr, steps=steps, restarts=restarts, init_perturb=init_perturb, prediction_optimizer=prediction_optimizer, grid_size=grid_size, grid_refine=grid_refine, y_bounds=y_bounds, y_margin_fraction=y_margin_fraction)
        else:
            y_pred_parts = []
            num_samples = len(X_test)
            for i in range(0, num_samples, batch_size):
                X_batch = X_test[i : i + batch_size]
                y_pred_batch = projector.predict_y_batch(X_batch, lr=lr, steps=steps, restarts=restarts, init_perturb=init_perturb, prediction_optimizer=prediction_optimizer, grid_size=grid_size, grid_refine=grid_refine, y_bounds=y_bounds, y_margin_fraction=y_margin_fraction)
                y_pred_parts.append(y_pred_batch)
            
            y_pred = np.concatenate(y_pred_parts)
            
        if y_test is not None:
            rmse_y = rmse(y_test, y_pred)
            sad_y = sad(y_test, y_pred)
            mae_y = mae(y_test, y_pred)
        else:
            rmse_y = None
            sad_y = None
            mae_y = None

        with torch.no_grad():
            X_eval = X_test
            y_eval = y_pred
            
            if getattr(model, 'x_scaler', None) is not None:
                X_eval = model.x_scaler.transform(X_eval)
            if getattr(model, 'y_scaler', None) is not None:
                y_eval = model.y_scaler.transform(y_eval.reshape(-1, 1)).ravel()
                
            x_tensor = torch.tensor(X_eval, dtype=torch.float64, device=projector.device)
            y_pred_tensor = torch.tensor(y_eval, dtype=torch.float64, device=projector.device).view(-1, 1)
            
            residuals = projector.E_phi_batch(x_tensor, y_pred_tensor)
            mean_residual = np.float64(residuals.mean().item())
            root_mean_residual = np.sqrt(np.maximum(mean_residual, 0.0))
            
        rows.append({
            "kernel": kernel_name,
            "params": param_label,
            "m": model.m,
            "RMSE_yhat_vs_y": rmse_y,
            "SAD_yhat_vs_y": sad_y,
            "MAE_yhat_vs_y": mae_y,
            "root_mean_feature": root_mean_residual,
            "torch_device": str(projector.device),
            "lr": float(lr),
            "steps": int(steps),
            "restarts": int(restarts),
            "init_perturb": float(init_perturb),
            "prediction_optimizer": str(prediction_optimizer),
            "grid_size": int(grid_size),
            "grid_refine": bool(grid_refine),
            "y_bounds": y_bounds,
            "y_margin_fraction": float(y_margin_fraction),
        })
        results.append({
            "kernel": kernel_name,
            'params': param_raw,
            "params_label": param_label,
            "pred": y_pred,
            "torch": {
                "device": str(projector.device),
                "lr": float(lr),
                "steps": int(steps),
                "restarts": int(restarts),
                "init_perturb": float(init_perturb),
                "prediction_optimizer": str(prediction_optimizer),
                "grid_size": int(grid_size),
                "grid_refine": bool(grid_refine),
                "y_bounds": y_bounds,
                "y_margin_fraction": float(y_margin_fraction),
            },
        })

    return pd.DataFrame(rows), results  
