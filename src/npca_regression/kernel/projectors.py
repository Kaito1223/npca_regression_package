from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Tuple

from .pca import KPCAResult, KernelConfig, MomentPolynomialPCA

Array = np.ndarray

def _validate_prediction_optimizer(name: str) -> str:
    name = str(name).lower()
    valid = {"torch", "grid", "grid_then_torch"}
    if name not in valid:
        raise ValueError(
            "prediction_optimizer must be one of: "
            "'torch', 'grid', 'grid_then_torch'."
        )
    return name


def _make_y_grid(
    y_train: torch.Tensor,
    grid_size: int,
    margin_fraction: float,
    y_bounds: Optional[Tuple[float, float]],
    device: torch.device,
) -> torch.Tensor:
    """Make candidate y-grid in the internal/scaled y-space."""
    grid_size = max(3, int(grid_size))

    if y_bounds is not None:
        lo, hi = map(float, y_bounds)
    else:
        lo = float(torch.min(y_train).detach().cpu())
        hi = float(torch.max(y_train).detach().cpu())

        width = hi - lo
        if width <= 0:
            width = max(abs(lo), 1.0)

        margin = float(margin_fraction) * width
        lo = lo - margin
        hi = hi + margin

    if not lo < hi:
        raise ValueError("y_bounds must satisfy lower < upper.")

    return torch.linspace(
        lo,
        hi,
        grid_size,
        dtype=torch.float64,
        device=device,
    )


def _to_internal_y_bounds(
    y_bounds: Optional[Tuple[float, float]],
    y_scaler,
) -> Optional[Tuple[float, float]]:
    """Convert user-facing y_bounds to scaled internal bounds if y_scaler exists."""
    if y_bounds is None:
        return None

    lo, hi = map(float, y_bounds)
    if not lo < hi:
        raise ValueError("y_bounds must satisfy lower < upper.")

    if y_scaler is None:
        return lo, hi

    vals = np.array([[lo], [hi]], dtype=float)
    vals_scaled = y_scaler.transform(vals).reshape(-1)
    lo_s, hi_s = float(vals_scaled[0]), float(vals_scaled[1])

    return min(lo_s, hi_s), max(lo_s, hi_s)

def centered_kernel_row(model: KPCAResult, z: Array) -> Tuple[Array, np.float64]:
    """Compute centered kernel row k_c(z) and k_c(z,z) vs training set."""
    Ztr = model.Z_train
    kfun = model.kernel_cfg.kernel_fn()
    k_row = kfun(z[None, :], Ztr).ravel()  # (n,)
    row_mean = k_row.mean()
    k_c_row = k_row - row_mean - model.bar_k + model.bar_K

    k_self = kfun(z[None, :], z[None, :])[0, 0]
    k_c_self = k_self - 2 * row_mean + model.bar_K
    return k_c_row, k_c_self

def kpca_feature_error(model: KPCAResult, z: Array) -> np.float64:
    """Feature-space orthogonal squared error: E_phi(z) = k_c(z,z) - ||t(z)||^2."""
    if model.m == 0:
        _, k_c_self = centered_kernel_row(model, z)
        return np.float64(k_c_self)
    k_c_row, k_c_self = centered_kernel_row(model, z)
    t = model.A_m.T @ k_c_row
    return np.float64(np.maximum(0.0, k_c_self - np.dot(t, t)))

def kpca_train_rmse_feature(model: KPCAResult) -> np.float64:
    """Training mean feature-space error using eigen shortcut."""
    n = model.Kc.shape[0]

    if model.m == 0:
        mse = model.lambdas.mean()
        return np.float64(np.sqrt(np.maximum(0.0, mse)))
    
    residual_lambdas = model.lambdas[model.m:]
    Q_res = model.Q[:, model.m:]

    E_per = (Q_res**2) @ residual_lambdas
    mse = E_per.mean()

    return np.float64(np.sqrt(np.maximum(0.0, mse)))

def kpca_test_rmse_feature(model: KPCAResult, Z_test: Array) -> np.float64:
    errs = [kpca_feature_error(model, z) for z in Z_test]
    mse = np.mean(errs)
    return np.float64(np.sqrt(np.maximum(0.0, mse)))

def _torch_kernel_from_cfg(cfg: KernelConfig):
    if cfg.kind == "rbf":
        sigma = np.float64(cfg.params.get("sigma", 1.0))
        inv2sigma2 = 1.0 / (2.0 * sigma * sigma)
        def k(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            A_sq = (A**2).sum(dim=1, keepdims=True)
            B_sq = (B**2).sum(dim=1, keepdims=True).T
            dist2 = A_sq + B_sq - 2.0 * (A @ B.T)
            return torch.exp(-dist2 * inv2sigma2)
        return k
    elif cfg.kind == "poly":
        degree = int(cfg.params.get("degree", 2))
        c0 = np.float64(cfg.params.get("c0", 0.0))
        def k(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            return (A @ B.T + c0) ** degree
        return k
    else:
        raise ValueError("Unknown kernel kind for torch")


class TorchProjector_Kernel:
    """Optimizes y for a KPCAResult (Kernel/Gram) model."""
    def __init__(self, model: KPCAResult, k:int = 0, lambda_knn: np.float64 = 0.0, device: Optional[str] = None):
        self.model = model
        self.device = torch.device(device) if device else torch.device('cpu')
        
        self.x_scaler = getattr(model, 'x_scaler', None)
        self.y_scaler = getattr(model, 'y_scaler', None)

        self.Ztr = torch.tensor(model.Z_train, dtype=torch.float64, device=self.device)   # (n,p)
        self.A_m = torch.tensor(model.A_m, dtype=torch.float64, device=self.device)       # (n,m)
        self.bar_k = torch.tensor(model.bar_k, dtype=torch.float64, device=self.device)   # (n,)
        self.bar_K = torch.tensor([model.bar_K], dtype=torch.float64, device=self.device) # (1,)
        self.k_torch = _torch_kernel_from_cfg(model.kernel_cfg)

        self.n = self.Ztr.shape[0]
        self.p = self.Ztr.shape[1]
        self.m = self.A_m.shape[1]

        self.k = k
        self.lambda_knn = lambda_knn

        self.Xtr = torch.tensor(model.Z_train[:, :-1], dtype=torch.float64, device=self.device) # (n, p-1)
        self.ytr = torch.tensor(model.Z_train[:, -1], dtype=torch.float64, device=self.device) # (n,)

        self.mean_y = np.float64(model.Z_train[:, -1].mean())
        self._calibrate_loss_scales()
        
    def E_phi_batch(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        b = X.shape[0]
        Z = torch.cat([X, Y], dim=1)

        k_matrix = self.k_torch(Z, self.Ztr)
        row_means = k_matrix.mean(dim=1, keepdim=True)

        k_c_matrix = k_matrix - row_means - self.bar_k + self.bar_K

        if self.m == 0:
            proj_energy = torch.zeros(b, dtype=torch.float64, device=self.device)
        else:
            T = (self.A_m.T @ k_c_matrix.T).T
            proj_energy = (T * T).sum(dim=1)

        k_selfs = torch.diag(self.k_torch(Z, Z))

        k_c_selfs = k_selfs - 2.0 * row_means.squeeze() + self.bar_K[0]
        E = k_c_selfs - proj_energy
        return k_c_selfs - proj_energy
    
    def _find_knn_neighbors(self, X: torch.Tensor, exclude_self = False) -> Tuple[torch.Tensor, torch.Tensor]:
        dist2_matrix = torch.cdist(X, self.Xtr, p=2)**2

        if exclude_self and X.shape[0] == self.Xtr.shape[0]:
            dist2_matrix = dist2_matrix.clone()
            idx = torch.arange(X.shape[0], device=self.device)
            dist2_matrix[idx, idx] = float("inf")
        
        neighbor_dists_sq, neighbor_indices = torch.topk(
            dist2_matrix + 1e-8, 
            k=self.k, 
            dim=1, 
            largest=False
        )

        y_neighbors = self.ytr[neighbor_indices]
        
        neighbor_dists = torch.sqrt(neighbor_dists_sq)
        tau = neighbor_dists[:, -1:].clamp(min=1e-6)
        weights = torch.exp(-neighbor_dists_sq / (2.0 * tau * tau))
        weights = weights / weights.sum(dim=1, keepdims=True)

        return y_neighbors, weights
    
    def R_knn_batch(self, 
                    Y: torch.Tensor, 
                    y_neighbors: torch.Tensor, 
                    weights: torch.Tensor) -> torch.Tensor:
        Y_expanded = Y.expand(-1, self.k)
        
        diff_sq = (Y_expanded - y_neighbors) ** 2
        weighted_diff_sq = weights * diff_sq
        
        R_knn = weighted_diff_sq.sum(dim=1)
        return R_knn
    
    def _calibrate_loss_scales(self):
        with torch.no_grad():
            X_ref = self.Xtr
            Y_ref = torch.full(
                (self.Xtr.shape[0], 1),
                self.mean_y,
                dtype=torch.float64,
                device=self.device
            )

            E_ref = self.E_phi_batch(X_ref, Y_ref)
            self.ephi_scale = torch.clamp(E_ref.median(), min=1e-8)

            if self.k > 0 and self.lambda_knn > 0.0:
                y_neighbors, weights = self._find_knn_neighbors(X_ref, exclude_self=True)
                R_ref = self.R_knn_batch(Y_ref, y_neighbors, weights)
                self.rknn_scale = torch.clamp(R_ref.median(), min=1e-8)
            else:
                self.rknn_scale = torch.tensor(1.0, dtype=torch.float64, device=self.device)
    
    def total_loss_batch(self, 
                         X: torch.Tensor, 
                         Y: torch.Tensor,
                         y_neighbors: Optional[torch.Tensor],
                         weights: Optional[torch.Tensor]) -> torch.Tensor:
        E_phi = self.E_phi_batch(X, Y) / self.ephi_scale

        if self.lambda_knn == 0.0 or self.k == 0:
            return E_phi

        R_knn = self.R_knn_batch(Y, y_neighbors, weights) / self.rknn_scale
        return E_phi + self.lambda_knn * R_knn

    def predict_y_batch(
        self,
        X_np: Array,
        y_init: Optional[Array] = None,
        lr: np.float64 = 0.05,
        steps: int = 300,
        restarts: int = 1,
        init_perturb: np.float64 = 0.5,
        prediction_optimizer: str = "torch",
        grid_size: int = 101,
        grid_refine: bool = True,
        y_bounds: Optional[Tuple[float, float]] = None,
        y_margin_fraction: float = 0.15, ) -> Array:
        """
        Predict y by minimizing the kernel/moment NPCA objective.

        prediction_optimizer:
            "torch":
                Current behavior. Direct PyTorch gradient descent.

            "grid":
                Try a grid of y-values and pick the one with lowest loss.

            "grid_then_torch":
                First grid search, then use the best grid value as initialization
                for PyTorch gradient descent.
        """
        prediction_optimizer = _validate_prediction_optimizer(prediction_optimizer)

        if self.x_scaler is not None:
            X_scaled = self.x_scaler.transform(X_np)
        else:
            X_scaled = X_np

        X = torch.tensor(X_scaled, dtype=torch.float64, device=self.device)
        b = X.shape[0]

        with torch.no_grad():
            if self.k > 0 and self.lambda_knn > 0.0:
                y_neighbors_precomputed, weights_precomputed = self._find_knn_neighbors(X)
            else:
                y_neighbors_precomputed, weights_precomputed = None, None

        y_bounds_internal = _to_internal_y_bounds(y_bounds, self.y_scaler)

        if prediction_optimizer in {"grid", "grid_then_torch"}:
            y_grid = _make_y_grid(
                self.ytr,
                grid_size=grid_size,
                margin_fraction=y_margin_fraction,
                y_bounds=y_bounds_internal,
                device=self.device,
            )

            best_Y_grid = torch.zeros(b, 1, dtype=torch.float64, device=self.device)
            best_vals_grid = torch.full(
                (b,),
                np.float64("inf"),
                dtype=torch.float64,
                device=self.device,
            )

            with torch.no_grad():
                for y_value in y_grid:
                    Y_candidate = torch.full(
                        (b, 1),
                        float(y_value),
                        dtype=torch.float64,
                        device=self.device,
                    )

                    current_vals = self.total_loss_batch(
                        X,
                        Y_candidate,
                        y_neighbors_precomputed,
                        weights_precomputed,
                    )

                    is_better = current_vals < best_vals_grid
                    best_vals_grid[is_better] = current_vals[is_better]
                    best_Y_grid[is_better] = Y_candidate[is_better]

            if prediction_optimizer == "grid" or not grid_refine:
                y_out = best_Y_grid.squeeze(dim=1).cpu().numpy()

                if self.y_scaler is not None:
                    y_out = self.y_scaler.inverse_transform(
                        y_out.reshape(-1, 1)
                    ).ravel()

                return y_out

            y_init_internal = best_Y_grid.squeeze(dim=1).detach().cpu().numpy()

        else:
            y_init_internal = None

        best_Y = torch.zeros(b, 1, dtype=torch.float64, device=self.device)
        best_vals = torch.full(
            (b,),
            np.float64("inf"),
            dtype=torch.float64,
            device=self.device,
        )

        restarts = max(1, int(restarts))

        for r in range(restarts):
            if y_init_internal is not None:
                base_init = torch.tensor(
                    y_init_internal,
                    dtype=torch.float64,
                    device=self.device,
                ).view(-1, 1)

            elif y_init is not None:
                if self.y_scaler is not None:
                    y_init_scaled = self.y_scaler.transform(
                        np.asarray(y_init).reshape(-1, 1)
                    ).ravel()
                else:
                    y_init_scaled = np.asarray(y_init, dtype=float).reshape(-1)

                base_init = torch.tensor(
                    y_init_scaled,
                    dtype=torch.float64,
                    device=self.device,
                ).view(-1, 1)

            else:
                base_init = torch.full(
                    (b, 1),
                    self.mean_y,
                    dtype=torch.float64,
                    device=self.device,
                )

            if restarts > 1:
                perturbation = torch.randn(
                    b,
                    1,
                    dtype=torch.float64,
                    device=self.device,
                ) * float(init_perturb)
                Y0 = base_init + perturbation
            else:
                Y0 = base_init

            Y = Y0.clone().detach().requires_grad_(True)
            opt = torch.optim.AdamW([Y], lr=float(lr))

            for _ in range(int(steps)):
                opt.zero_grad(set_to_none=True)

                loss = self.total_loss_batch(
                    X,
                    Y,
                    y_neighbors_precomputed,
                    weights_precomputed,
                ).sum()

                loss.backward()
                opt.step()

            with torch.no_grad():
                current_vals = self.total_loss_batch(
                    X,
                    Y,
                    y_neighbors_precomputed,
                    weights_precomputed,
                )

                is_better = current_vals < best_vals
                best_vals[is_better] = current_vals[is_better]
                best_Y[is_better] = Y[is_better]

        y_out = best_Y.squeeze(dim=1).cpu().numpy()

        if self.y_scaler is not None:
            y_out = self.y_scaler.inverse_transform(
                y_out.reshape(-1, 1)
            ).ravel()

        return y_out

class TorchProjector_Moment:
    """Optimizes y for a MomentPolynomialPCA model."""
    def __init__(self, model: MomentPolynomialPCA, k:int = 0, lambda_knn: np.float64 = 0.0, device: Optional[str] = None):
        self.model = model
        self.device = torch.device(device) if device else torch.device('cpu')

        self.x_scaler = getattr(model, 'x_scaler', None)
        self.y_scaler = getattr(model, 'y_scaler', None)

        self.p = model.p_
        self.m = model.m
        self.d = model.degree
        self.const = model.const
        
        self.k = k
        self.lambda_knn = lambda_knn
        if hasattr(model, 'Z_train_') and model.Z_train_ is not None:
            self.Xtr = torch.tensor(model.Z_train_[:, :-1], dtype=torch.float64, device=self.device) # (n, p-1)
            self.ytr = torch.tensor(model.Z_train_[:, -1], dtype=torch.float64, device=self.device) # (n,)

            self.mean_y = np.float64(model.Z_train_[:, -1].mean())
        else:
            print("Warning: MomentPolynomialPCA model has no Z_train_data. kNN disabled.")
            self.k = 0
            self.lambda_knn = 0.0
            self.Xtr = torch.empty(0, 0, device=self.device)
            self.ytr = torch.empty(0, 0, device=self.device)
            self.mean_y = 0.0

        self.exponents = torch.tensor(np.array(model.exponents_), dtype=torch.int64, device=self.device)
        self.bin_sqrts = torch.tensor(model.bin_sqrts_, dtype=torch.float64, device=self.device).view(-1, 1)
        self.d_p = self.exponents.shape[0]

        self.V_m = torch.tensor(model.eigvecs_, dtype=torch.float64, device=self.device)
        self.m_feat = torch.tensor(model.m_feat_, dtype=torch.float64, device=self.device).view(-1, 1)
        self.has_const = (self.const is not None) and (self.const != 0.0)
        self.sqrt_u = np.float64(self.const) ** 0.5 if self.has_const else 0.0
        self._calibrate_loss_scales()

    def _torch_feature_map(self, Z_batch: torch.Tensor) -> torch.Tensor:
        Z_trans = Z_batch.T
        all_powers = Z_trans.unsqueeze(0) ** self.exponents.unsqueeze(2)
        Phi = torch.prod(all_powers, dim=1)
        return self.bin_sqrts * Phi

    def E_phi_batch(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        b = X.shape[0]
        if self.has_const:
            sqrt_u_tensor = torch.full((b, 1), self.sqrt_u, device=self.device)
            Z_batch = torch.cat([X, Y, sqrt_u_tensor], dim=1)
        else:
            Z_batch = torch.cat([X, Y], dim=1)

        Phi = self._torch_feature_map(Z_batch)
        Phi_c = Phi - self.m_feat
        total_sq_norm = torch.sum(Phi_c * Phi_c, dim=0)
        
        if self.m == 0:
            return total_sq_norm
            
        T = self.V_m.T @ Phi_c
        proj_sq_norm = torch.sum(T * T, dim=0)
        error = total_sq_norm - proj_sq_norm
        return torch.clamp(error, min=0.0)
    
    def _find_knn_neighbors(self, X: torch.Tensor, exclude_self: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        dist2_matrix = torch.cdist(X, self.Xtr, p=2) ** 2

        if exclude_self and X.shape[0] == self.Xtr.shape[0]:
            dist2_matrix = dist2_matrix.clone()
            idx = torch.arange(X.shape[0], device=self.device)
            dist2_matrix[idx, idx] = float("inf")

        neighbor_dists_sq, neighbor_indices = torch.topk(
            dist2_matrix + 1e-8,
            k=self.k,
            dim=1,
            largest=False
        )

        y_neighbors = self.ytr[neighbor_indices]

        neighbor_dists = torch.sqrt(neighbor_dists_sq)
        tau = neighbor_dists[:, -1:].clamp(min=1e-6)
        weights = torch.exp(-neighbor_dists_sq / (2.0 * tau * tau))
        weights = weights / weights.sum(dim=1, keepdims=True)

        return y_neighbors, weights
    
    def R_knn_batch(self, 
                    Y: torch.Tensor, 
                    y_neighbors: torch.Tensor, 
                    weights: torch.Tensor) -> torch.Tensor:
        Y_expanded = Y.expand(-1, self.k)
        
        diff_sq = (Y_expanded - y_neighbors) ** 2
        weighted_diff_sq = weights * diff_sq
        
        R_knn = weighted_diff_sq.sum(dim=1)
        return R_knn
    
    def _calibrate_loss_scales(self):
        with torch.no_grad():
            if self.Xtr.numel() == 0:
                self.ephi_scale = torch.tensor(1.0, dtype=torch.float64, device=self.device)
                self.rknn_scale = torch.tensor(1.0, dtype=torch.float64, device=self.device)
                return

            X_ref = self.Xtr
            Y_ref = torch.full(
                (self.Xtr.shape[0], 1),
                self.mean_y,
                dtype=torch.float64,
                device=self.device
            )

            E_ref = self.E_phi_batch(X_ref, Y_ref)
            self.ephi_scale = torch.clamp(E_ref.median(), min=1e-8)

            if self.k > 0 and self.lambda_knn > 0.0:
                y_neighbors, weights = self._find_knn_neighbors(X_ref, exclude_self=True)
                R_ref = self.R_knn_batch(Y_ref, y_neighbors, weights)
                self.rknn_scale = torch.clamp(R_ref.median(), min=1e-8)
            else:
                self.rknn_scale = torch.tensor(1.0, dtype=torch.float64, device=self.device)
    
    def total_loss_batch(self, 
                         X: torch.Tensor, 
                         Y: torch.Tensor,
                         y_neighbors: Optional[torch.Tensor],
                         weights: Optional[torch.Tensor]) -> torch.Tensor:
        E_phi = self.E_phi_batch(X, Y) / self.ephi_scale

        if self.lambda_knn == 0.0 or self.k == 0:
            return E_phi

        R_knn = self.R_knn_batch(Y, y_neighbors, weights) / self.rknn_scale
        return E_phi + self.lambda_knn * R_knn

    def predict_y_batch(
        self,
        X_np: Array,
        y_init: Optional[Array] = None,
        lr: np.float64 = 0.05,
        steps: int = 300,
        restarts: int = 1,
        init_perturb: np.float64 = 0.5,
        prediction_optimizer: str = "torch",
        grid_size: int = 101,
        grid_refine: bool = True,
        y_bounds: Optional[Tuple[float, float]] = None,
        y_margin_fraction: float = 0.15,) -> Array:
        """
        Predict y by minimizing the kernel/moment NPCA objective.

        prediction_optimizer:
            "torch":
                Current behavior. Direct PyTorch gradient descent.

            "grid":
                Try a grid of y-values and pick the one with lowest loss.

            "grid_then_torch":
                First grid search, then use the best grid value as initialization
                for PyTorch gradient descent.
        """
        prediction_optimizer = _validate_prediction_optimizer(prediction_optimizer)

        if self.x_scaler is not None:
            X_scaled = self.x_scaler.transform(X_np)
        else:
            X_scaled = X_np

        X = torch.tensor(X_scaled, dtype=torch.float64, device=self.device)
        b = X.shape[0]

        with torch.no_grad():
            if self.k > 0 and self.lambda_knn > 0.0:
                y_neighbors_precomputed, weights_precomputed = self._find_knn_neighbors(X)
            else:
                y_neighbors_precomputed, weights_precomputed = None, None

        y_bounds_internal = _to_internal_y_bounds(y_bounds, self.y_scaler)

        if prediction_optimizer in {"grid", "grid_then_torch"}:
            y_grid = _make_y_grid(
                self.ytr,
                grid_size=grid_size,
                margin_fraction=y_margin_fraction,
                y_bounds=y_bounds_internal,
                device=self.device,
            )

            best_Y_grid = torch.zeros(b, 1, dtype=torch.float64, device=self.device)
            best_vals_grid = torch.full(
                (b,),
                np.float64("inf"),
                dtype=torch.float64,
                device=self.device,
            )

            with torch.no_grad():
                for y_value in y_grid:
                    Y_candidate = torch.full(
                        (b, 1),
                        float(y_value),
                        dtype=torch.float64,
                        device=self.device,
                    )

                    current_vals = self.total_loss_batch(
                        X,
                        Y_candidate,
                        y_neighbors_precomputed,
                        weights_precomputed,
                    )

                    is_better = current_vals < best_vals_grid
                    best_vals_grid[is_better] = current_vals[is_better]
                    best_Y_grid[is_better] = Y_candidate[is_better]

            if prediction_optimizer == "grid" or not grid_refine:
                y_out = best_Y_grid.squeeze(dim=1).cpu().numpy()

                if self.y_scaler is not None:
                    y_out = self.y_scaler.inverse_transform(
                        y_out.reshape(-1, 1)
                    ).ravel()

                return y_out

            y_init_internal = best_Y_grid.squeeze(dim=1).detach().cpu().numpy()

        else:
            y_init_internal = None

        best_Y = torch.zeros(b, 1, dtype=torch.float64, device=self.device)
        best_vals = torch.full(
            (b,),
            np.float64("inf"),
            dtype=torch.float64,
            device=self.device,
        )

        restarts = max(1, int(restarts))

        for r in range(restarts):
            if y_init_internal is not None:
                base_init = torch.tensor(
                    y_init_internal,
                    dtype=torch.float64,
                    device=self.device,
                ).view(-1, 1)

            elif y_init is not None:
                if self.y_scaler is not None:
                    y_init_scaled = self.y_scaler.transform(
                        np.asarray(y_init).reshape(-1, 1)
                    ).ravel()
                else:
                    y_init_scaled = np.asarray(y_init, dtype=float).reshape(-1)

                base_init = torch.tensor(
                    y_init_scaled,
                    dtype=torch.float64,
                    device=self.device,
                ).view(-1, 1)

            else:
                base_init = torch.full(
                    (b, 1),
                    self.mean_y,
                    dtype=torch.float64,
                    device=self.device,
                )

            if restarts > 1:
                perturbation = torch.randn(
                    b,
                    1,
                    dtype=torch.float64,
                    device=self.device,
                ) * float(init_perturb)
                Y0 = base_init + perturbation
            else:
                Y0 = base_init

            Y = Y0.clone().detach().requires_grad_(True)
            opt = torch.optim.AdamW([Y], lr=float(lr))

            for _ in range(int(steps)):
                opt.zero_grad(set_to_none=True)

                loss = self.total_loss_batch(
                    X,
                    Y,
                    y_neighbors_precomputed,
                    weights_precomputed,
                ).sum()

                loss.backward()
                opt.step()

            with torch.no_grad():
                current_vals = self.total_loss_batch(
                    X,
                    Y,
                    y_neighbors_precomputed,
                    weights_precomputed,
                )

                is_better = current_vals < best_vals
                best_vals[is_better] = current_vals[is_better]
                best_Y[is_better] = Y[is_better]

        y_out = best_Y.squeeze(dim=1).cpu().numpy()

        if self.y_scaler is not None:
            y_out = self.y_scaler.inverse_transform(
                y_out.reshape(-1, 1)
            ).ravel()

        return y_out
