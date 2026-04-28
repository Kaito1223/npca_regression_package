from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

try:
    from scipy.spatial.distance import cdist
except ImportError:
    cdist = None


Array = np.ndarray


def zscore(X: Array) -> Tuple[Array, Array, Array]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu) / sd, mu, sd


def pairwise_sq_dists_numpy(X1: Array, X2: Array) -> Array:
    X1s = np.sum(X1**2, axis=1, keepdims=True)
    X2s = np.sum(X2**2, axis=1, keepdims=True).T
    return np.maximum(X1s + X2s - 2 * X1 @ X2.T, 0.0)


def K_radial(u: Array, name: str) -> Array:
    if name == "rbf":
        return np.exp(-0.5 * u**2)

    if name == "epanechnikov":
        w = 0.75 * (1 - u**2)
        return np.where(u <= 1, np.maximum(w, 0.0), 0.0)

    if name == "tricube":
        w = (1 - np.abs(u) ** 3) ** 3
        return np.where(np.abs(u) <= 1, w, 0.0)

    if name == "laplace":
        return 0.5 * np.exp(-np.abs(u))

    if name == "uniform":
        return 0.5 * (np.abs(u) <= 1).astype(float)

    if name == "triangular":
        return np.maximum(1 - np.abs(u), 0.0)

    raise ValueError(f"Unknown kernel '{name}'")


def loo_from_hat(y: Array, yhat: Array, sdiag: Array, eps: float = 1e-12) -> Array:
    den = 1 - np.clip(sdiag, 0.0, 1.0 - eps)
    return (yhat - sdiag * y) / den


def rmse(a: Array, b: Array) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: Array, b: Array) -> float:
    return float(np.mean(np.abs(a - b)))


@dataclass
class NadarayaWatsonRegressor:
    kernel: Literal[
        "rbf",
        "epanechnikov",
        "tricube",
        "laplace",
        "uniform",
        "triangular",
    ] = "rbf"
    h: float = 1.0
    per_feature_bandwidth: Optional[Array] = None
    standardize: bool = True
    truncate: Optional[float] = None

    X_: Optional[Array] = None
    y_: Optional[Array] = None
    mu_: Optional[Array] = None
    sd_: Optional[Array] = None
    _train_dists: Optional[Array] = None

    def _prep(self, X: Array) -> Array:
        X = np.atleast_2d(X)
        return (X - self.mu_) / self.sd_ if self.standardize else X

    def fit(self, X: Array, y: Array) -> "NadarayaWatsonRegressor":
        X = np.atleast_2d(X)
        y = np.asarray(y).reshape(-1)

        if self.standardize:
            self.X_, self.mu_, self.sd_ = zscore(X)
        else:
            self.X_ = X
            self.mu_ = np.zeros((1, X.shape[1]))
            self.sd_ = np.ones((1, X.shape[1]))

        self.y_ = y
        return self

    def _get_distances(self, Xq: Array, X: Array) -> Array:
        if self.per_feature_bandwidth is not None:
            if cdist is None:
                raise ImportError("SciPy is required for per_feature_bandwidth.")

            h_sq = np.asarray(self.per_feature_bandwidth).flatten() ** 2
            return cdist(Xq, X, metric="seuclidean", V=h_sq)

        if cdist is not None:
            return cdist(Xq, X, metric="euclidean")

        return np.sqrt(pairwise_sq_dists_numpy(Xq, X))

    def _weights(
        self,
        Xq: Array,
        h: Optional[float] = None,
        _dists: Optional[Array] = None,
    ) -> Array:
        h = h if h is not None else self.h
        Xq_prep = self._prep(np.atleast_2d(Xq))

        if self.per_feature_bandwidth is not None:
            u = self._get_distances(Xq_prep, self.X_)
        else:
            dists = _dists if _dists is not None else self._get_distances(Xq_prep, self.X_)
            u = dists / (h + 1e-12)

        W = K_radial(u, self.kernel)

        if self.truncate is not None:
            W = np.where(u <= self.truncate, W, 0.0)

        denom = W.sum(axis=1, keepdims=True) + 1e-12
        return W / denom

    def predict(self, Xq: Array) -> Array:
        W = self._weights(Xq)
        return W @ self.y_

    def smoother_matrix(self, h: Optional[float] = None) -> Tuple[Array, Array]:
        if self._train_dists is None:
            self._train_dists = self._get_distances(self.X_, self.X_)

        W = self._weights(self.X_, h=h, _dists=self._train_dists)
        return W, W.diagonal()

    def predict_train(self) -> Array:
        W, _ = self.smoother_matrix()
        return W @ self.y_

    def loo_predict_train(self) -> Array:
        W, sdiag = self.smoother_matrix()
        yhat = W @ self.y_
        return loo_from_hat(self.y_, yhat, sdiag)

    def select_bandwidth(self, h_grid: Array) -> float:
        best_h, best_cv = self.h, np.inf
        self._train_dists = self._get_distances(self.X_, self.X_)

        for h_val in h_grid:
            W, sdiag = self.smoother_matrix(h=float(h_val))
            yhat = W @ self.y_
            yloo = loo_from_hat(self.y_, yhat, sdiag)
            cv = rmse(self.y_, yloo)

            if cv < best_cv:
                best_cv, best_h = cv, float(h_val)

        self.h = best_h
        self._train_dists = None
        return best_h

    def score(self, kind: Literal["mse", "mae"] = "mse", loo: bool = False) -> float:
        ypred = self.loo_predict_train() if loo else self.predict_train()
        fn = rmse if kind == "mse" else mae
        return fn(self.y_, ypred)


@dataclass
class KernelRegression:
    kernel: str = "rbf"
    h: float = 1.0
    standardize: bool = True
    per_feature_bandwidth: Optional[Array] = None

    _model: Optional[NadarayaWatsonRegressor] = None

    def fit(self, X: Array, y: Array) -> "KernelRegression":
        self._model = NadarayaWatsonRegressor(
            kernel=self.kernel,
            h=self.h,
            per_feature_bandwidth=self.per_feature_bandwidth,
            standardize=self.standardize,
        ).fit(X, y)

        return self

    def predict(self, Xq: Array) -> Array:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        return self._model.predict(Xq)

    def select_bandwidth(self, h_grid: Array) -> float:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        return self._model.select_bandwidth(h_grid)

    def score(self, kind: Literal["mse", "mae"] = "mse", loo: bool = False) -> float:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        return self._model.score(kind=kind, loo=loo)

    def __getattr__(self, name):
        if self._model is not None and hasattr(self._model, name):
            return getattr(self._model, name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")