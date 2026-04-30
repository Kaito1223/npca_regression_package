from __future__ import annotations

import numpy as np

Array = np.ndarray


def orthogonalize_rows(W: Array, eps: float = 1e-12) -> Array:
    """Symmetric row orthogonalization: (W W^T)^(-1/2) W."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be a 2D matrix.")
    gram = W @ W.T
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, float(eps))
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return inv_sqrt @ W


def row_orthogonality_error(W: Array) -> float:
    W = np.asarray(W, dtype=float)
    eye = np.eye(W.shape[0])
    return float(np.linalg.norm(W @ W.T - eye, ord="fro"))
