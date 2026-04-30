"""Backward-compatible neural regression imports.

New code should import from ``npca_regression.neural.regressor`` and
``npca_regression.neural.evaluation``. This module is kept so older imports from
``npca_regression.neural.regression`` continue to work.
"""

from .evaluation import evaluate_neural_npca_regression
from .regressor import NeuralNPCARegressionConfig, NeuralNPCARegressor

__all__ = [
    "NeuralNPCARegressionConfig",
    "NeuralNPCARegressor",
    "evaluate_neural_npca_regression",
]
