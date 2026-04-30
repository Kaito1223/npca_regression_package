from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._utils import as_2d_float_array

Array = np.ndarray


@dataclass(frozen=True)
class ActivationStats:
    """Moment statistics used to make neural activation deterministic at inference."""

    second_moment: Array
    kurtosis_sign: Array


def estimate_activation_stats(Y: Array) -> ActivationStats:
    Y = as_2d_float_array(Y, name="Y")
    second_moment = np.mean(Y**2, axis=0, keepdims=True)
    fourth_moment = np.mean(Y**4, axis=0, keepdims=True)
    kappa4 = fourth_moment - 3.0 * second_moment**2
    sign = np.sign(kappa4)
    sign[sign == 0.0] = 1.0
    return ActivationStats(second_moment=second_moment, kurtosis_sign=sign)


def neural_activation(
    Y: Array,
    *,
    second_moment: Optional[Array] = None,
    kurtosis_sign: Optional[Array] = None,
) -> Array:
    """Cumulant-based activation from the neural NPCA prototype.

    During fitting, moments can be estimated from the current block by leaving
    ``second_moment`` and ``kurtosis_sign`` as None. During inference,
    fitted training moments should be supplied to avoid sample-dependent output.
    """
    Y = as_2d_float_array(Y, name="Y")
    if second_moment is None or kurtosis_sign is None:
        stats = estimate_activation_stats(Y)
        if second_moment is None:
            second_moment = stats.second_moment
        if kurtosis_sign is None:
            kurtosis_sign = stats.kurtosis_sign
    return np.asarray(kurtosis_sign, dtype=float) * (
        Y**3 - 3.0 * Y * np.asarray(second_moment, dtype=float)
    )
