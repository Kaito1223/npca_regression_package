# npca-regression

Refactored package layout for the uploaded NPCA regression code.

## Install locally

```bash
pip install -e .
```

For example dependencies:

```bash
pip install -e '.[examples]'
```

## Basic usage

```python
import numpy as np
from npca_regression import fit_models, predict

Z_train = np.column_stack([X_train, y_train])

models = fit_models(
    Z_train,
    method="kernel",
    evr_target=0.99,
    scale_data=True,
    kernel_choice="all",
    poly_degrees=[2, 3],
    rbf_sigmas=[0.5, 1.0, 2.0],
)

results_df, predictions = predict(
    models,
    X_test,
    y_test=y_test,
    k=8,
    lambda_knn=0.0,
    lr=0.05,
    steps=300,
)
```

## Layout

```text
src/npca_regression/
  __init__.py
  kernels.py
  pca.py
  projectors.py
  regression.py
  metrics.py
  model_selection.py
  poly_pca.py        # compatibility wrapper for old imports
  NPCA_reg_op.py     # compatibility wrapper for old imports
examples/
  ccpp_example.py
benchmarks/
  compare_npca_spline_kernel.py
```

## Notes

The benchmark script still expects your existing local helper modules:
`make_dataset.py`, `spline_smoothing.py`, and `kernel_reg.py`. Those files
were referenced by the uploaded evaluation script but were not included in
this upload, so they were not fabricated here.
