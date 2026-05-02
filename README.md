# NPCA Regression

This package implements an NPCA-based regression method.

The main idea is to treat the full data point as

```python
Z = [X, y]
```

and learn a nonlinear structure in the joint space using Kernel PCA or moment-based polynomial PCA.  
For prediction, the input `X_test` is fixed, and the target value `y` is found by minimizing the feature-space reconstruction error.

The package also includes baseline regression methods for comparison:

- Kernel regression / Nadaraya-Watson smoothing
- Thin plate spline / polyharmonic spline smoothing

---

## Project Structure

```text
npca_regression_package/
├── pyproject.toml
├── README.md
├── src/
│   └── npca_regression/
│       ├── kernel/
│           ├── __init__.py
│           ├── datasets.py
│           ├── model_selection.py
│           ├── pca.py
│           ├── projectors.py
│           ├── regression.py
│       └── baselines/
│           ├── __init__.py
│           ├── kernel_regression.py
│           └── spline_smoothing.py
|       └── neural/
|           ├── __init__.py
|           ├── ultis.py
            ...
        ...
├── examples/
│   ├── ccpp_example.py
│   ├── synthetic_2d_example.py
│   └── synthetic_3d_example.py
├── benchmarks/
│   └── compare_npca_spline_kernel.py
└── tests/
    └── test_imports.py
```
---

## Installation

From the root folder of the project, run:

```bash
python -m pip install -e .
```
---

## Basic Usage

```python
import numpy as np

from npca_regression import fit_models, predict
from npca_regression.datasets import make_data_combined


Z_train, Z_test = make_data_combined(
    n_train=450,
    n_test=100,
    n_features=1,
    seed=0,
    noise_level=0.15,
)

X_test = Z_test[:, :-1]
y_test = Z_test[:, -1]

models = fit_models(
    Z_train,
    method="kernel",
    evr_target=0.99,
    m=None,
    scale_data=True,
    kernel_choice="all",
    poly_degrees=[2, 3, 4],
    rbf_sigmas=[0.5, 1.0, 2.0],
    constant=1.0,
)

df_results, predictions = predict(
    models,
    X_test,
    y_test=y_test,
    k=8,
    lambda_knn=0.0,
    lr=0.05,
    steps=300,
)

print(df_results)
```

---

## Main API

The main NPCA regression functions are:

```python
from npca_regression import fit_models, predict
```

### `fit_models`

Fits one or more NPCA models.

Example:

```python
models = fit_models(
    Z_train,
    method="kernel",
    evr_target=0.99,
    m=None,
    scale_data=True,
    kernel_choice="all",
    poly_degrees=[2, 3],
    rbf_sigmas=[0.5, 1.0, 2.0],
    constant=1.0,
)
```

Important arguments:

```text
Z_train       Full training data with y as the last column
method        "kernel" or "moment"
evr_target    Explained-variance threshold for choosing components
m             Fixed number of components, if not using evr_target
scale_data    Whether to standardize X and y before fitting
```

### `predict`

Predicts target values for new input data.

Example:

```python
df_results, predictions = predict(
    models,
    X_test,
    y_test=y_test,
    k=8,
    lambda_knn=0.0,
    lr=0.05,
    steps=300,
)
```

Important arguments:

```text
models        Output from fit_models(...)
X_test        Test input features
y_test        Optional true targets, used for RMSE reporting
k             Number of neighbors for optional kNN regularization
lambda_knn    Weight of the kNN regularization term
lr            Learning rate for optimizing y
steps         Number of optimization steps
```

The function returns:

```text
df_results     DataFrame containing RMSE and model information
predictions    List containing raw predicted values for each fitted model
```

---

## Synthetic Datasets

Synthetic datasets are provided through:

```python
from npca_regression.datasets import make_data_separated, make_data_combined
```

### Separate Format

```python
X_train, y_train, X_test, y_test = make_data_separated(
    n_train=400,
    n_test=100,
    n_features=1,
    seed=0,
    noise_level=0.15,
)
```

### Combined Format

```python
Z_train, Z_test = make_data_combined(
    n_train=400,
    n_test=100,
    n_features=1,
    seed=0,
    noise_level=0.15,
)
```

The combined format stores the target as the last column:

```python
X_train = Z_train[:, :-1]
y_train = Z_train[:, -1]
```

This is the format used by the NPCA method.

---

## Baselines

The package includes baseline methods under:

```python
npca_regression.baselines
```

You can import them with:

```python
from npca_regression.baselines import KernelRegression, TPSRegressor
```

or explicitly:

```python
from npca_regression.baselines.kernel_regression import KernelRegression
from npca_regression.baselines.spline_smoothing import TPSRegressor
```

### Kernel Regression Baseline

```python
from npca_regression.baselines import KernelRegression

model = KernelRegression(
    kernel="rbf",
    standardize=True,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

This baseline uses Nadaraya-Watson kernel regression.

### Thin Plate Spline Baseline

```python
from npca_regression.baselines import TPSRegressor

model = TPSRegressor(
    lambda_=0.01,
    m=2,
    standardize=True,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

This baseline uses a polyharmonic spline / thin-plate-spline-style smoother.

---

## Examples

Run the 2D synthetic example:

```bash
python examples/synthetic_2d_example.py
```

Run the 3D synthetic example:

```bash
python examples/synthetic_3d_example.py
```

Run the CCPP real-data example:

```bash
python examples/ccpp_example.py
```

---

## Benchmarks

The benchmark script compares NPCA regression against the baseline methods:

```bash
python benchmarks/compare_npca_spline_kernel.py
```

```python
X_train, y_train, X_test, y_test = make_data_separated(...)

Z_train = np.column_stack([X_train, y_train])
```

---