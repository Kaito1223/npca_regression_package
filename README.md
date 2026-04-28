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
│       ├── __init__.py
│       ├── datasets.py
│       ├── kernels.py
│       ├── metrics.py
│       ├── model_selection.py
│       ├── pca.py
│       ├── poly_pca.py
│       ├── projectors.py
│       ├── regression.py
│       ├── NPCA_reg_op.py
│       └── baselines/
│           ├── __init__.py
│           ├── kernel_regression.py
│           └── spline_smoothing.py
├── examples/
│   ├── ccpp_example.py
│   ├── synthetic_2d_example.py
│   └── synthetic_3d_example.py
├── benchmarks/
│   └── compare_npca_spline_kernel.py
└── tests/
    └── test_imports.py
```

Only this folder is the actual importable Python package:

```text
src/npca_regression/
```

The folders `examples/`, `benchmarks/`, and `tests/` are project-level folders.  
They are not part of the package core.

---

## Installation

From the root folder of the project, run:

```bash
python -m pip install -e .
```

The `-e` means editable mode. This makes the package importable while still using the files in your current working directory.

After installation, this should work:

```bash
python -c "import npca_regression; print(npca_regression.__file__)"
```

The printed path should point to something like:

```text
.../npca_regression_package/src/npca_regression/__init__.py
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

The benchmark should use the same train/test split for all methods.

A clean comparison should follow this pattern:

```python
X_train, y_train, X_test, y_test = make_data_separated(...)

Z_train = np.column_stack([X_train, y_train])
```

Then:

- NPCA uses `Z_train`
- Kernel regression uses `X_train, y_train`
- Spline smoothing uses `X_train, y_train`

This keeps the comparison fair.

---

## Development Notes

This project uses a `src` layout.

That means this import:

```python
import npca_regression
```

comes from:

```text
src/npca_regression/
```

If Python cannot find the package, reinstall it in editable mode:

```bash
python -m pip install -e .
```

If VS Code still shows import warnings, reload the window or make sure the selected Python interpreter is the same one where the package was installed.

Generated folders such as:

```text
src/npca_regression.egg-info/
__pycache__/
```

should not be edited manually.

They can be ignored in Git using:

```gitignore
*.egg-info/
__pycache__/
*.pyc
build/
dist/
```

---

## Notes on Old Imports

Old local imports such as:

```python
from make_dataset import make_data_combined
from kernel_reg import KernelRegression
from spline_smoothing import TPSRegressor
```

should now be replaced by package imports:

```python
from npca_regression.datasets import make_data_combined
from npca_regression.baselines import KernelRegression, TPSRegressor
```

For NPCA regression, use:

```python
from npca_regression import fit_models, predict
```

instead of importing directly from old script-style files.