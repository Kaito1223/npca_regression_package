import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from npca_regression.kernel import fit_models, predict
from npca_regression.datasets import make_data_combined


PCA_METHOD = "kernel"
EVR_TARGET = 0.95
n_components = None

kernel_params = {
    "kernel_choice": "all",
    "poly_degrees": [2, 3, 4],
    "rbf_sigmas": [0.5, 1.0, 2.0],
    "constant": 1.0,
}

knn_params = {
    "k": 15,
    "lambda_knn": 0,
}

torch_params = {
    "torch_device": "cuda",
    "lr": 0.01,
    "steps": 300,
    "restarts": 1,
    "init_perturb": 0.5,
}

Z_train, Z_test = make_data_combined(
    n_train=450,
    n_test=100,
    n_features=1,
    seed=0,
    noise_level=0.15,
)

X_train = Z_train[:, :-1]
y_train = Z_train[:, -1]

X_test = Z_test[:, :-1]
y_test = Z_test[:, -1]

Xq = np.linspace(X_train.min(), X_train.max(), 400)[:, None]

fitted_models = fit_models(
    Z_train,
    method=PCA_METHOD,
    evr_target=EVR_TARGET,
    m=n_components,
    scale_data=True,
    **kernel_params,
)

df_test, results_test = predict(
    fitted_models,
    X_test,
    y_test=y_test,
    batch_size=None,
    **torch_params,
    **knn_params,
)

print("\n--- Synthetic 2D Test Results ---")
print(df_test)

df_grid, results_grid = predict(
    fitted_models,
    Xq,
    y_test=None,
    batch_size=None,
    **torch_params,
    **knn_params,
)

for row in results_grid:
    kernel = row["kernel"]
    param_label = row["params_label"]
    yq_pred = row["pred"]

    plt.figure(figsize=(8, 5))
    plt.plot(Xq[:, 0], yq_pred, label=f"{kernel}, {param_label}", linewidth=3.5)
    plt.scatter(X_train[:, 0], y_train, s=25, alpha=0.8, label="train")
    plt.scatter(X_test[:, 0], y_test, s=25, alpha=0.7, label="test")
    plt.xlabel("x")
    plt.ylabel("y / prediction")
    plt.title("NPCA Regression on Synthetic 2D Data")
    plt.legend()
    plt.tight_layout()

plt.show()