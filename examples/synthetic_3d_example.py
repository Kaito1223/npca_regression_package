import numpy as np
import matplotlib.pyplot as plt

from npca_regression.kernel import fit_models, predict
from npca_regression.datasets import make_data_combined


PCA_METHOD = "kernel"
EVR_TARGET = 0.99
n_components = None

kernel_params = {
    "kernel_choice": "all",
    "poly_degrees": [2, 3, 4],
    "rbf_sigmas": [0.5, 1.0, 2.0],
    "constant": 1.0,
}

knn_params = {
    "k": 8,
    "lambda_knn": 0.0,
}

torch_params = {
    "torch_device": "cuda",
    "lr": 0.05,
    "steps": 300,
    "restarts": 1,
    "init_perturb": 0.1,
}

Z_train, Z_test = make_data_combined(
    n_train=450,
    n_test=100,
    n_features=2,
    seed=0,
    noise_level=0.15,
)

X_train = Z_train[:, :-1]
y_train = Z_train[:, -1]

X_test = Z_test[:, :-1]
y_test = Z_test[:, -1]

x1_min, x1_max = X_train[:, 0].min(), X_train[:, 0].max()
x2_min, x2_max = X_train[:, 1].min(), X_train[:, 1].max()

n_grid = 80
x1_vals = np.linspace(x1_min, x1_max, n_grid)
x2_vals = np.linspace(x2_min, x2_max, n_grid)
X1g, X2g = np.meshgrid(x1_vals, x2_vals)

Xq = np.column_stack([X1g.ravel(), X2g.ravel()])

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

print("\n--- Synthetic 3D Test Results ---")
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

    Yg = yq_pred.reshape(X1g.shape)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X1g, X2g, Yg, alpha=0.6)
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, s=20, alpha=0.8, label="train")
    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, s=20, alpha=0.7, label="test")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y / prediction")
    ax.set_title(f"NPCA Regression: {kernel}, {param_label}")
    ax.legend()

plt.show()