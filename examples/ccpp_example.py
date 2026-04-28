"""Example experiment using the packaged NPCA regression API.

This was moved out of the package core from the original NPCA_reg_op.py
__main__ block. The experiment logic is unchanged except for package imports.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from npca_regression import fit_models, predict

PCA_METHOD = 'kernel' # 'kernel' or 'moment'
EVR_TARGET = 0.99 # Optional float between 0 and 1 to select number of components by EVR threshold, e.g. 0.95. If None, use m-1 directly.
n_components = None
kernel_params = {
    'kernel_choice': 'all', 
    'poly_degrees': [2, 3, 4], 
    'rbf_sigmas': [0.5, 1.0, 2.0],
    'constant': 1.0
}                                                                         
moment_params = {
    'degrees': [1, 2, 3],
    'consts': [1.0]
}
knn_params = {
    'k': 8,
    'lambda_knn': 0.5
}
fit_args = kernel_params if PCA_METHOD == 'kernel' else moment_params

#Real dataset example (uncomment to use):
from ucimlrepo import fetch_ucirepo
ccpp = fetch_ucirepo(id=291)

X = ccpp.data.features.to_numpy(dtype=float)
y = ccpp.data.targets.to_numpy(dtype=float).ravel()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Z_train = np.column_stack([X_train, y_train])
Z_test = np.column_stack([X_test, y_test])

#Benchmark
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
models = {
    "Linear regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),
}
rows = []

for name, model in models.items():
    model.fit(X_train, y_train) # y_train is unscaled, so predictions will be unscaled
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    rows.append({
        "model": name,
        "MSE": mse,
        "RMSE": rmse,
    })

benchmark_results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

print("\nBenchmark results:")
print(benchmark_results.to_string(index=False))

#Experiment
fitted_models = fit_models(
    Z_train,
    method=PCA_METHOD,
    evr_target=EVR_TARGET,
    m=n_components,
    scale_data=True,
    **fit_args
)   

df_argmin_test, results_test = predict(
    fitted_models,
    X_test,
    y_test=y_test,
    batch_size=None,
    lr=0.05,
    steps=300,
    **knn_params
)


print(f"\n--- {PCA_METHOD.upper()} PCA Prediction on Test Set ---")
print(df_argmin_test)
# df_1 = pd.DataFrame(df_argmin_test)
# df_1.to_excel(f"{PCA_METHOD}_test_noisy_results.xlsx", index=False)
