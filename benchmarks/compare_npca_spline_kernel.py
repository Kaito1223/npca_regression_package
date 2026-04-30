from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from npca_regression.datasets import make_data_separated
from npca_regression.baselines import (
    KernelRegression,
    TPSRegressor,
    zscore,
    pairwise_sq_dists_numpy,
    cdist,
)

from npca_regression.feature_space import evaluate_kernels_with_torch_argmin

from npca_regression.neural import (
    NeuralPCAConfig,
    NeuralNPCARegressionConfig,
    evaluate_neural_npca_regression,
)


Array = np.ndarray


@dataclass(frozen=True)
class DatasetConfig:
    n_train: int = 400
    n_test: int = 100
    n_features: int = 1
    noise_level: float = 0.5
    distribution: str = "uniform"
    dist_params: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class BenchmarkConfig:
    n_datasets: int = 5
    output_dir: str = "benchmark_outputs"
    save_csv: bool = True
    show_detailed_table: bool = True


@dataclass
class ModelResult:
    model: str
    rmse: float
    params: str
    runtime_seconds: float
    root_mean_feature: Optional[float] = None
    y_pred: Optional[Array] = None


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "feature_npca": {
        "kernel_choice": "all",
        "poly_degrees": [2, 3],
        "rbf_sigmas": [0.5, 1.0, 2.0],
        "evr_target": 0.99,
        "lr": 0.05,
        "steps": 200,
    },
    "neural_npca": {
        "n_components": 2,
        "block_size": 32,
        "step_size": 0.05,
        "max_iter": 300,
        "tol": 1e-6,
        "random_state": 0,
        "grid_size": 81,
        "refine": True,
    },
    "kernel_regression": {
        "kernel": "rbf",
        "standardize": True,
        "bandwidth_grid_size": 20,
        "bandwidth_low_factor": 0.1,
        "bandwidth_high_factor": 10.0,
    },
    "spline": {
        "lambda_grid": [0.001, 0.003, 0.01, 0.03, 0.1],
        "m": 2,
        "standardize": True,
        "validation_fraction": 0.25,
        "random_state": 0,
    },
}


DEFAULT_MODELS = [
    "feature_npca",
    "neural_npca",
    "kernel_regression",
    "spline",
]


def rmse(y_true: Array, y_pred: Array) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compact_params(params: Any, max_len: int = 120) -> str:
    text = str(params)
    text = " ".join(text.split())
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def make_dataset(seed: int, config: DatasetConfig):
    X_train, y_train, X_test, y_test = make_data_separated(
        n_train=config.n_train,
        n_test=config.n_test,
        n_features=config.n_features,
        seed=seed,
        noise_level=config.noise_level,
        distribution=config.distribution,
        dist_params=config.dist_params,
    )
    Z_train = np.column_stack([X_train, y_train])
    Z_test = np.column_stack([X_test, y_test])
    return X_train, y_train, X_test, y_test, Z_train, Z_test


def run_feature_npca(
    X_train: Array,
    y_train: Array,
    X_test: Array,
    y_test: Array,
    Z_train: Array,
    **params: Any,
) -> ModelResult:
    del X_train, y_train
    start = perf_counter()

    df_results, raw_results = evaluate_kernels_with_torch_argmin(
        Z_train,
        X_test,
        y_test,
        **params,
    )

    runtime = perf_counter() - start

    if df_results.empty:
        return ModelResult(
            model="feature_npca",
            rmse=np.inf,
            params="no valid model",
            runtime_seconds=runtime,
            root_mean_feature=np.nan,
            y_pred=None,
        )

    best_idx = int(df_results["RMSE_yhat_vs_y"].idxmin())
    best = df_results.loc[best_idx]

    y_pred = None
    if isinstance(raw_results, list):
        # Match by row position when possible. If idxmin is a DataFrame index, not
        # positional, fall back to searching by kernel/params label.
        if 0 <= best_idx < len(raw_results):
            y_pred = raw_results[best_idx].get("pred")
        else:
            for item in raw_results:
                if item.get("kernel") == best.get("kernel") and item.get("params_label") == best.get("params"):
                    y_pred = item.get("pred")
                    break

    return ModelResult(
        model="feature_npca",
        rmse=float(best["RMSE_yhat_vs_y"]),
        params=f"kernel={best.get('kernel')}, params={best.get('params')}, m={best.get('m')}",
        runtime_seconds=runtime,
        root_mean_feature=float(best.get("root_mean_feature", np.nan)),
        y_pred=y_pred,
    )


def _build_neural_config(params: Dict[str, Any]) -> NeuralNPCARegressionConfig:
    neural_keys = {
        "n_components",
        "whitening",
        "epsilon",
        "block_size",
        "step_size",
        "max_iter",
        "tol",
        "random_state",
        "verbose",
    }
    regression_keys = {
        "y_bounds",
        "y_margin_fraction",
        "grid_size",
        "refine",
    }

    neural_kwargs = {key: params[key] for key in neural_keys if key in params}
    regression_kwargs = {key: params[key] for key in regression_keys if key in params}

    return NeuralNPCARegressionConfig(
        neural_pca=NeuralPCAConfig(**neural_kwargs),
        **regression_kwargs,
    )


def run_neural_npca(
    X_train: Array,
    y_train: Array,
    X_test: Array,
    y_test: Array,
    Z_train: Array,
    **params: Any,
) -> ModelResult:
    del X_train, y_train
    start = perf_counter()

    try:
        evaluated, raw_results = evaluate_neural_npca_regression(
            Z_train,
            X_test,
            y_test=y_test,
            **params,
        )
    except TypeError:
        # Compatibility with the earlier adapter that accepted config=... only.
        config = _build_neural_config(params)
        evaluated, raw_results = evaluate_neural_npca_regression(
            Z_train,
            X_test,
            y_test=y_test,
            config=config,
        )

    runtime = perf_counter() - start

    if isinstance(evaluated, pd.DataFrame):
        if evaluated.empty:
            return ModelResult(
                model="neural_npca",
                rmse=np.inf,
                params="no valid model",
                runtime_seconds=runtime,
                root_mean_feature=np.nan,
                y_pred=None,
            )
        best = evaluated.iloc[int(evaluated["RMSE_yhat_vs_y"].astype(float).idxmin())]
        y_pred = None
        if isinstance(raw_results, list) and raw_results:
            y_pred = raw_results[0].get("pred")
        elif isinstance(raw_results, dict):
            y_pred = raw_results.get("pred")
        return ModelResult(
            model="neural_npca",
            rmse=float(best["RMSE_yhat_vs_y"]),
            params=compact_params(best.get("params", params)),
            runtime_seconds=runtime,
            root_mean_feature=float(best.get("root_mean_feature", np.nan)),
            y_pred=y_pred,
        )

    # Compatibility with row-dict adapter.
    row = dict(evaluated)
    y_pred = None
    if isinstance(raw_results, dict):
        y_pred = raw_results.get("pred")
    root_mean_feature = np.nan
    if "root_mean_feature" in row:
        root_mean_feature = float(row["root_mean_feature"])
    elif "mean_objective" in row:
        root_mean_feature = float(np.sqrt(max(float(row["mean_objective"]), 0.0)))

    return ModelResult(
        model="neural_npca",
        rmse=float(row["RMSE_yhat_vs_y"]),
        params=compact_params(row.get("params", params)),
        runtime_seconds=runtime,
        root_mean_feature=root_mean_feature,
        y_pred=y_pred,
    )


def run_kernel_regression(
    X_train: Array,
    y_train: Array,
    X_test: Array,
    y_test: Array,
    Z_train: Array,
    **params: Any,
) -> ModelResult:
    del Z_train
    start = perf_counter()

    grid_size = int(params.pop("bandwidth_grid_size", 20))
    low_factor = float(params.pop("bandwidth_low_factor", 0.1))
    high_factor = float(params.pop("bandwidth_high_factor", 10.0))

    model = KernelRegression(**params)
    model.fit(X_train, y_train)

    Xz, _, _ = zscore(X_train)
    if cdist is not None:
        dists = cdist(Xz, Xz)
    else:
        dists = np.sqrt(pairwise_sq_dists_numpy(Xz, Xz))

    upper = dists[np.triu_indices(len(X_train), 1)]
    med = float(np.median(upper)) if upper.size else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0

    h_grid = np.logspace(
        np.log10(med * low_factor),
        np.log10(med * high_factor),
        grid_size,
    )
    best_h = model.select_bandwidth(h_grid)
    y_pred = model.predict(X_test)

    runtime = perf_counter() - start
    final_params = dict(params)
    final_params["selected_h"] = float(best_h)

    return ModelResult(
        model="kernel_regression",
        rmse=rmse(y_test, y_pred),
        params=compact_params(final_params),
        runtime_seconds=runtime,
        root_mean_feature=None,
        y_pred=y_pred,
    )


def train_validation_split(
    X: Array,
    y: Array,
    validation_fraction: float,
    random_state: int,
):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_val = max(1, int(round(validation_fraction * n)))
    n_val = min(n_val, n - 1)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def run_spline(
    X_train: Array,
    y_train: Array,
    X_test: Array,
    y_test: Array,
    Z_train: Array,
    **params: Any,
) -> ModelResult:
    del Z_train
    start = perf_counter()

    lambda_grid = params.pop("lambda_grid", None)
    validation_fraction = float(params.pop("validation_fraction", 0.25))
    random_state = int(params.pop("random_state", 0))

    if lambda_grid is None:
        model = TPSRegressor(**params)
        model.fit(X_train, y_train)
        selected_lambda = getattr(model, "lambda_", None)
    else:
        X_fit, y_fit, X_val, y_val = train_validation_split(
            X_train,
            y_train,
            validation_fraction=validation_fraction,
            random_state=random_state,
        )
        best_lambda = None
        best_score = np.inf
        for lambda_value in lambda_grid:
            candidate_params = dict(params)
            candidate_params["lambda_"] = float(lambda_value)
            candidate = TPSRegressor(**candidate_params)
            candidate.fit(X_fit, y_fit)
            val_pred = candidate.predict(X_val)
            val_rmse = rmse(y_val, val_pred)
            if val_rmse < best_score:
                best_score = val_rmse
                best_lambda = float(lambda_value)

        selected_lambda = best_lambda
        final_params = dict(params)
        final_params["lambda_"] = float(selected_lambda)
        model = TPSRegressor(**final_params)
        model.fit(X_train, y_train)
        params = final_params

    y_pred = model.predict(X_test)
    runtime = perf_counter() - start

    final_params = dict(params)
    if selected_lambda is not None:
        final_params["selected_lambda"] = float(selected_lambda)

    return ModelResult(
        model="spline",
        rmse=rmse(y_test, y_pred),
        params=compact_params(final_params),
        runtime_seconds=runtime,
        root_mean_feature=None,
        y_pred=y_pred,
    )


RUNNERS: Dict[str, Callable[..., ModelResult]] = {
    "feature_npca": run_feature_npca,
    "neural_npca": run_neural_npca,
    "kernel_regression": run_kernel_regression,
    "spline": run_spline,
}


def evaluate_models(
    models_to_run: Iterable[str] = DEFAULT_MODELS,
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    dataset_config: DatasetConfig = DatasetConfig(),
    benchmark_config: BenchmarkConfig = BenchmarkConfig(),
) -> pd.DataFrame:
    model_configs = MODEL_CONFIGS if model_configs is None else model_configs
    rows = []

    for seed in range(int(benchmark_config.n_datasets)):
        dataset_id = seed + 1
        print(f"\n=== Dataset {dataset_id}/{benchmark_config.n_datasets} ===")

        X_train, y_train, X_test, y_test, Z_train, _ = make_dataset(seed, dataset_config)
        dataset_rows = []

        for model_name in models_to_run:
            if model_name not in RUNNERS:
                raise ValueError(f"Unknown model name: {model_name!r}. Valid names: {sorted(RUNNERS)}")

            print(f"Running {model_name}...")
            params = dict(model_configs.get(model_name, {}))
            result = RUNNERS[model_name](
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                Z_train=Z_train,
                **params,
            )

            row = {
                "dataset": dataset_id,
                "model": result.model,
                "rmse": result.rmse,
                "root_mean_feature": result.root_mean_feature,
                "runtime_seconds": result.runtime_seconds,
                "params": result.params,
            }
            rows.append(row)
            dataset_rows.append(row)

        dataset_table = pd.DataFrame(dataset_rows).sort_values("rmse").reset_index(drop=True)
        dataset_table.insert(0, "rank", np.arange(1, len(dataset_table) + 1))
        print(dataset_table[["rank", "model", "rmse", "runtime_seconds", "params"]].to_string(index=False))

    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby("model", as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            median_rmse=("rmse", "median"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .sort_values("mean_rmse")
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))

    best_counts = (
        results.loc[results.groupby("dataset")["rmse"].idxmin(), "model"]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="best_count")
    )
    summary = summary.merge(best_counts, on="model", how="left")
    summary["best_count"] = summary["best_count"].fillna(0).astype(int)
    return summary


def save_outputs(results: pd.DataFrame, summary: pd.DataFrame, output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    results.to_csv(path / "detailed_results.csv", index=False)
    summary.to_csv(path / "summary_ranking.csv", index=False)
    print(f"\nSaved detailed results to: {path / 'detailed_results.csv'}")
    print(f"Saved summary ranking to: {path / 'summary_ranking.csv'}")


def main() -> None:
    dataset_config = DatasetConfig(
        n_train=400,
        n_test=100,
        n_features=1,
        noise_level=0.5,
    )
    benchmark_config = BenchmarkConfig(
        n_datasets=5,
        output_dir="benchmark_outputs",
        save_csv=True,
        show_detailed_table=True,
    )

    models_to_run = DEFAULT_MODELS

    results = evaluate_models(
        models_to_run=models_to_run,
        model_configs=MODEL_CONFIGS,
        dataset_config=dataset_config,
        benchmark_config=benchmark_config,
    )

    summary = summarize_results(results)

    print("\n=== Final ranking by average RMSE ===")
    print(summary.to_string(index=False))

    if benchmark_config.show_detailed_table:
        print("\n=== Detailed results ===")
        detailed = results.sort_values(["dataset", "rmse"]).reset_index(drop=True)
        print(detailed.to_string(index=False))

    if benchmark_config.save_csv:
        save_outputs(results, summary, benchmark_config.output_dir)


if __name__ == "__main__":
    main()
