import numpy as np
import pytest

from npca_regression.neural import (
    NeuralPCAConfig,
    NeuralNPCARegressionConfig,
    NeuralNPCARegressor,
    evaluate_neural_npca_regression,
)


torch = pytest.importorskip("torch")


def make_data(n_train=80, n_test=20, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)

    X_train = rng.uniform(-np.pi, np.pi, size=(n_train, 1))
    y_train = np.sin(X_train[:, 0]) + noise * rng.normal(size=n_train)

    X_test = rng.uniform(-np.pi, np.pi, size=(n_test, 1))
    y_test = np.sin(X_test[:, 0]) + noise * rng.normal(size=n_test)

    return X_train, y_train, X_test, y_test


def test_torch_prediction_optimizer_runs():
    X_train, y_train, X_test, y_test = make_data(seed=0)

    config = NeuralNPCARegressionConfig(
        neural_pca=NeuralPCAConfig(
            n_components=2,
            block_size=20,
            step_size=0.05,
            max_iter=20,
            random_state=0,
        ),
        prediction_optimizer="torch",
        torch_learning_rate=0.05,
        torch_steps=20,
        torch_restarts=1,
        torch_device="cpu",
    )

    model = NeuralNPCARegressor(config)
    model.fit(X_train, y_train)

    y_pred, details = model.predict(X_test, return_details=True)

    assert y_pred.shape == y_test.shape
    assert np.all(np.isfinite(y_pred))
    assert "objective" in details
    assert np.all(np.isfinite(details["objective"]))
    assert details["prediction_optimizer"] == "torch"


def test_grid_then_torch_prediction_optimizer_runs():
    X_train, y_train, X_test, y_test = make_data(seed=1)

    config = NeuralNPCARegressionConfig(
        neural_pca=NeuralPCAConfig(
            n_components=2,
            block_size=20,
            step_size=0.05,
            max_iter=20,
            random_state=0,
        ),
        prediction_optimizer="grid_then_torch",
        grid_size=21,
        refine=False,
        torch_learning_rate=0.05,
        torch_steps=20,
        torch_restarts=1,
        torch_device="cpu",
    )

    model = NeuralNPCARegressor(config)
    model.fit(X_train, y_train)

    y_pred, details = model.predict(X_test, return_details=True)

    assert y_pred.shape == y_test.shape
    assert np.all(np.isfinite(y_pred))
    assert "objective" in details
    assert np.all(np.isfinite(details["objective"]))
    assert details["prediction_optimizer"] == "grid_then_torch"


def test_evaluate_neural_npca_regression_accepts_torch_optimizer():
    X_train, y_train, X_test, y_test = make_data(seed=2)
    Z_train = np.column_stack([X_train, y_train])

    df, results = evaluate_neural_npca_regression(
        Z_train,
        X_test,
        y_test=y_test,
        n_components=2,
        block_size=20,
        step_size=0.05,
        max_iter=20,
        prediction_optimizer="grid_then_torch",
        grid_size=21,
        refine=False,
        torch_learning_rate=0.05,
        torch_steps=20,
        torch_restarts=1,
        torch_device="cpu",
    )

    expected_columns = {
        "kernel",
        "params",
        "m",
        "RMSE_yhat_vs_y",
        "root_mean_feature",
    }

    assert expected_columns.issubset(df.columns)
    assert len(df) == 1
    assert np.isfinite(df.loc[0, "RMSE_yhat_vs_y"])
    assert np.isfinite(df.loc[0, "root_mean_feature"])

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["pred"].shape == y_test.shape
    assert np.all(np.isfinite(results[0]["pred"]))


def test_invalid_prediction_optimizer_raises():
    X_train, y_train, X_test, y_test = make_data(seed=3)

    config = NeuralNPCARegressionConfig(
        neural_pca=NeuralPCAConfig(
            n_components=2,
            block_size=20,
            step_size=0.05,
            max_iter=10,
            random_state=0,
        ),
        prediction_optimizer="bad_optimizer",
    )

    model = NeuralNPCARegressor(config)
    model.fit(X_train, y_train)

    with pytest.raises(ValueError):
        model.predict(X_test)