import numpy as np
import pytest

from npca_regression.datasets import make_data_combined

from npca_regression.kernel import (
    rbf_kernel,
    poly_kernel,
    KernelConfig,
    KPCAResult,
    MomentPolynomialPCA,
    center_gram,
    kpca_fit,
    fit_models,
    predict,
)


def make_small_regression_data(seed=0):
    Z_train, Z_test = make_data_combined(
        n_train=40,
        n_test=12,
        n_features=1,
        seed=seed,
        noise_level=0.05,
    )

    X_test = Z_test[:, :-1]
    y_test = Z_test[:, -1]

    return Z_train, X_test, y_test


def test_feature_space_imports_work():
    assert callable(rbf_kernel)
    assert callable(poly_kernel)
    assert callable(center_gram)
    assert callable(kpca_fit)
    assert callable(fit_models)
    assert callable(predict)

    assert KPCAResult is not None
    assert MomentPolynomialPCA is not None


def test_rbf_kernel_shape_symmetry_and_diagonal():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(8, 3))

    K = rbf_kernel(Z, Z, sigma=1.0)

    assert K.shape == (8, 8)
    assert np.all(np.isfinite(K))
    assert np.allclose(K, K.T, atol=1e-10)
    assert np.allclose(np.diag(K), 1.0, atol=1e-10)


def test_poly_kernel_shape_symmetry():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(8, 3))

    K = poly_kernel(Z, Z, degree=2, c0=1.0)

    assert K.shape == (8, 8)
    assert np.all(np.isfinite(K))
    assert np.allclose(K, K.T, atol=1e-10)


def test_center_gram_returns_centered_matrix():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(10, 2))
    K = rbf_kernel(Z, Z, sigma=1.0)

    Kc, bar_k, bar_K = center_gram(K)

    assert Kc.shape == K.shape
    assert bar_k.shape == (10,)
    assert isinstance(bar_K, float) or np.isscalar(bar_K)

    assert np.all(np.isfinite(Kc))
    assert np.allclose(Kc.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(Kc.mean(axis=1), 0.0, atol=1e-10)


def test_kpca_fit_returns_expected_result_shapes():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(30, 2))

    cfg = KernelConfig(kind="rbf", params={"sigma": 1.0})
    model = kpca_fit(Z, cfg, evr_target=0.95)

    assert isinstance(model, KPCAResult)

    assert model.Z_train.shape == Z.shape
    assert model.K.shape == (30, 30)
    assert model.Kc.shape == (30, 30)
    assert model.Q.shape[0] == 30
    assert model.A_m.shape[0] == 30
    assert model.m == model.A_m.shape[1]

    assert np.all(np.isfinite(model.K))
    assert np.all(np.isfinite(model.Kc))
    assert np.all(np.isfinite(model.lambdas))


def test_fit_models_kernel_method_runs_small_case():
    Z_train, X_test, y_test = make_small_regression_data(seed=4)

    models = fit_models(
        Z_train,
        method="kernel",
        evr_target=0.95,
        m=None,
        scale_data=True,
        kernel_choice="rbf",
        rbf_sigmas=[1.0],
    )

    assert isinstance(models, list)
    assert len(models) == 1
    assert isinstance(models[0], KPCAResult)
    assert models[0].m >= 0


def test_predict_kernel_method_output_format():
    Z_train, X_test, y_test = make_small_regression_data(seed=5)

    models = fit_models(
        Z_train,
        method="kernel",
        evr_target=0.95,
        m=None,
        scale_data=True,
        kernel_choice="rbf",
        rbf_sigmas=[1.0],
    )

    df_results, results = predict(
        models,
        X_test,
        y_test=y_test,
        k=0,
        lambda_knn=0.0,
        batch_size=None,
        lr=0.05,
        steps=10,
    )

    expected_columns = {
        "kernel",
        "params",
        "m",
        "RMSE_yhat_vs_y",
        "root_mean_feature",
    }

    assert expected_columns.issubset(df_results.columns)
    assert len(df_results) == 1
    assert np.isfinite(df_results.loc[0, "RMSE_yhat_vs_y"])
    assert np.isfinite(df_results.loc[0, "root_mean_feature"])

    assert isinstance(results, list)
    assert len(results) == 1
    assert "pred" in results[0]
    assert results[0]["pred"].shape == y_test.shape
    assert np.all(np.isfinite(results[0]["pred"]))


def test_fit_models_moment_method_runs_small_case():
    Z_train, X_test, y_test = make_small_regression_data(seed=6)

    models = fit_models(
        Z_train,
        method="moment",
        evr_target=0.95,
        m=None,
        scale_data=True,
        degrees=[2],
        consts=[1.0],
    )

    assert isinstance(models, list)
    assert len(models) == 1
    assert isinstance(models[0], MomentPolynomialPCA)
    assert models[0].m >= 0


def test_predict_moment_method_output_format():
    Z_train, X_test, y_test = make_small_regression_data(seed=7)

    models = fit_models(
        Z_train,
        method="moment",
        evr_target=0.95,
        m=None,
        scale_data=True,
        degrees=[2],
        consts=[1.0],
    )

    df_results, results = predict(
        models,
        X_test,
        y_test=y_test,
        k=0,
        lambda_knn=0.0,
        batch_size=None,
        lr=0.05,
        steps=10,
    )

    expected_columns = {
        "kernel",
        "params",
        "m",
        "RMSE_yhat_vs_y",
        "root_mean_feature",
    }

    assert expected_columns.issubset(df_results.columns)
    assert len(df_results) == 1
    assert np.isfinite(df_results.loc[0, "RMSE_yhat_vs_y"])
    assert np.isfinite(df_results.loc[0, "root_mean_feature"])

    assert isinstance(results, list)
    assert len(results) == 1
    assert "pred" in results[0]
    assert results[0]["pred"].shape == y_test.shape
    assert np.all(np.isfinite(results[0]["pred"]))


def test_invalid_kernel_config_raises_error():
    rng = np.random.default_rng(8)
    Z = rng.normal(size=(10, 2))

    cfg = KernelConfig(kind="bad_kernel", params={})

    with pytest.raises(ValueError):
        kpca_fit(Z, cfg)