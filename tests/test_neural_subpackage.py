import numpy as np

from npca_regression.neural import (
    PCAWhitening,
    NeuralPCA,
    NeuralPCAConfig,
    NeuralNPCARegressor,
    NeuralNPCARegressionConfig,
    evaluate_neural_npca_regression,
)
from npca_regression.neural.activations import estimate_activation_stats, neural_activation
from npca_regression.neural.linalg import orthogonalize_rows, row_orthogonality_error


def make_regression_data(n=100, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi, np.pi, size=(n, 1))
    y = np.sin(X[:, 0]) + noise * rng.normal(size=n)
    return X, y


def test_whitening_shapes_and_centering():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    whitener = PCAWhitening(n_components=3)
    Xw = whitener.fit_transform(X)
    assert Xw.shape == (120, 3)
    assert np.all(np.isfinite(Xw))
    assert np.allclose(Xw.mean(axis=0), 0.0, atol=1e-6)


def test_orthogonalize_rows():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(3, 3))
    Q = orthogonalize_rows(W)
    assert Q.shape == (3, 3)
    assert row_orthogonality_error(Q) < 1e-8


def test_activation_is_finite_and_shape_preserving():
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(50, 2))
    stats = estimate_activation_stats(Y)
    Phi = neural_activation(
        Y,
        second_moment=stats.second_moment,
        kurtosis_sign=stats.kurtosis_sign,
    )
    assert Phi.shape == Y.shape
    assert np.all(np.isfinite(Phi))


def test_neural_pca_fit_transform_reconstruction_error():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 4))
    model = NeuralPCA(
        NeuralPCAConfig(
            n_components=2,
            block_size=20,
            step_size=0.05,
            max_iter=15,
            random_state=0,
        )
    ).fit(X)
    S = model.transform(X)
    err = model.reconstruction_error(X)
    assert S.shape == (80, 2)
    assert err.shape == (80,)
    assert np.all(np.isfinite(S))
    assert np.all(np.isfinite(err))


def test_neural_npca_regressor_predict_shape():
    X, y = make_regression_data(n=80, seed=4)
    model = NeuralNPCARegressor(
        NeuralNPCARegressionConfig(
            neural_pca=NeuralPCAConfig(
                n_components=2,
                block_size=20,
                step_size=0.05,
                max_iter=15,
                random_state=0,
            ),
            grid_size=25,
            refine=False,
        )
    ).fit(X, y)
    pred, details = model.predict(X[:8], return_details=True)
    assert pred.shape == (8,)
    assert details["objective"].shape == (8,)
    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(details["objective"]))


def test_evaluate_neural_npca_regression_kpca_style_output():
    X, y = make_regression_data(n=80, seed=5)
    Z = np.column_stack([X, y])
    df, results = evaluate_neural_npca_regression(
        Z,
        X[:10],
        y_test=y[:10],
        n_components=2,
        block_size=20,
        step_size=0.05,
        max_iter=15,
        grid_size=25,
        refine=False,
        random_state=0,
    )
    required = {"kernel", "params", "m", "RMSE_yhat_vs_y", "root_mean_feature"}
    assert required.issubset(df.columns)
    assert len(df) == 1
    assert np.isfinite(df.loc[0, "RMSE_yhat_vs_y"])
    assert np.isfinite(df.loc[0, "root_mean_feature"])
    assert len(results) == 1
    assert results[0]["pred"].shape == (10,)
