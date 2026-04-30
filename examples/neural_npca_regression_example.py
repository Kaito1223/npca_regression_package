import numpy as np
import matplotlib.pyplot as plt

from npca_regression.neural import (
    NeuralPCAConfig,
    NeuralNPCARegressionConfig,
    NeuralNPCARegressor,
    evaluate_neural_npca_regression,
)


def make_data(n_train=140, n_test=60, noise=0.08, seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.uniform(-np.pi, np.pi, size=(n_train, 1))
    y_train = np.sin(X_train[:, 0]) + noise * rng.normal(size=n_train)
    X_test = rng.uniform(-np.pi, np.pi, size=(n_test, 1))
    y_test = np.sin(X_test[:, 0]) + noise * rng.normal(size=n_test)
    return X_train, y_train, X_test, y_test


def main():
    X_train, y_train, X_test, y_test = make_data()
    Z_train = np.column_stack([X_train, y_train])

    config = NeuralNPCARegressionConfig(
        neural_pca=NeuralPCAConfig(
            n_components=2,
            block_size=32,
            step_size=0.05,
            max_iter=100,
            random_state=0,
        ),
        grid_size=61,
        refine=True,
    )

    df, _ = evaluate_neural_npca_regression(Z_train, X_test, y_test, config=config)
    print(df)

    model = NeuralNPCARegressor(config).fit(X_train, y_train)
    X_grid = np.linspace(X_train.min(), X_train.max(), 300).reshape(-1, 1)
    y_grid = model.predict(X_grid)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train[:, 0], y_train, s=20, alpha=0.7, label="train")
    plt.scatter(X_test[:, 0], y_test, s=20, alpha=0.7, label="test")
    plt.plot(X_grid[:, 0], y_grid, linewidth=2.5, label="neural NPCA prediction")
    plt.xlabel("x")
    plt.ylabel("y / prediction")
    plt.title("Neural NPCA regression")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
