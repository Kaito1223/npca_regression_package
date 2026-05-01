def test_public_api_imports():
    import npca_regression.kernel

    assert hasattr(npca_regression.kernel, "fit_models")
    assert hasattr(npca_regression.kernel, "predict")
    assert hasattr(npca_regression.kernel, "KernelConfig")
