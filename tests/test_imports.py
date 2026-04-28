def test_public_api_imports():
    import npca_regression

    assert hasattr(npca_regression, "fit_models")
    assert hasattr(npca_regression, "predict")
    assert hasattr(npca_regression, "KernelConfig")
