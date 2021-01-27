import pytest
from numpy.testing import assert_equal
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from swi_ml import (
    logger,
    set_automatic_fallback,
    set_backend,
    set_logging_level,
)
from swi_ml.regression import (
    LinearRegressionGD,
    LassoRegressionGD,
    RidgeRegressionGD,
    ElasticNetRegressionGD,
    PolynomialRegressionGD,
)

set_logging_level("DEBUG")
logger.setLevel("DEBUG")

X, Y = make_regression(n_samples=1000, n_features=5, random_state=42)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1 / 3, random_state=0
)


def test_linear_regression():
    # only test which tests both NumPy and CuPy backends
    num_iters = 10
    lr = 0.05

    # set numpy backend first
    set_backend("numpy")
    model_np = LinearRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
    )

    with pytest.raises(ValueError):
        model_np.fit([], [])

    model_np.fit(X_train, Y_train)
    Y_pred_np = model_np.predict(X_test)

    with pytest.raises(NotImplementedError):
        wrong_initialiser_model = LinearRegressionGD(
            num_iterations=num_iters,
            learning_rate=lr,
            initialiser="wrong_initialiser",
        )
        wrong_initialiser_model.fit(X_train, Y_train)

    # set cupy backend
    set_backend("cupy")
    model_cp = LinearRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
    )

    model_cp.fit(X_train, Y_train)

    Y_pred_cp = model_cp.predict(X_test)

    assert len(model_np.history) == len(model_cp.history) == num_iters
    assert Y_pred_np.shape == Y_pred_cp.shape

    # switch backend back to numpy
    set_backend("numpy")


def test_lasso_regression():
    num_iters = 10
    lr = 0.05

    # needs l1_cost
    with pytest.raises(TypeError):
        _ = LassoRegressionGD(
            num_iterations=num_iters,
            learning_rate=lr,
        )

    model = LassoRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        l1_cost=100,
    )

    model.fit(X_train, Y_train)

    _ = model.predict(X_test)

    assert model.MSE_loss is not None
    assert model.regularisation is not None


def test_ridge_regression():
    num_iters = 10
    lr = 0.05

    # needs l2_cost
    with pytest.raises(TypeError):
        _ = RidgeRegressionGD(
            num_iterations=num_iters,
            learning_rate=lr,
        )

    model = RidgeRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        l2_cost=100,
    )

    model.fit(X_train, Y_train)

    _ = model.predict(X_test)

    assert model.MSE_loss is not None
    assert model.regularisation is not None


def test_elasticnet_regression():
    num_iters = 10
    lr = 0.05

    # needs multiplying factor, as well as l1_ratio
    with pytest.raises(TypeError):
        _ = ElasticNetRegressionGD(
            num_iterations=num_iters,
            learning_rate=lr,
        )

    # test l1_ratio, initialiser set to zero to have same initial weights
    elastic_net_lasso_model = ElasticNetRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        multiply_factor=100,
        l1_ratio=1,
        initialiser="zeros",
        normalize=False,
    )
    elastic_net_ridge_model = ElasticNetRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        multiply_factor=100,
        l1_ratio=0,
        initialiser="zeros",
        normalize=False,
    )

    lasso_model = LassoRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        l1_cost=100,
        initialiser="zeros",
        normalize=False,
    )
    ridge_model = RidgeRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        l2_cost=100,
        initialiser="zeros",
        normalize=False,
    )

    elastic_net_lasso_model.fit(X_train, Y_train)
    elastic_net_lasso_pred = elastic_net_lasso_model.predict(X_test)

    elastic_net_ridge_model.fit(X_train, Y_train)
    elastic_net_ridge_pred = elastic_net_ridge_model.predict(X_test)

    lasso_model.fit(X_train, Y_train)
    lasso_pred = lasso_model.predict(X_test)

    ridge_model.fit(X_train, Y_train)
    ridge_pred = ridge_model.predict(X_test)

    assert_equal(elastic_net_lasso_pred, lasso_pred)
    assert_equal(elastic_net_ridge_pred, ridge_pred)


def test_polynomial_regression():
    num_iters = 10
    lr = 0.05

    # needs degree
    with pytest.raises(TypeError):
        _ = PolynomialRegressionGD(
            num_iterations=num_iters,
            learning_rate=lr,
        )

    # test degree
    linear_regression_model = LinearRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        initialiser="zeros",
        normalize=False,
    )
    polynomial_regression_model = PolynomialRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        initialiser="zeros",
        normalize=False,
        degree=1,
    )

    linear_regression_model.fit(X_train, Y_train)
    linear_regression_pred = linear_regression_model.predict(X_test)

    polynomial_regression_model.fit(X_train, Y_train)
    polynomial_regression_pred = polynomial_regression_model.predict(X_test)

    # assert_equal(polynomial_regression_pred, linear_regression_pred)

    assert polynomial_regression_model.MSE_loss is not None
    assert polynomial_regression_model.degree is not None
