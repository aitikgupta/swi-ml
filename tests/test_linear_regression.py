import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from cu_ml import (
    LinearRegressionGD,
    LassoRegressionGD,
    RidgeRegressionGD,
    ElasticNetRegressionGD,
)

X, Y = make_regression(n_samples=1000, n_features=5, random_state=42)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1 / 3, random_state=0
)


def test_linear_regression():
    num_iters = 10
    lr = 0.05

    model_np = LinearRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        backend="numpy",
        verbose="ERROR",
    )
    model_cp = LinearRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        backend="cupy",
        verbose="ERROR",
    )

    with pytest.raises(ValueError):
        model_np.fit([], [])

    with pytest.raises(NotImplementedError):
        wrong_initialiser_model = LinearRegressionGD(
            num_iterations=num_iters,
            learning_rate=lr,
            initialiser="ok",
            backend="numpy",
            verbose="ERROR",
        )
        wrong_initialiser_model.fit(X_train, Y_train)

    model_np.fit(X_train, Y_train)
    model_cp.fit(X_train, Y_train)

    Y_pred_np = model_np.predict(X_test)
    Y_pred_cp = model_cp.predict(X_test)

    assert len(model_np.history) == len(model_cp.history) == num_iters
    assert Y_pred_np.shape == Y_pred_cp.shape


def test_lasso_regression():
    num_iters = 10
    lr = 0.05

    # Needs l1_cost
    with pytest.raises(TypeError):
        _ = LassoRegressionGD(
            num_iterations=num_iters,
            learning_rate=lr,
            backend="numpy",
            verbose="ERROR",
        )

    model = LassoRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        l1_cost=100,
        backend="numpy",
        verbose="ERROR",
    )

    model.fit(X_train, Y_train)

    _ = model.predict(X_test)

    assert model.MSE_loss != None
    assert model.regularisation != None


def test_ridge_regression():
    num_iters = 10
    lr = 0.05

    # Needs l2_cost
    with pytest.raises(TypeError):
        _ = RidgeRegressionGD(
            num_iterations=num_iters,
            learning_rate=lr,
            backend="numpy",
            verbose="ERROR",
        )

    model = RidgeRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        l2_cost=100,
        backend="numpy",
        verbose="ERROR",
    )

    model.fit(X_train, Y_train)

    _ = model.predict(X_test)

    assert model.MSE_loss != None
    assert model.regularisation != None


def test_elasticnet_regression():
    num_iters = 10
    lr = 0.05

    # Needs multiplying factor, as well as l1_ratio
    with pytest.raises(TypeError):
        _ = ElasticNetRegressionGD(
            num_iterations=num_iters,
            learning_rate=lr,
            backend="numpy",
            verbose="ERROR",
        )

    # Test l1_ratio, initialiser set to zero to have same initial weights
    elastic_net_lasso_model = ElasticNetRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        multiply_factor=100,
        l1_ratio=1,
        backend="numpy",
        verbose="ERROR",
        initialiser="zeros",
    )
    elastic_net_ridge_model = ElasticNetRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        multiply_factor=100,
        l1_ratio=0,
        backend="numpy",
        verbose="ERROR",
        initialiser="zeros",
    )

    lasso_model = LassoRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        l1_cost=100,
        backend="numpy",
        verbose="ERROR",
        initialiser="zeros",
    )
    ridge_model = RidgeRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        l2_cost=100,
        backend="numpy",
        verbose="ERROR",
        initialiser="zeros",
    )

    elastic_net_lasso_model.fit(X_train, Y_train)
    elastic_net_lasso_pred = elastic_net_lasso_model.predict(X_test)

    elastic_net_ridge_model.fit(X_train, Y_train)
    elastic_net_ridge_pred = elastic_net_ridge_model.predict(X_test)

    lasso_model.fit(X_train, Y_train)
    lasso_pred = lasso_model.predict(X_test)

    ridge_model.fit(X_train, Y_train)
    ridge_pred = ridge_model.predict(X_test)

    assert (elastic_net_lasso_pred == lasso_pred).all()
    assert (elastic_net_ridge_pred == ridge_pred).all()
