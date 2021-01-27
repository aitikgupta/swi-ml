import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from swi_ml import (
    logger,
    set_automatic_fallback,
    set_backend,
    set_logging_level,
)
from swi_ml.classification import (
    LogisticRegressionGD,
)

set_logging_level("DEBUG")
logger.setLevel("DEBUG")

X, Y = make_classification(n_samples=1000, n_features=5, random_state=42)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1 / 3, random_state=0
)


def test_logistic_regression():
    num_iters = 10
    lr = 0.05

    # set numpy backend first
    set_backend("numpy")
    model_np = LogisticRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
    )

    with pytest.raises(ValueError):
        model_np.fit([], [])

    model_np.fit(X_train, Y_train)
    Y_pred_np = model_np.predict(X_test)

    with pytest.raises(NotImplementedError):
        wrong_initialiser_model = LogisticRegressionGD(
            num_iterations=num_iters,
            learning_rate=lr,
            initialiser="wrong_initialiser",
        )
        wrong_initialiser_model.fit(X_train, Y_train)

    # set cupy backend
    set_backend("cupy")
    model_cp = LogisticRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
    )

    model_cp.fit(X_train, Y_train)

    Y_pred_cp = model_cp.predict(X_test)

    assert len(model_np.history) == len(model_cp.history) == num_iters
    assert Y_pred_np.shape == Y_pred_cp.shape

    # switch backend back to numpy
    set_backend("numpy")

    model = LogisticRegressionGD(
        num_iterations=num_iters,
        learning_rate=lr,
        multiply_factor=100,
        l1_ratio=0.5,
        normalize=True,
    )

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test, probability=True)

    assert Y_pred is not None
