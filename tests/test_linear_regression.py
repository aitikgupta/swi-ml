import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from cu_ml import LinearRegressionGD

X, Y = make_regression(n_samples=100, n_features=5, random_state=42)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1 / 3, random_state=0
)


def test_linear_regression():
    num_iters = 10
    model_np = LinearRegressionGD(
        num_iterations=num_iters, learning_rate=0.05, backend="numpy"
    )
    model_cp = LinearRegressionGD(
        num_iterations=num_iters, learning_rate=0.05, backend="cupy"
    )

    with pytest.raises(ValueError):
        model_np.fit([], [])

    model_np.fit(X_train, Y_train)
    model_cp.fit(X_train, Y_train)

    Y_pred_np = model_np.predict(X_test)
    Y_pred_cp = model_cp.predict(X_test)

    assert len(model_np.history) == len(model_cp.history) == num_iters
    assert Y_pred_np.shape == Y_pred_cp.shape == (34,)
