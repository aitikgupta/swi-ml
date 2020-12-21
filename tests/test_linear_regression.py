import pytest
import numpy as cp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from cu_ml import LinearRegressionGD

X, Y = make_regression(n_samples=100, n_features=5, random_state=42)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1 / 3, random_state=0
)

X_train = cp.asarray(X_train)
X_test = cp.asarray(X_test)
Y_train = cp.asarray(Y_train)
Y_test = cp.asarray(Y_test)


def test_linear_regression():
    model = LinearRegressionGD(num_iterations=10, learning_rate=0.05)

    with pytest.raises(AttributeError):
        model.fit([], [])

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    model.plot_loss()
