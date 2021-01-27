import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from swi_ml import (
    logger,
    set_automatic_fallback,
    set_backend,
    set_logging_level,
)
from swi_ml.svm import (
    SVM,
)

set_logging_level("DEBUG")
logger.setLevel("DEBUG")

X, Y = make_blobs(
    n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
)
Y = np.where(Y == 0, -1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1 / 3, random_state=0
)


def test_svm():
    num_iters = 10
    lr = 0.05

    # set numpy backend first
    set_backend("numpy")
    model_np = SVM(
        num_iterations=num_iters,
        learning_rate=lr,
    )

    with pytest.raises(ValueError):
        model_np.fit([], [])

    # regularisation ratio must be between 0 and 1
    with pytest.raises(ValueError):
        wrong_ratio_model = SVM(
            num_iterations=num_iters,
            learning_rate=lr,
            regularisation_ratio=1.5,
        )
        wrong_ratio_model.fit(X_train, Y_train)

    model_np.fit(X_train, Y_train)
    Y_pred_np = model_np.predict(X_test)

    with pytest.raises(NotImplementedError):
        wrong_initialiser_model = SVM(
            num_iterations=num_iters,
            learning_rate=lr,
            initialiser="wrong_initialiser",
        )
        wrong_initialiser_model.fit(X_train, Y_train)

    # set cupy backend
    set_backend("cupy")
    model_cp = SVM(
        num_iterations=num_iters,
        learning_rate=lr,
    )

    model_cp.fit(X_train, Y_train)

    Y_pred_cp = model_cp.predict(X_test)

    assert len(model_np.history) == len(model_cp.history) == num_iters
    assert Y_pred_np.shape == Y_pred_cp.shape

    # switch backend back to numpy
    set_backend("numpy")

    model = SVM(
        num_iterations=num_iters,
        learning_rate=lr,
        normalize=True,
    )

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test, hinge=True)

    assert Y_pred is not None
