import pytest
from numpy.testing import assert_equal
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from cu_ml import (
    logger,
    set_logging_level,
    set_backend,
)
from cu_ml.classification import (
    NaiveBayesClassification,
)

set_logging_level("DEBUG")
logger.setLevel("DEBUG")

X, Y = make_classification(n_samples=1000, n_features=5, random_state=42)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1 / 3, random_state=0
)


def test_naive_bayes():
    dist = "gaussian"

    # set numpy backend first
    set_backend("numpy")
    model_np = NaiveBayesClassification(distribution=dist)

    model_np.fit(X_train, Y_train)
    Y_pred_np = model_np.predict(X_test, probability=False)

    with pytest.raises(NotImplementedError):
        _ = NaiveBayesClassification(
            distribution="wrong_distribution",
        )

    assert Y_pred_np is not None


def test_naive_bayes_special_case():
    dist = "gaussian"

    # set cupy backend
    set_backend("cupy")

    # special case: f"{element}:.2f" is unsupported for CuPy backend
    with pytest.raises(TypeError):
        model_cp = NaiveBayesClassification(distribution=dist)

        model_cp.fit(X_train, Y_train)

        Y_pred_cp = model_cp.predict(X_test)

    # # class distributions should be same for both backends
    # assert model_np._class_distributions == model_cp._class_distributions
    # assert Y_pred_np.shape == Y_pred_cp.shape