import pytest
from numpy.testing import assert_equal
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from swi_ml import (
    logger,
    set_automatic_fallback,
    set_backend,
    set_logging_level,
)
from swi_ml.classification import (
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


def test_naive_bayes_probability():
    dist = "gaussian"

    model_np = NaiveBayesClassification(distribution=dist)

    model_np.fit(X_train, Y_train)
    Y_pred = model_np.predict(X_test, probability=False)
    Y_pred_proba = model_np.predict(X_test, probability=True)

    assert Y_pred.shape[0] == Y_pred_proba.shape[0]
    assert Y_pred_proba.shape[1] == 2


@pytest.mark.xfail
def test_naive_bayes_special_case():
    dist = "gaussian"

    # set cupy backend
    set_backend("cupy")

    # special case: f"{element}:.2f" is unsupported for CuPy backend
    # fixed in https://github.com/cupy/cupy/issues/4532
    with pytest.raises(TypeError):
        model_cp = NaiveBayesClassification(distribution=dist)

        model_cp.fit(X_train, Y_train)

        Y_pred_cp = model_cp.predict(X_test)

        set_backend("numpy")
        model_np = NaiveBayesClassification(distribution=dist)

        model_np.fit(X_train, Y_train)
        Y_pred_np = model_np.predict(X_test, probability=False)

        assert Y_pred_np.shape == Y_pred_cp.shape
