from __future__ import annotations

import logging
import math
import time

import matplotlib.pyplot as plt

from cu_ml import logger as _global_logger
from cu_ml.backend import _Backend
from cu_ml import manipulations

logger = logging.getLogger(__name__)


class _BaseRegularisation(_Backend):
    """
    Base Class for Regularisation, L1 and L2 regularisations inherit from this
    NOTE: Can be used directly as a L1_L2 (ElasticNet) Regularisation
    """

    def __init__(self, multiply_factor: float, l1_ratio: float) -> None:
        self.multiply_factor = (
            multiply_factor if multiply_factor is not None else 1
        )
        self.l1_ratio = l1_ratio if l1_ratio is not None else 1
        self.backend = super().get_backend()

    def add_cost_regularisation(self, W):
        l1_regularisation = self.l1_ratio * self.backend.linalg.norm(W)
        l2_regularisation = (1 - self.l1_ratio) * W.T.dot(W)
        return self.multiply_factor * (l1_regularisation + l2_regularisation)

    def add_gradient_regularisation(self, W):
        l1_regularisation = self.l1_ratio * self.backend.sign(W)
        l2_regularisation = (1 - self.l1_ratio) * 2 * W
        return self.multiply_factor * (l1_regularisation + l2_regularisation)


class L1Regularisation(_BaseRegularisation):
    """
    Lasso Regression Regularisation
    """

    def __init__(self, l1_cost: float) -> None:
        multiply_factor = l1_cost
        l1_ratio = 1
        super().__init__(multiply_factor, l1_ratio)


class L2Regularisation(_BaseRegularisation):
    """
    Ridge Regression Regularisation
    """

    def __init__(self, l2_cost: float) -> None:
        multiply_factor = l2_cost
        l1_ratio = 0
        super().__init__(multiply_factor, l1_ratio)


class L1_L2Regularisation(_BaseRegularisation):
    """
    ElasticNet Regression Regularisation
    """

    def __init__(self, multiply_factor: float, l1_ratio: float) -> None:
        super().__init__(multiply_factor, l1_ratio)


class _BaseRegression(_Backend):
    """
    Base Class for Regression, all regression classes inherit from this
    """

    def __init__(
        self,
        num_iterations: int,
        learning_rate: float,
        normalize=True,
        regularisation=None,
        initialiser="uniform",
        verbose=None,
    ) -> None:
        if verbose is not None:
            logger.setLevel(verbose)
        else:
            logger.setLevel(_global_logger.getEffectiveLevel())
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.normalize = normalize
        self.initialiser = initialiser
        self.history = []
        self.regularisation = regularisation
        self.backend = super().get_backend()

    def _initialise_uniform_weights(self, shape: tuple) -> None:
        self.num_samples, self.num_features = shape
        limit = 1 / math.sqrt(self.num_features)
        self.W = self.backend.asarray(
            self.backend.random.uniform(-limit, limit, (self.num_features,))
        )
        self.b = self.backend.zeros(
            1,
        )

    def _initialise_zeros_weights(self, shape: tuple) -> None:
        self.num_samples, self.num_features = shape
        self.W = self.backend.asarray(
            self.backend.zeros(
                self.num_features,
            )
        )
        self.b = self.backend.zeros(
            1,
        )

    def _update_history(self) -> None:
        self.history.append(self.curr_loss)

    def _update_weights(self):
        self.W = self.W - self.learning_rate * self._dW
        self.b = self.b - self.learning_rate * self._db
        return self

    def MSE_loss(self, Y_true, Y_pred):
        """
        Given ground truth and predicted values, it returns the Mean Square Error
        """
        return self.backend.mean(
            0.5 * (Y_true - Y_pred) ** 2
        ) + self.regularisation.add_cost_regularisation(self.W)

    def initialise_weights(self, X) -> None:
        """
        Initialises weights with correct dimensions
        """
        if self.initialiser == "uniform":
            self._initialise_uniform_weights(X.shape)
        elif self.initialiser == "zeros":
            self._initialise_zeros_weights(X.shape)
        else:
            raise NotImplementedError(
                "Only 'uniform' and 'zeros' initialisers are supported"
            )

    def _fit_preprocess(self, data, labels):
        # offload to-be-used-once tasks to CPU
        if self.normalize:
            data = manipulations.normalize(data)
        # cast to array, CuPy backend will load the arrays on GPU
        X = self.backend.asarray(data)
        Y = self.backend.asarray(labels)
        return X, Y

    def fit(self, data, labels):
        """
        Given data and labels, it runs the actual training logic
        """
        logger.debug(f"Current Backend: {self.backend}")
        X, Y = self._fit_preprocess(data, labels)

        self.initialise_weights(X)

        start = time.time()
        for it in range(self.num_iterations):
            Y_pred = self._predict(X)
            diff = self.backend.subtract(Y, Y_pred)

            self.curr_loss = self.MSE_loss(Y, Y_pred)
            logger.info(
                f"MSE ({it+1}/{self.num_iterations}): {self.curr_loss}"
            )
            # regularisation magnitude is 0 for vanilla linear regression
            self._dW = (
                -(
                    2 * (X.T).dot(diff)
                    - self.regularisation.add_gradient_regularisation(self.W)
                )
                / self.num_samples
            )
            self._db = -2 * self.backend.sum(diff) / self.num_samples

            self._update_weights()
            self._update_history()

        logger.info(f"Training time: {time.time()-start} seconds")

        return self

    def _predict_preprocess(self, data):
        if self.normalize:
            data = manipulations.normalize(data)
        return self.backend.asarray(data)

    def predict(self, X):
        """
        Given an input array X, it returns the prediction array
        (GPU array if CuPy backend is enabled) after inferencing
        """
        return self._predict_preprocess(X).dot(self.W) + self.b

    def _predict(self, X):
        return X.dot(self.W) + self.b

    def plot_loss(self) -> None:
        """
        Plots the loss history curve during the training period.
        NOTE: This function just plots the graph, to display it
        explicitly call `matplotlib.pyplot.show()`
        """
        plt.title("MSE History")
        plt.xlabel("num_iterations")
        plt.ylabel("MSE")
        plt.legend()


class LinearRegressionGD(_BaseRegression):
    def __init__(
        self,
        num_iterations: int,
        learning_rate: float,
        normalize=True,
        initialiser="uniform",
        verbose=None,
    ) -> None:
        # regularisation of alpha 0 (essentially NIL)
        regularisation = _BaseRegularisation(multiply_factor=0, l1_ratio=0)
        super().__init__(
            num_iterations,
            learning_rate,
            normalize,
            regularisation,
            initialiser,
            verbose,
        )

    def plot_loss(self) -> None:
        plt.plot(self.history, label="Linear Regression")
        super().plot_loss()


class LassoRegressionGD(_BaseRegression):
    def __init__(
        self,
        num_iterations: int,
        learning_rate: float,
        l1_cost: float,
        normalize=True,
        initialiser="uniform",
        verbose=None,
    ) -> None:
        regularisation = L1Regularisation(l1_cost=l1_cost)
        super().__init__(
            num_iterations,
            learning_rate,
            normalize,
            regularisation,
            initialiser,
            verbose,
        )

    def plot_loss(self) -> None:
        plt.plot(self.history, label="Lasso Regression")
        super().plot_loss()


class RidgeRegressionGD(_BaseRegression):
    def __init__(
        self,
        num_iterations: int,
        learning_rate: float,
        l2_cost: float,
        normalize=True,
        initialiser="uniform",
        backend="cupy",
        verbose=None,
    ) -> None:
        regularisation = L2Regularisation(l2_cost=l2_cost)
        super().__init__(
            num_iterations,
            learning_rate,
            normalize,
            regularisation,
            initialiser,
            verbose,
        )

    def plot_loss(self) -> None:
        plt.plot(self.history, label="Ridge Regression")
        super().plot_loss()


class ElasticNetRegressionGD(_BaseRegression):
    def __init__(
        self,
        num_iterations: int,
        learning_rate: float,
        multiply_factor: float,
        l1_ratio: float,
        normalize=True,
        initialiser="uniform",
        verbose=None,
    ) -> None:
        regularisation = L1_L2Regularisation(
            multiply_factor=multiply_factor, l1_ratio=l1_ratio
        )
        super().__init__(
            num_iterations,
            learning_rate,
            normalize,
            regularisation,
            initialiser,
            verbose,
        )

    def plot_loss(self) -> None:
        plt.plot(self.history, label="Elastic Net Regression")
        super().plot_loss()


class PolynomialRegressionGD(ElasticNetRegressionGD):
    def __init__(
        self,
        num_iterations: int,
        learning_rate: float,
        degree: float,
        multiply_factor=None,
        l1_ratio=None,
        normalize=True,
        initialiser="uniform",
        verbose=None,
    ) -> None:
        self.degree = degree
        super().__init__(
            num_iterations,
            learning_rate,
            multiply_factor,
            l1_ratio,
            normalize,
            initialiser,
            verbose,
        )

    def _fit_preprocess(self, data, labels):
        poly_data = manipulations.transform_polynomial(data, self.degree)
        return super()._fit_preprocess(poly_data, labels)

    def _predict_preprocess(self, data):
        poly_data = manipulations.transform_polynomial(data, self.degree)
        return super()._predict_preprocess(poly_data)

    def plot_loss(self) -> None:
        plt.plot(
            self.history, label=f"Polynomial Regression, degree={self.degree}"
        )
        super().plot_loss()
