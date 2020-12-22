from __future__ import annotations

import math
import time
import logging

import cupy
import numpy
import matplotlib.pyplot as plt


class LinearRegressionGD:
    def __init__(
        self,
        num_iterations: int,
        learning_rate: float,
        backend="cupy",
        verbose="INFO",
    ) -> None:
        logging.basicConfig(level=logging._nameToLevel[verbose])
        if backend == "cupy":
            self.backend = cupy
        elif backend == "numpy":
            self.backend = numpy
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.history = []

    def _initialise_uniform_weights(self, shape: tuple) -> None:
        self.num_samples, self.num_features = shape
        limit = 1 / math.sqrt(self.num_features)
        self.W = self.backend.asarray(
            self.backend.random.uniform(-limit, limit, (self.num_features,))
        )
        self.b = self.backend.zeros(
            1,
        )

    def _update_history(self) -> None:
        self.history.append(self.curr_loss)

    def _update_weights(self) -> LinearRegressionGD:
        self.W = self.W - self.learning_rate * self._dW
        self.b = self.b - self.learning_rate * self._db
        return self

    # @staticmethod
    def MSE_loss(self, Y_true, Y_pred) -> float:
        return self.backend.mean(0.5 * (Y_true - Y_pred) ** 2)

    def fit(self, data, labels) -> LinearRegressionGD:
        X = self.backend.asarray(data)
        Y = self.backend.asarray(labels)
        self._initialise_uniform_weights(X.shape)
        start = time.time()

        for it in range(self.num_iterations):
            Y_pred = self.predict(X)
            diff = Y - Y_pred

            self.curr_loss = self.MSE_loss(Y, Y_pred)
            logging.info(
                f" MSE ({it+1}/{self.num_iterations}): {self.curr_loss}"
            )
            self._dW = -(2 * (X.T).dot(diff)) / self.num_samples
            self._db = -2 * self.backend.sum(diff) / self.num_samples
            self._update_weights()
            self._update_history()

        logging.info(f" Training time: {time.time()-start} seconds")

        return self

    def predict(self, X):
        return self.backend.asarray(X).dot(self.W) + self.b

    def plot_loss(self) -> None:
        plt.plot(self.history, label="Linear Regression")
        plt.title("MSE History")
        plt.xlabel("num_iterations")
        plt.ylabel("MSE")
        plt.legend()


class LassoRegressionGD:
    def __init__(
        self,
        num_iterations: int,
        learning_rate: float,
        l1_cost: float,
        backend="cupy",
        verbose="INFO",
    ) -> None:
        logging.basicConfig(level=logging._nameToLevel[verbose])
        if backend == "cupy":
            self.backend = cupy
        elif backend == "numpy":
            self.backend = numpy
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.history = []
        self.l1_cost = l1_cost

    def _initialise_uniform_weights(self, shape: tuple) -> None:
        self.num_samples, self.num_features = shape
        limit = 1 / math.sqrt(self.num_features)
        self.W = self.backend.asarray(
            self.backend.random.uniform(-limit, limit, (self.num_features,))
        )
        self.b = self.backend.zeros(
            1,
        )

    def _update_history(self) -> None:
        self.history.append(self.curr_loss)

    def _update_weights(self) -> LinearRegressionGD:
        self.W = self.W - self.learning_rate * self._dW
        self.b = self.b - self.learning_rate * self._db
        return self

    # @staticmethod
    def MSE_loss(self, Y_true, Y_pred) -> float:
        return self.backend.mean(0.5 * (Y_true - Y_pred) ** 2)

    def fit(self, data, labels) -> LinearRegressionGD:
        X = self.backend.asarray(data)
        Y = self.backend.asarray(labels)
        self._initialise_uniform_weights(X.shape)
        start = time.time()
        for it in range(self.num_iterations):
            Y_pred = self.predict(X)
            diff = Y - Y_pred

            self.curr_loss = self.MSE_loss(Y, Y_pred)
            logging.info(
                f" MSE ({it+1}/{self.num_iterations}): {self.curr_loss}"
            )
            self._dW = (
                -(
                    2 * (X.T).dot(diff)
                    - self.l1_cost * self.backend.sign(self.W)
                )
                / self.num_samples
            )
            self._db = -2 * self.backend.sum(diff) / self.num_samples
            self._update_weights()
            self._update_history()

        logging.info(f" Training time: {time.time()-start} seconds")

        return self

    def predict(self, X):
        return self.backend.asarray(X).dot(self.W) + self.b

    def plot_loss(self) -> None:
        plt.plot(self.history, label="Lasso Regression")
        plt.title("MSE History")
        plt.xlabel("num_iterations")
        plt.ylabel("MSE")
        plt.legend()


class RidgeRegressionGD:
    def __init__(
        self,
        num_iterations: int,
        learning_rate: float,
        l2_cost: float,
        backend="cupy",
        verbose="INFO",
    ) -> None:
        logging.basicConfig(level=logging._nameToLevel[verbose])
        if backend == "cupy":
            self.backend = cupy
        elif backend == "numpy":
            self.backend = numpy
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.history = []
        self.l2_cost = l2_cost

    def _initialise_uniform_weights(self, shape: tuple) -> None:
        self.num_samples, self.num_features = shape
        limit = 1 / math.sqrt(self.num_features)
        self.W = self.backend.asarray(
            self.backend.random.uniform(-limit, limit, (self.num_features,))
        )
        self.b = self.backend.zeros(
            1,
        )

    def _update_history(self) -> None:
        self.history.append(self.curr_loss)

    def _update_weights(self) -> LinearRegressionGD:
        self.W = self.W - self.learning_rate * self._dW
        self.b = self.b - self.learning_rate * self._db
        return self

    # @staticmethod
    def MSE_loss(self, Y_true, Y_pred) -> float:
        return self.backend.mean(0.5 * (Y_true - Y_pred) ** 2)

    def fit(self, data, labels) -> LinearRegressionGD:
        X = self.backend.asarray(data)
        Y = self.backend.asarray(labels)
        self._initialise_uniform_weights(X.shape)
        start = time.time()
        for it in range(self.num_iterations):
            Y_pred = self.predict(X)
            diff = Y - Y_pred

            self.curr_loss = self.MSE_loss(Y, Y_pred)
            logging.info(
                f" MSE ({it+1}/{self.num_iterations}): {self.curr_loss}"
            )
            self._dW = (
                -(2 * (X.T).dot(diff) - 2 * self.l2_cost * self.W)
                / self.num_samples
            )
            self._db = -2 * self.backend.sum(diff) / self.num_samples
            self._update_weights()
            self._update_history()

        logging.info(f" Training time: {time.time()-start} seconds")

        return self

    def predict(self, X):
        return self.backend.asarray(X).dot(self.W) + self.b

    def plot_loss(self) -> None:
        plt.plot(self.history, label="Ridge Regression")
        plt.title("MSE History")
        plt.xlabel("num_iterations")
        plt.ylabel("MSE")
        plt.legend()


class ElasticNetRegressionGD:
    def __init__(
        self,
        num_iterations: int,
        learning_rate: float,
        multiply_factor: float,
        l1_ratio: float,
        backend="cupy",
        verbose="INFO",
    ) -> None:
        logging.basicConfig(level=logging._nameToLevel[verbose])
        if backend == "cupy":
            self.backend = cupy
        elif backend == "numpy":
            self.backend = numpy
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.history = []
        self.multiply_factor = multiply_factor
        self.l1_ratio = l1_ratio

    def _initialise_uniform_weights(self, shape: tuple) -> None:
        self.num_samples, self.num_features = shape
        limit = 1 / math.sqrt(self.num_features)
        self.W = self.backend.asarray(
            self.backend.random.uniform(-limit, limit, (self.num_features,))
        )
        self.b = self.backend.zeros(
            1,
        )

    def _update_history(self) -> None:
        self.history.append(self.curr_loss)

    def _update_weights(self) -> LinearRegressionGD:
        self.W = self.W - self.learning_rate * self._dW
        self.b = self.b - self.learning_rate * self._db
        return self

    # @staticmethod
    def MSE_loss(self, Y_true, Y_pred) -> float:
        return self.backend.mean(0.5 * (Y_true - Y_pred) ** 2)

    def fit(self, data, labels) -> LinearRegressionGD:
        X = self.backend.asarray(data)
        Y = self.backend.asarray(labels)
        self._initialise_uniform_weights(X.shape)
        start = time.time()
        for it in range(self.num_iterations):
            Y_pred = self.predict(X)
            diff = Y - Y_pred

            self.curr_loss = self.MSE_loss(Y, Y_pred)
            logging.info(
                f" MSE ({it+1}/{self.num_iterations}): {self.curr_loss}"
            )
            self._dW = (
                -(
                    2 * (X.T).dot(diff)
                    - self.multiply_factor
                    * self.l1_ratio
                    * self.backend.sign(self.W)
                    - self.multiply_factor * 2 * (1 - self.l1_ratio) * self.W
                )
                / self.num_samples
            )
            self._db = -2 * self.backend.sum(diff) / self.num_samples
            self._update_weights()
            self._update_history()

        logging.info(f" Training time: {time.time()-start} seconds")

        return self

    def predict(self, X):
        return self.backend.asarray(X).dot(self.W) + self.b

    def plot_loss(self) -> None:
        plt.plot(self.history, label="Elastic Net Regression")
        plt.title("MSE History")
        plt.xlabel("num_iterations")
        plt.ylabel("MSE")
        plt.legend()
