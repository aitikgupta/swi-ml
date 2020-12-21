from __future__ import annotations
import math

import matplotlib.pyplot as plt


class LinearRegressionGD:
    def __init__(
        self, num_iterations: int, learning_rate: float, backend="cupy"
    ) -> None:
        if backend == "cupy":
            import cupy as cp
        elif backend == "numpy":
            import numpy as cp
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.history = []

    def _initialise_normal_weights(self, shape: tuple) -> None:
        self.num_samples, self.num_features = shape
        limit = 1 / math.sqrt(self.num_features)
        self.W = cp.asarray(
            cp.random.uniform(-limit, limit, (self.num_features,))
        )
        self.b = cp.zeros(
            1,
        )

    def _update_history(self) -> None:
        self.history.append(self.curr_loss)

    def _update_weights(self) -> LinearRegressionGD:
        self.W = self.W - self.learning_rate * self._dW
        self.b = self.b - self.learning_rate * self._db
        return self

    @staticmethod
    def MSE_loss(Y_true, Y_pred) -> float:
        return cp.mean(0.5 * (Y_true - Y_pred) ** 2)

    def fit(self, data, labels) -> LinearRegressionGD:
        self._initialise_normal_weights(data.shape)
        for _ in range(self.num_iterations):
            Y_pred = self.predict(data)
            diff = labels - Y_pred

            self.curr_loss = self.MSE_loss(labels, Y_pred)
            print(f"Current MSE: {self.curr_loss}")
            self._dW = -(2 * (data.T).dot(diff)) / self.num_samples
            self._db = -2 * cp.sum(diff) / self.num_samples
            self._update_weights()
            self._update_history()
        return self

    def predict(self, data) -> cp.ndarray:
        return data.dot(self.W) + self.b

    def plot_loss(self) -> None:
        plt.plot(self.history, color="orange")
        plt.title("MSE History")
        plt.xlabel("num_iterations")
        plt.ylabel("MSE")
        plt.show()
