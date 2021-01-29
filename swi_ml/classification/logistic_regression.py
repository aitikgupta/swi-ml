from swi_ml import activations
from swi_ml.regression.linear_regression import (
    _BaseRegression,
    L1_L2Regularisation,
)


class LogisticRegressionGD(_BaseRegression):
    def __init__(
        self,
        num_iterations: int,
        learning_rate: float,
        multiply_factor=None,
        l1_ratio=None,
        normalize=False,
        initialiser="uniform",
        verbose=None,
    ):
        self.activation = activations.Sigmoid()
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

    def predict(self, X, probability=False):
        """
        Given an input array X, it returns the prediction array
        (GPU array if CuPy backend is enabled) after inferencing
        """
        if probability:
            return self._predict(self._predict_preprocess(X))
        else:
            activated_pred = self.activation.activate(
                self._predict(self._predict_preprocess(X))
            )
            return self.backend.where(activated_pred > 0.5, 1, 0)

    def _predict(self, X):
        return self.activation.activate(
            self.backend.asarray(X).dot(self.W) + self.b
        )
