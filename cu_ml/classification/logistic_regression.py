from cu_ml import activations
from cu_ml.regression import ElasticNetRegressionGD


class LogisticRegressionGD(ElasticNetRegressionGD):
    def __init__(
        self,
        num_iterations: int,
        learning_rate: float,
        multiply_factor=None,
        l1_ratio=None,
        normalize=True,
        initialiser="uniform",
        verbose=None,
    ) -> None:
        self.activation = activations.Sigmoid()
        super().__init__(
            num_iterations,
            learning_rate,
            multiply_factor,
            l1_ratio,
            normalize,
            initialiser,
            verbose,
        )

    def predict(self, X, probability=False):
        """
        Given an input array X, it returns the prediction array
        (GPU array if CuPy backend is enabled) after inferencing
        """
        if probability:
            return self._predict_preprocess(X).dot(self.W) + self.b
        else:
            activated_pred = self.activation.activate(
                self._predict_preprocess(X).dot(self.W) + self.b
            )
            return self.backend.where(activated_pred > 0.5, 1, 0)

    def _predict(self, X):
        return self.activation.activate(
            self.backend.asarray(X).dot(self.W) + self.b
        )
