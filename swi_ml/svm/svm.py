import logging
import math
import time

from swi_ml.backend import _Backend
from swi_ml import logger as _global_logger, manipulations

logger = logging.getLogger(__name__)


class SVM(_Backend):
    def __init__(
        self,
        num_iterations: int,
        learning_rate: float,
        regularisation_ratio=0.5,
        hinge_constant=1.0,
        normalize=False,
        regularisation=None,
        initialiser="uniform",
        verbose=None,
    ):
        if verbose is not None:
            logger.setLevel(verbose)
        else:
            logger.setLevel(_global_logger.getEffectiveLevel())
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.regularisation_ratio = regularisation_ratio
        self.hinge_constant = hinge_constant
        self.normalize = normalize
        self.initialiser = initialiser
        self.history = []
        self.backend = super().get_backend()

    def _initialise_uniform_weights(self, shape: tuple):
        self.num_samples, self.num_features = shape
        limit = 1 / math.sqrt(self.num_features)
        self.W = self.backend.asarray(
            self.backend.random.uniform(-limit, limit, (self.num_features,))
        )
        self.b = self.backend.zeros(
            1,
        )
        self.hinge_constant = self.backend.asarray(
            self.hinge_constant,
        )

    def _initialise_zeros_weights(self, shape: tuple):
        self.num_samples, self.num_features = shape
        self.W = self.backend.asarray(
            self.backend.zeros(
                self.num_features,
            )
        )
        self.b = self.backend.zeros(
            1,
        )

    def initialise_weights(self, X):
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

    def hinge_loss(self, Y, Y_pred):
        mul = self.backend.multiply(Y, Y_pred)
        sub = self.backend.subtract(self.hinge_constant, mul)
        return self.backend.mean(self.backend.where(sub < 0, 0, sub))

    def _fit_preprocess(self, data, labels):
        if (
            self.regularisation_ratio <= 0.0
            or self.regularisation_ratio >= 1.0
        ):
            raise ValueError("regularisation ratio should be between 0 and 1")

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
            Y_pred = self._predict(X, hinge=True)

            self.curr_loss = self.hinge_loss(Y, Y_pred)
            logger.info(
                f"Hinge Loss ({it+1}/{self.num_iterations}): {self.curr_loss}"
            )

            safe_zones = self.backend.where(
                Y <= 0, -self.hinge_constant, self.hinge_constant
            )

            for idx, sample in enumerate(X):
                if (
                    safe_zones[idx] * self._predict(sample)
                    >= self.hinge_constant
                ):
                    self._dW = 2 * self.regularisation_ratio * self.W
                else:
                    self._dW = 2 * self.regularisation_ratio * self.W - (
                        1 - self.regularisation_ratio
                    ) * sample.dot(safe_zones[idx])
                self._db = safe_zones[idx]

                self._update_weights()
            self._update_history()

        logger.info(f"Training time: {time.time()-start} seconds")

        return self

    def _update_weights(self):
        self.W = self.W - self.learning_rate * self._dW
        self.b = self.b - self.learning_rate * self._db
        return self

    def _predict(self, X, hinge=False):
        if hinge:
            return X.dot(self.W) - self.b
        else:
            return self.backend.sign(X.dot(self.W) - self.b)

    def _predict_preprocess(self, data):
        if self.normalize:
            data = manipulations.normalize(data)
        return self.backend.asarray(data)

    def predict(self, X, hinge=False):
        """
        Given an input array X, it returns the prediction array
        (GPU array if CuPy backend is enabled) after inferencing
        """
        return self._predict(self._predict_preprocess(X), hinge=hinge)

    def _update_history(self):
        self.history.append(self.curr_loss)
