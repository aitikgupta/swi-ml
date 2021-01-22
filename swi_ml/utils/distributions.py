import math
from swi_ml.backend import _Backend


class GaussianDistribution(_Backend):
    """
    Gaussian Distribution Class
    """

    def __init__(self, mean, variance):
        self.mean = mean
        self.var = variance
        self.backend = super().get_backend()

    def pdf(self, x):
        """
        Returns likelihood of sample according to the distribution
        """
        # prevent division by zero
        eps = 1e-4
        coefficient = 1 / math.sqrt(2 * math.pi * self.var + eps)
        exponent = math.exp(
            -(math.pow(x - self.mean, 2) / (2 * self.var + eps))
        )
        return coefficient * exponent

    def __repr__(self):
        """
        Represents the characteristics of the distribution
        """
        return f"Gaussian [Mean: {self.mean:.2f}, Variance: {self.var:.2f}]"
