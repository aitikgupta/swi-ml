from swi_ml.backend import _Backend


class _BaseRegularisation(_Backend):
    """
    Base Class for Regularisation, L1 and L2 regularisations inherit from this
    NOTE: Can be used directly as a L1_L2 (ElasticNet) Regularisation
    """

    def __init__(self, multiply_factor: float, l1_ratio: float):
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

    def __init__(self, l1_cost: float):
        multiply_factor = l1_cost
        l1_ratio = 1
        super().__init__(multiply_factor, l1_ratio)


class L2Regularisation(_BaseRegularisation):
    """
    Ridge Regression Regularisation
    """

    def __init__(self, l2_cost: float):
        multiply_factor = l2_cost
        l1_ratio = 0
        super().__init__(multiply_factor, l1_ratio)


class L1_L2Regularisation(_BaseRegularisation):
    """
    ElasticNet Regression Regularisation
    """

    def __init__(self, multiply_factor: float, l1_ratio: float):
        super().__init__(multiply_factor, l1_ratio)
