from swi_ml.backend import _Backend


class Sigmoid(_Backend):
    def __init__(self):
        self.backend = super().get_backend()

    def activate(self, X):
        return 1 / (1 + self.backend.exp(-X))
