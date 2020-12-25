import cupy
import numpy

from cu_ml import logger

params = dict()


class _Backend:
    def __init__(self):
        global params
        self.backend = None

    def set_backend(self, backend):
        global params
        logger.info(f"Setting backend: {backend}")
        if backend == "numpy":
            params["backend"] = numpy
        elif backend == "cupy":
            params["backend"] = cupy
        else:
            raise NotImplementedError(
                "Only 'numpy' and 'cupy' backends are supported"
            )
        self.backend = params["backend"]

    def get_backend(self):
        global params
        if "backend" not in params.keys():
            logger.critical("Backend is not set, using default `numpy`")
            self.set_backend("numpy")
        return params["backend"]
