import logging

logger = logging.getLogger(__name__)
params = dict()

try:
    import numpy
except ImportError:
    raise ImportError(
        "'swi-ml has a single NumPy dependency, visit their installation "
        "guide: https://numpy.org/install/"
    )

try:
    import cupy

    _raise_cupy_error = False
except ImportError:
    _raise_cupy_error = True
    logger.warning(
        "No 'cupy' installation found, backend will be defaulted to 'numpy'"
    )


class _Backend:
    def __init__(self):
        global params
        self.backend = None

    def set_backend(self, backend):
        global params
        logger.warning(f"Setting backend: {backend}")
        if backend == "numpy":
            params["backend"] = numpy
        elif backend == "cupy":
            from swi_ml import _fallback_to_numpy

            if not _raise_cupy_error:
                params["backend"] = cupy
            elif _fallback_to_numpy:
                logger.warning(
                    "'cupy' backend not found, falling back to 'numpy'"
                )
                self.set_backend("numpy")
            else:
                raise ImportError(
                    "'cupy' backend needs to be installed first, visit "
                    "https://docs.cupy.dev/en/stable/install.html#install-cupy"
                )

        else:
            raise NotImplementedError(
                "Only 'numpy' and 'cupy' backends are supported"
            )
        self.backend = params["backend"]

    def get_backend(self):
        global params
        if "backend" not in params.keys():
            logger.critical("Backend is not set, using default 'numpy'")
            self.set_backend("numpy")
        return params["backend"]
