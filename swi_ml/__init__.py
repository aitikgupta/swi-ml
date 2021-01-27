# changenote: If you're viewing this commit, then it means the name of the
# library is changed from cu-ml to swi-ml to avoid confusion with cuML, a
# library from RapidsAI (https://github.com/rapidsai/cuml)

import logging
import sys

from .logs import INFOFORMATTER, DEBUGFORMATTER

_fallback_to_numpy = False


def set_automatic_fallback(boolean):
    global _fallback_to_numpy
    _fallback_to_numpy = boolean


logger = logging.getLogger(__name__)

# create the stream handler
_ch = logging.StreamHandler()


def set_logging_level(level):
    # sets the handler info
    _ch.setLevel(level)
    # set the handler formatting
    if level == "DEBUG":
        _ch.setFormatter(logging.Formatter(DEBUGFORMATTER))
    else:
        _ch.setFormatter(logging.Formatter(INFOFORMATTER))


set_logging_level("INFO")
# adds the handler to the global variable: log
logger.addHandler(_ch)
logger.setLevel(logging.WARN)


from .backend import _Backend

# backend throughout the library, to be set by user
_global_backend = _Backend()


def set_backend(module_backend):
    _global_backend.set_backend(module_backend)


from .utils import (
    activations,
    distributions,
    manipulations,
    regularisers,
)

for module in (activations, distributions, manipulations, regularisers):
    full_name = "{}.{}".format(__package__, module.__name__.rsplit(".")[-1])
    sys.modules[full_name] = sys.modules[module.__name__]
