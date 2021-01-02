import logging

from .logs import INFOFORMATTER, DEBUGFORMATTER

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
