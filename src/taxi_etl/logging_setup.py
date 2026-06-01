"""Logging configuration shared across the pipeline.

A single ``get_logger`` helper gives every module a consistently formatted
logger. The format includes the stage name so multi-stage runs are easy to
follow in the console.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def _configure_root(level: int) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger("taxi_etl")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a namespaced logger under the ``taxi_etl`` root.

    Args:
        name: Logger name, conventionally the module's ``__name__``.
        level: Log level applied to the shared root on first call.
    """
    _configure_root(level)
    return logging.getLogger(f"taxi_etl.{name}")
