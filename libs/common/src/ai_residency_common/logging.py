"""Logging setup.

The six repos between them contained roughly 495 `print()` calls and no logging
configuration at all, so output format, verbosity and destination varied per repo
and could not be turned down. `get_logger` replaces those calls with one consistent
format that respects a level.
"""

import logging
import os
import sys

__all__ = ["get_logger"]

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_configured = False


def _configure_root() -> None:
    """Attach a single stdout handler to the root logger, once per process.

    Level comes from the `LOG_LEVEL` environment variable, defaulting to INFO, so a
    container or a Makefile target can turn verbosity up without a code change.
    """
    global _configured
    if _configured:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for `name`, configuring output on first use.

    Args:
        name: Logger name. Pass `__name__` from the calling module so the log line
            identifies where it came from.

    Returns:
        A logger writing to stdout at the level set by `LOG_LEVEL`.
    """
    _configure_root()
    return logging.getLogger(name)
