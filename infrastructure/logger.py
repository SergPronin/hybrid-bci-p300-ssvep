import logging
from typing import Optional


def _configure_root_logger(level: int = logging.INFO) -> None:
    """
    Configure the root logger once.

    This keeps logging setup in the infrastructure layer so that application
    and domain code do not repeat configuration details.
    """
    if logging.getLogger().handlers:
        # Already configured
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger instance.

    In the future this is the place to add LSL or file-based logging without
    touching the stimulus or application layers.
    """
    _configure_root_logger()
    return logging.getLogger(name)

