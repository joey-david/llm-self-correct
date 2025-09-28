"""Centralised logging helpers."""
from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger.

    The harness uses a simple stream handler to avoid noisy duplicate handlers when scripts are imported vs executed directly.
    """

    logger = logging.getLogger(name or "ucot")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
