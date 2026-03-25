"""Shared utilities: logging setup, timing, helpers."""

import logging
import functools
import time
from typing import Any, Callable

from ..config import PipelineConfig


def setup_logging(level: str = "INFO") -> None:
    """Configure pipeline-wide logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


def log_stage(stage_name: str, record_count: int) -> None:
    """Log stage execution."""
    logger = logging.getLogger(stage_name)
    logger.info("Processing %d records", record_count)


def timed(func: Callable) -> Callable:
    """Decorator that logs function execution time."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        result = func(*args, **kwargs)
        elapsed = time.monotonic() - start
        logging.getLogger(func.__qualname__).debug(
            "Completed in %.3fs", elapsed,
        )
        return result

    return wrapper
