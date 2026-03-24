"""Logging middleware for pipeline stages."""

import logging
from typing import Any

from ..base import StageBase

logger = logging.getLogger(__name__)


def logging_middleware(stage: StageBase) -> StageBase:
    """Wrap a stage with logging around process() calls."""
    original_process = stage.process

    def wrapped(data: list[Any], config: Any) -> list[Any]:
        logger.info("Stage %s: processing %d records", stage.name, len(data))
        result = original_process(data, config)
        logger.info("Stage %s: produced %d records", stage.name, len(result))
        return result

    stage.process = wrapped  # type: ignore[assignment]
    return stage
