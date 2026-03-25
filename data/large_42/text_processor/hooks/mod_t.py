"""Error handling lifecycle hook — executed by the pipeline runner."""

import logging
from typing import Any

from ..config import PipelineConfig
from ..models import BaseModel

logger = logging.getLogger(__name__)


def on_error_hook(context: dict[str, Any]) -> None:
    """Executed when pipeline processing encounters an error."""
    config = context.get("config")
    logger.info("on_error hook executing")

    if config and hasattr(config, "log_level"):
        logger.debug("Pipeline config: log_level=%s", config.log_level)


def execute(context: dict[str, Any]) -> None:
    """Uniform hook interface — delegates to on_error_hook."""
    on_error_hook(context)
