"""Post-pipeline lifecycle hook — executed by the pipeline runner."""

import logging
from typing import Any

from ..config import PipelineConfig
from ..models import BaseModel

logger = logging.getLogger(__name__)


def post_pipeline_hook(context: dict[str, Any]) -> None:
    """Executed after pipeline processing completes."""
    config = context.get("config")
    logger.info("post_pipeline hook executing")

    if config and hasattr(config, "log_level"):
        logger.debug("Pipeline config: log_level=%s", config.log_level)


def execute(context: dict[str, Any]) -> None:
    """Uniform hook interface — delegates to post_pipeline_hook."""
    post_pipeline_hook(context)
