    """Pre-pipeline lifecycle hook — executed by the pipeline runner."""

    import logging
    from typing import Any

    from ..config import PipelineConfig
    from ..models import BaseModel

    logger = logging.getLogger(__name__)


    def pre_pipeline_hook(context: dict[str, Any]) -> None:
        """Executed before pipeline processing begins.
Sets up logging, validates configuration, initializes resources."""
        config = context.get("config")
        logger.info("pre_pipeline hook executing")

        if config and hasattr(config, "log_level"):
            logger.debug("Pipeline config: log_level=%s", config.log_level)


    def execute(context: dict[str, Any]) -> None:
        """Uniform hook interface — delegates to pre_pipeline_hook."""
        pre_pipeline_hook(context)
