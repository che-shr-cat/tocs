"""Pipeline runner: loads config, resolves stages, executes pipeline."""

import json
import logging
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .registry import get_registry
from .models import *  # noqa: F401,F403
from .base import StageBase

logger = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig | None = None) -> list[Any]:
    """Execute the full pipeline.

    Loads stages from the registry in config-specified order,
    applies middleware, and chains data through each stage.
    """
    if config is None:
        config = PipelineConfig()

    registry = get_registry()
    registry.discover(__package__ or "", config.stage_order)

    data: list[Any] = []
    for stage_name in config.stage_order:
        stage_cls = registry.get(stage_name)
        stage = stage_cls()
        data = stage.process(data, config)
        logger.info(
            "Stage %s: %d records", stage_name, len(data),
        )

    return data
