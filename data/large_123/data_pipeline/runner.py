"""Pipeline runner: loads config, resolves stages, executes pipeline.

Stage ordering is driven by pipeline_config.json — the runner reads
the config file, extracts the stage list, and resolves each stage
through the registry via importlib.
"""

import json
import logging
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .registry import get_registry
from .models import *  # noqa: F401,F403
from .base import StageBase

from .middleware.mod_m import logging_middleware
from .middleware.mod_n import retry_middleware
from .middleware.mod_o import metrics_middleware
from .hooks.mod_s import pre_pipeline_hook
from .hooks.mod_t import post_pipeline_hook
from .hooks.mod_u import on_error_hook
from .serializers.mod_v import serialize_json

logger = logging.getLogger(__name__)

# Path to the pipeline configuration file
_CONFIG_PATH = Path(__file__).parent / "pipeline_config.json"


def _load_stage_order() -> list[str]:
    """Load stage ordering from pipeline_config.json.

    The config file lists stage modules in execution order.
    Each entry has a 'module' name (e.g. 'mod_a') and 'enabled' flag.
    """
    with open(_CONFIG_PATH) as f:
        raw = json.load(f)
    return [
        s["module"] for s in raw["pipeline"]["stages"]
        if s.get("enabled", True)
    ]


def run_pipeline(config: PipelineConfig | None = None) -> list[Any]:
    """Execute the full pipeline.

    Loads stage ordering from pipeline_config.json, resolves each
    stage through the registry, applies middleware, and chains data.
    """
    if config is None:
        config = PipelineConfig()

    # Load stage ordering from config file
    if not config.stage_order:
        config.stage_order = _load_stage_order()

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
