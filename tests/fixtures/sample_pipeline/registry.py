"""Stage registry — discovers pipeline stages from pipeline_config.json.

The pipeline order and stage classes are defined in the JSON config,
NOT hard-coded. This module dynamically imports stage modules at runtime.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from .base import StageBase

_CONFIG_PATH = Path(__file__).parent / "pipeline_config.json"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def get_pipeline() -> list[StageBase]:
    """Return ordered list of stage instances as defined in pipeline_config.json."""
    config = _load_config()
    stages: list[StageBase] = []
    for entry in config["pipeline"]["stages"]:
        mod = importlib.import_module(f'.{entry["module"]}', package=__package__)
        cls = getattr(mod, entry["class"])
        stages.append(cls())
    return stages


def get_batch_size() -> int:
    """Return batch_size from pipeline config."""
    return _load_config()["batch_size"]
