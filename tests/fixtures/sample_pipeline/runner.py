"""Pipeline orchestrator — loads stages from registry and runs them in sequence."""

from __future__ import annotations

from .models import PipelineConfig, Record
from .registry import get_batch_size, get_pipeline


def run_pipeline(
    records: list[Record], config: PipelineConfig | None = None
) -> list[Record]:
    """Execute all pipeline stages in the order defined by pipeline_config.json."""
    if config is None:
        config = PipelineConfig(batch_size=get_batch_size())
    stages = get_pipeline()
    data = records
    for stage in stages:
        data = stage.process(data, config)
    return data
