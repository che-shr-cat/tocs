"""Metrics middleware for pipeline stages."""

import time
from typing import Any

from ..base import StageBase

_stage_durations: dict[str, list[float]] = {}


def metrics_middleware(stage: StageBase) -> StageBase:
    """Wrap a stage with timing metrics around process() calls."""
    original_process = stage.process

    def wrapped(data: list[Any], config: Any) -> list[Any]:
        start = time.monotonic()
        result = original_process(data, config)
        elapsed = time.monotonic() - start
        _stage_durations.setdefault(stage.name, []).append(elapsed)
        return result

    stage.process = wrapped  # type: ignore[assignment]
    return stage
