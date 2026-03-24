"""Retry middleware for pipeline stages."""

import time
from typing import Any

from ..base import StageBase


def retry_middleware(
    stage: StageBase, max_retries: int = 3, backoff: float = 0.1,
) -> StageBase:
    """Wrap a stage with retry logic around process() calls."""
    original_process = stage.process

    def wrapped(data: list[Any], config: Any) -> list[Any]:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return original_process(data, config)
            except Exception as exc:
                last_exc = exc
                time.sleep(backoff * (2 ** attempt))
        raise RuntimeError(
            f"Stage {stage.name} failed after {max_retries} retries"
        ) from last_exc

    stage.process = wrapped  # type: ignore[assignment]
    return stage
