"""Adapter providing external interface for stage_aggregate stage.

Delegates to the stage_aggregate stage implementation, providing
a stable interface for external consumers.
"""

from typing import Any

from ..base import StageBase
from ..models import BaseModel
from ..stages.mod_g import StageAggregateStage


class StageAggregateAdapter:
    """Wraps the stage_aggregate stage for external consumers.

    Provides a stable interface that decouples external code from
    the stage's internal implementation.
    """

    def __init__(self, stage: StageBase | None = None) -> None:
        self._stage = stage or StageAggregateStage()

    def adapt_aggregate_records(self, data: list[Any], config: Any) -> list[Any]:
        """Adapt aggregate_records for external use."""
        if not self._stage.validate_input(data):
            return []
        return self._stage.process(data, config)
