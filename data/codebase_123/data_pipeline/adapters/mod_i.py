"""Adapter providing external interface for stage_transform stage.

Delegates to the stage_transform stage implementation, providing
a stable interface for external consumers.
"""

from typing import Any

from ..base import StageBase
from ..models import BaseModel
from ..stages.mod_e import StageTransformStage


class StageTransformAdapter:
    """Wraps the stage_transform stage for external consumers.

    Provides a stable interface that decouples external code from
    the stage's internal implementation.
    """

    def __init__(self, stage: StageBase | None = None) -> None:
        self._stage = stage or StageTransformStage()

    def adapt_transform_records(self, data: list[Any], config: Any) -> list[Any]:
        """Adapt transform_records for external use."""
        if not self._stage.validate_input(data):
            return []
        return self._stage.process(data, config)
