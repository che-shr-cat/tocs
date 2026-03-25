"""Adapter providing external interface for stage_validate stage.

Delegates to the stage_validate stage implementation, providing
a stable interface for external consumers.
"""

from typing import Any

from ..base import StageBase
from ..models import BaseModel
from ..stages.mod_b import StageValidateStage


class StageValidateAdapter:
    """Wraps the stage_validate stage for external consumers.

    Provides a stable interface that decouples external code from
    the stage's internal implementation.
    """

    def __init__(self, stage: StageBase | None = None) -> None:
        self._stage = stage or StageValidateStage()

    def adapt_validate_records(self, data: list[Any], config: Any) -> list[Any]:
        """Adapt validate_records for external use."""
        if not self._stage.validate_input(data):
            return []
        return self._stage.process(data, config)
