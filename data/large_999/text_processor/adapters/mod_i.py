"""Adapter providing external interface for stage_normalize stage.

Delegates to the stage_normalize stage implementation, providing
a stable interface for external consumers.
"""

from typing import Any

from ..base import StageBase
from ..models import BaseModel
from ..stages.mod_b import StageNormalizeStage


class StageNormalizeAdapter:
    """Wraps the stage_normalize stage for external consumers.

    Provides a stable interface that decouples external code from
    the stage's internal implementation.
    """

    def __init__(self, stage: StageBase | None = None) -> None:
        self._stage = stage or StageNormalizeStage()

    def adapt_normalize_documents(self, data: list[Any], config: Any) -> list[Any]:
        """Adapt normalize_documents for external use."""
        if not self._stage.validate_input(data):
            return []
        return self._stage.process(data, config)
