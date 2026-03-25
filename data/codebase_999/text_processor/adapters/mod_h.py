"""Adapter providing external interface for stage_index stage.

Delegates to the stage_index stage implementation, providing
a stable interface for external consumers.
"""

from typing import Any

from ..base import StageBase
from ..models import BaseModel
from ..stages.mod_f import StageIndexStage


class StageIndexAdapter:
    """Wraps the stage_index stage for external consumers.

    Provides a stable interface that decouples external code from
    the stage's internal implementation.
    """

    def __init__(self, stage: StageBase | None = None) -> None:
        self._stage = stage or StageIndexStage()

    def adapt_index_documents(self, data: list[Any], config: Any) -> list[Any]:
        """Adapt index_documents for external use."""
        if not self._stage.validate_input(data):
            return []
        return self._stage.process(data, config)
