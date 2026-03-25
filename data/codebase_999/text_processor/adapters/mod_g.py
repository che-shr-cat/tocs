"""Adapter providing external interface for stage_vectorize stage.

Delegates to the stage_vectorize stage implementation, providing
a stable interface for external consumers.
"""

from typing import Any

from ..base import StageBase
from ..models import BaseModel
from ..stages.mod_c import StageVectorizeStage


class StageVectorizeAdapter:
    """Wraps the stage_vectorize stage for external consumers.

    Provides a stable interface that decouples external code from
    the stage's internal implementation.
    """

    def __init__(self, stage: StageBase | None = None) -> None:
        self._stage = stage or StageVectorizeStage()

    def adapt_vectorize_documents(self, data: list[Any], config: Any) -> list[Any]:
        """Adapt vectorize_documents for external use."""
        if not self._stage.validate_input(data):
            return []
        return self._stage.process(data, config)
