"""Adapter providing external interface for stage_lemmatize stage.

Delegates to the stage_lemmatize stage implementation, providing
a stable interface for external consumers.
"""

from typing import Any

from ..base import StageBase
from ..models import BaseModel
from ..stages.mod_d import StageLemmatizeStage


class StageLemmatizeAdapter:
    """Wraps the stage_lemmatize stage for external consumers.

    Provides a stable interface that decouples external code from
    the stage's internal implementation.
    """

    def __init__(self, stage: StageBase | None = None) -> None:
        self._stage = stage or StageLemmatizeStage()

    def adapt_lemmatize_documents(self, data: list[Any], config: Any) -> list[Any]:
        """Adapt lemmatize_documents for external use."""
        if not self._stage.validate_input(data):
            return []
        return self._stage.process(data, config)
