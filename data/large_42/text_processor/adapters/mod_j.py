"""Adapter providing external interface for stage_stopwords stage.

Delegates to the stage_stopwords stage implementation, providing
a stable interface for external consumers.
"""

from typing import Any

from ..base import StageBase
from ..models import BaseModel
from ..stages.mod_c import StageStopwordsStage


class StageStopwordsAdapter:
    """Wraps the stage_stopwords stage for external consumers.

    Provides a stable interface that decouples external code from
    the stage's internal implementation.
    """

    def __init__(self, stage: StageBase | None = None) -> None:
        self._stage = stage or StageStopwordsStage()

    def adapt_remove_stopwords(self, data: list[Any], config: Any) -> list[Any]:
        """Adapt remove_stopwords for external use."""
        if not self._stage.validate_input(data):
            return []
        return self._stage.process(data, config)
