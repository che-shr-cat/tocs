"""Adapter providing external interface for stage_export stage.

Delegates to the stage_export stage implementation, providing
a stable interface for external consumers.
"""

from typing import Any

from ..base import StageBase
from ..models import BaseModel
from ..stages.mod_g import StageExportStage


class StageExportAdapter:
    """Wraps the stage_export stage for external consumers.

    Provides a stable interface that decouples external code from
    the stage's internal implementation.
    """

    def __init__(self, stage: StageBase | None = None) -> None:
        self._stage = stage or StageExportStage()

    def adapt_export_records(self, data: list[Any], config: Any) -> list[Any]:
        """Adapt export_records for external use."""
        if not self._stage.validate_input(data):
            return []
        return self._stage.process(data, config)
