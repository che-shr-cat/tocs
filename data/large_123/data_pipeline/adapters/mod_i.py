"""Adapter providing external interface for stage_validate stage."""

from typing import Any

from ..base import StageBase


class StageValidateAdapter:
    """Wraps the stage_validate stage for external consumers.

    Provides a stable interface that decouples external code from
    the stage's internal implementation.
    """

    def __init__(self, stage: StageBase) -> None:
        self._stage = stage

    def adapt_validate_records(self, data: list[Any], config: Any) -> list[Any]:
        """Adapt validate_records for external use."""
        if not self._stage.validate_input(data):
            return []
        return self._stage.process(data, config)
