"""Standardizes formats, trims whitespace, applies canonical forms"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import ValidatedRecord, NormalizedRecord
from ..utils.helpers import log_stage, timed


class StageNormalizeStage(StageBase):
    """Pipeline stage: standardizes formats, trims whitespace, applies canonical forms"""

    name = "stage_normalize"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: ValidatedRecord -> NormalizedRecord."""
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._normalize_records_one(record, config))
        return results

    def _normalize_records_one(self, record: ValidatedRecord, config: Any) -> NormalizedRecord:
        """Process a single record."""
        return NormalizedRecord(id=record.id, data=record.data)


def normalize_records(data: list[ValidatedRecord], config: Any) -> list[NormalizedRecord]:
    """Functional interface for stage_normalize."""
    stage = StageNormalizeStage()
    return stage.process(data, config)
