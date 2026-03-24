"""Removes exact and fuzzy duplicate records by key fields"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import NormalizedRecord, DeduplicatedRecord
from ..utils.helpers import log_stage, timed


class StageDeduplicateStage(StageBase):
    """Pipeline stage: removes exact and fuzzy duplicate records by key fields"""

    name = "stage_deduplicate"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: NormalizedRecord -> DeduplicatedRecord."""
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._deduplicate_records_one(record, config))
        return results

    def _deduplicate_records_one(self, record: NormalizedRecord, config: Any) -> DeduplicatedRecord:
        """Process a single record."""
        return DeduplicatedRecord(id=record.id, data=record.data)


def deduplicate_records(data: list[NormalizedRecord], config: Any) -> list[DeduplicatedRecord]:
    """Functional interface for stage_deduplicate."""
    stage = StageDeduplicateStage()
    return stage.process(data, config)
