"""Joins external lookup data (e.g., geo-IP, currency rates)"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import DeduplicatedRecord, EnrichedRecord
from ..utils.helpers import log_stage, timed


class StageEnrichStage(StageBase):
    """Pipeline stage: joins external lookup data (e.g., geo-ip, currency rates)"""

    name = "stage_enrich"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: DeduplicatedRecord -> EnrichedRecord."""
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._enrich_records_one(record, config))
        return results

    def _enrich_records_one(self, record: DeduplicatedRecord, config: Any) -> EnrichedRecord:
        """Process a single record."""
        return EnrichedRecord(id=record.id, data=record.data)


def enrich_records(data: list[DeduplicatedRecord], config: Any) -> list[EnrichedRecord]:
    """Functional interface for stage_enrich."""
    stage = StageEnrichStage()
    return stage.process(data, config)
