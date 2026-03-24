"""Formats aggregated records for downstream consumption"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import AggregatedRecord, ExportedRecord
from ..utils.helpers import log_stage, timed


class StageExportStage(StageBase):
    """Pipeline stage: formats aggregated records for downstream consumption"""

    name = "stage_export"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: AggregatedRecord -> ExportedRecord."""
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._export_records_one(record, config))
        return results

    def _export_records_one(self, record: AggregatedRecord, config: Any) -> ExportedRecord:
        """Process a single record."""
        return ExportedRecord(id=record.id, data=record.data)


def export_records(data: list[AggregatedRecord], config: Any) -> list[ExportedRecord]:
    """Functional interface for stage_export."""
    stage = StageExportStage()
    return stage.process(data, config)
