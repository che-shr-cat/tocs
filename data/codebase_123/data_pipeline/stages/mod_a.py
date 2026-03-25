"""Reads raw input, parses format, emits typed records"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import RawRecord, IngestedRecord
from ..utils.helpers import log_stage, timed



class StageIngestStage(StageBase):
    """Pipeline stage: reads raw input, parses format, emits typed records"""

    name = "stage_ingest"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: RawRecord -> IngestedRecord."""
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._ingest_records_one(record, config))
        return results

    def _ingest_records_one(self, record: RawRecord, config: Any) -> IngestedRecord:
        """Process a single record."""
        return IngestedRecord(id=record.id, data=record.data)


def ingest_records(data: list[RawRecord], config: Any) -> list[IngestedRecord]:
    """Functional interface for stage_ingest."""
    stage = StageIngestStage()
    return stage.process(data, config)
