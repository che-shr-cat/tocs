"""Derives computed fields, converts types, applies business logic"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import EnrichedRecord, TransformedRecord
from ..utils.helpers import log_stage, timed


class StageTransformStage(StageBase):
    """Pipeline stage: derives computed fields, converts types, applies business logic"""

    name = "stage_transform"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: EnrichedRecord -> TransformedRecord."""
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._transform_records_one(record, config))
        return results

    def _transform_records_one(self, record: EnrichedRecord, config: Any) -> TransformedRecord:
        """Process a single record."""
        return TransformedRecord(id=record.id, data=record.data)


def transform_records(data: list[EnrichedRecord], config: Any) -> list[TransformedRecord]:
    """Functional interface for stage_transform."""
    stage = StageTransformStage()
    return stage.process(data, config)
