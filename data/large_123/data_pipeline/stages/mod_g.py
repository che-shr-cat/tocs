"""Groups records by key dimensions and computes summary statistics"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import TransformedRecord, AggregatedRecord
from ..utils.helpers import log_stage, timed

from ..validators.mod_r import validate_range


class StageAggregateStage(StageBase):
    """Pipeline stage: groups records by key dimensions and computes summary statistics"""

    name = "stage_aggregate"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: TransformedRecord -> AggregatedRecord."""
        validate_range(data, config)
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._aggregate_records_one(record, config))
        return results

    def _aggregate_records_one(self, record: TransformedRecord, config: Any) -> AggregatedRecord:
        """Process a single record."""
        return AggregatedRecord(id=record.id, data=record.data)


def aggregate_records(data: list[TransformedRecord], config: Any) -> list[AggregatedRecord]:
    """Functional interface for stage_aggregate."""
    stage = StageAggregateStage()
    return stage.process(data, config)
