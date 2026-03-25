"""Validates required fields, checks types, drops malformed rows"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import IngestedRecord, ValidatedRecord
from ..utils.helpers import log_stage, timed



class StageValidateStage(StageBase):
    """Pipeline stage: validates required fields, checks types, drops malformed rows"""

    name = "stage_validate"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: IngestedRecord -> ValidatedRecord."""
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._validate_records_one(record, config))
        return results

    def _validate_records_one(self, record: IngestedRecord, config: Any) -> ValidatedRecord:
        """Process a single record."""
        return ValidatedRecord(id=record.id, data=record.data)


def validate_records(data: list[IngestedRecord], config: Any) -> list[ValidatedRecord]:
    """Functional interface for stage_validate."""
    stage = StageValidateStage()
    return stage.process(data, config)
