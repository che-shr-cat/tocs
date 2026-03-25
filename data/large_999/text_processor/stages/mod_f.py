"""Assigns topic and category labels using trained classifier"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import VectorizedDocument, ClassifiedDocument
from ..utils.helpers import log_stage, timed

from ..validators.mod_q import validate_range


class StageClassifyStage(StageBase):
    """Pipeline stage: assigns topic and category labels using trained classifier"""

    name = "stage_classify"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: VectorizedDocument -> ClassifiedDocument."""
        validate_range(data, config)
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._classify_documents_one(record, config))
        return results

    def _classify_documents_one(self, record: VectorizedDocument, config: Any) -> ClassifiedDocument:
        """Process a single record."""
        return ClassifiedDocument(id=record.id, data=record.data)


def classify_documents(data: list[VectorizedDocument], config: Any) -> list[ClassifiedDocument]:
    """Functional interface for stage_classify."""
    stage = StageClassifyStage()
    return stage.process(data, config)
