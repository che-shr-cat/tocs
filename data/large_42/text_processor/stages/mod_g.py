"""Generates extractive summaries from classified documents"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import ClassifiedDocument, SummarizedDocument
from ..utils.helpers import log_stage, timed

from ..validators.mod_q import validate_range


class StageSummarizeStage(StageBase):
    """Pipeline stage: generates extractive summaries from classified documents"""

    name = "stage_summarize"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: ClassifiedDocument -> SummarizedDocument."""
        validate_range(data, config)
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._summarize_documents_one(record, config))
        return results

    def _summarize_documents_one(self, record: ClassifiedDocument, config: Any) -> SummarizedDocument:
        """Process a single record."""
        return SummarizedDocument(id=record.id, data=record.data)


def summarize_documents(data: list[ClassifiedDocument], config: Any) -> list[SummarizedDocument]:
    """Functional interface for stage_summarize."""
    stage = StageSummarizeStage()
    return stage.process(data, config)
