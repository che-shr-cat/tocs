"""Builds inverted index entries for search and retrieval"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import SummarizedDocument, IndexedDocument
from ..utils.helpers import log_stage, timed

from ..validators.mod_q import validate_range


class StageIndexStage(StageBase):
    """Pipeline stage: builds inverted index entries for search and retrieval"""

    name = "stage_index"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: SummarizedDocument -> IndexedDocument."""
        validate_range(data, config)
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._index_documents_one(record, config))
        return results

    def _index_documents_one(self, record: SummarizedDocument, config: Any) -> IndexedDocument:
        """Process a single record."""
        return IndexedDocument(id=record.id, data=record.data)


def index_documents(data: list[SummarizedDocument], config: Any) -> list[IndexedDocument]:
    """Functional interface for stage_index."""
    stage = StageIndexStage()
    return stage.process(data, config)
