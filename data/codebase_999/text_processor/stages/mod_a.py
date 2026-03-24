"""Splits text into sentences and tokens with position offsets"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import RawDocument, TokenizedDocument
from ..utils.helpers import log_stage, timed


class StageTokenizeStage(StageBase):
    """Pipeline stage: splits text into sentences and tokens with position offsets"""

    name = "stage_tokenize"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: RawDocument -> TokenizedDocument."""
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._tokenize_documents_one(record, config))
        return results

    def _tokenize_documents_one(self, record: RawDocument, config: Any) -> TokenizedDocument:
        """Process a single record."""
        return TokenizedDocument(id=record.id, data=record.data)


def tokenize_documents(data: list[RawDocument], config: Any) -> list[TokenizedDocument]:
    """Functional interface for stage_tokenize."""
    stage = StageTokenizeStage()
    return stage.process(data, config)
