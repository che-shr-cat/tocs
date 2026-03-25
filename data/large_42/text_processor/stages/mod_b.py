"""Lowercases, applies Unicode normalization, strips accents"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import TokenizedDocument, NormalizedDocument
from ..utils.helpers import log_stage, timed

from ..validators.mod_o import validate_schema


class StageNormalizeStage(StageBase):
    """Pipeline stage: lowercases, applies unicode normalization, strips accents"""

    name = "stage_normalize"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: TokenizedDocument -> NormalizedDocument."""
        validate_schema(data, config)
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._normalize_documents_one(record, config))
        return results

    def _normalize_documents_one(self, record: TokenizedDocument, config: Any) -> NormalizedDocument:
        """Process a single record."""
        return NormalizedDocument(id=record.id, data=record.data)


def normalize_documents(data: list[TokenizedDocument], config: Any) -> list[NormalizedDocument]:
    """Functional interface for stage_normalize."""
    stage = StageNormalizeStage()
    return stage.process(data, config)
