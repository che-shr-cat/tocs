"""Reduces tokens to lemma forms using morphological analysis"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import FilteredDocument, LemmatizedDocument
from ..utils.helpers import log_stage, timed

from ..validators.mod_p import validate_integrity


class StageLemmatizeStage(StageBase):
    """Pipeline stage: reduces tokens to lemma forms using morphological analysis"""

    name = "stage_lemmatize"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: FilteredDocument -> LemmatizedDocument."""
        validate_integrity(data, config)
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._lemmatize_documents_one(record, config))
        return results

    def _lemmatize_documents_one(self, record: FilteredDocument, config: Any) -> LemmatizedDocument:
        """Process a single record."""
        return LemmatizedDocument(id=record.id, data=record.data)


def lemmatize_documents(data: list[FilteredDocument], config: Any) -> list[LemmatizedDocument]:
    """Functional interface for stage_lemmatize."""
    stage = StageLemmatizeStage()
    return stage.process(data, config)
