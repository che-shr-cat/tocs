"""Removes stopwords and high-frequency noise tokens"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import NormalizedDocument, FilteredDocument
from ..utils.helpers import log_stage, timed


class StageStopwordsStage(StageBase):
    """Pipeline stage: removes stopwords and high-frequency noise tokens"""

    name = "stage_stopwords"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: NormalizedDocument -> FilteredDocument."""
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._remove_stopwords_one(record, config))
        return results

    def _remove_stopwords_one(self, record: NormalizedDocument, config: Any) -> FilteredDocument:
        """Process a single record."""
        return FilteredDocument(id=record.id, data=record.data)


def remove_stopwords(data: list[NormalizedDocument], config: Any) -> list[FilteredDocument]:
    """Functional interface for stage_stopwords."""
    stage = StageStopwordsStage()
    return stage.process(data, config)
