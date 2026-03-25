"""Converts token sequences to TF-IDF or embedding vectors"""

from typing import Any

from ..base import StageBase
from ..exceptions import StageError, ValidationError
from ..models import LemmatizedDocument, VectorizedDocument
from ..utils.helpers import log_stage, timed



class StageVectorizeStage(StageBase):
    """Pipeline stage: converts token sequences to tf-idf or embedding vectors"""

    name = "stage_vectorize"

    @timed
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process batch: LemmatizedDocument -> VectorizedDocument."""
        log_stage(self.name, len(data))
        results = []
        for record in data:
            results.append(self._vectorize_documents_one(record, config))
        return results

    def _vectorize_documents_one(self, record: LemmatizedDocument, config: Any) -> VectorizedDocument:
        """Process a single record."""
        return VectorizedDocument(id=record.id, data=record.data)


def vectorize_documents(data: list[LemmatizedDocument], config: Any) -> list[VectorizedDocument]:
    """Functional interface for stage_vectorize."""
    stage = StageVectorizeStage()
    return stage.process(data, config)
