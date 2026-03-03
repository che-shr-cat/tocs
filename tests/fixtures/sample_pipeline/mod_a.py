"""Stage A — initial data ingestion and parsing."""

from .base import StageBase
from .models import PipelineConfig, Record


class IngestStage(StageBase):
    """Reads raw input and produces initial Record objects."""

    @property
    def name(self) -> str:
        return "ingest"

    def process(
        self, records: list[Record], config: PipelineConfig
    ) -> list[Record]:
        result: list[Record] = []
        for r in records:
            parsed = Record(
                id=r.id.strip().lower(),
                payload={k: str(v).strip() for k, v in r.payload.items()},
                metadata={**r.metadata, "source_stage": self.name},
            )
            result.append(parsed)
        return result
