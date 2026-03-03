"""Stage D — final output formatting and serialization."""

import json as _json

from .base import StageBase
from .models import PipelineConfig, Record


class ExportStage(StageBase):
    """Formats records for output."""

    @property
    def name(self) -> str:
        return "export"

    def process(
        self, records: list[Record], config: PipelineConfig
    ) -> list[Record]:
        for r in records:
            r.metadata["output_format"] = "jsonl"
            r.metadata["final_stage"] = self.name
        return records

    def write(self, records: list[Record], config: PipelineConfig) -> str:
        """Serialize records to JSON-lines string."""
        lines = []
        for r in records:
            lines.append(_json.dumps({"id": r.id, "payload": r.payload}))
        return "\n".join(lines)
