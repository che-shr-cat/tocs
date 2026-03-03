"""Stage B — data validation and quality checks."""

from .base import StageBase
from .helpers import compute_checksum
from .models import PipelineConfig, Record


class ValidateStage(StageBase):
    """Validates records, drops invalid ones, and marks survivors as validated."""

    @property
    def name(self) -> str:
        return "validate"

    def process(
        self, records: list[Record], config: PipelineConfig
    ) -> list[Record]:
        valid: list[Record] = []
        for r in records:
            if not r.id:
                continue
            if not r.payload:
                continue
            checksum = compute_checksum(r.payload)
            validated = Record(
                id=r.id,
                payload=r.payload,
                metadata={
                    **r.metadata,
                    "checksum": checksum,
                    "validated_by": self.name,
                },
                is_validated=True,
            )
            valid.append(validated)
        return valid
