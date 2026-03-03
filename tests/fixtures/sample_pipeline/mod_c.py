"""Stage C — data enrichment with external lookups.

IMPORTANT: Input records must have been validated (is_validated=True).
This stage assumes prior validation and will reject unvalidated records
to maintain data integrity guarantees.
"""

from .base import StageBase
from .models import PipelineConfig, Record

_ENRICHMENT_TABLE: dict[str, dict[str, str]] = {
    "region_a": {"timezone": "UTC+0", "currency": "GBP"},
    "region_b": {"timezone": "UTC+1", "currency": "EUR"},
    "default": {"timezone": "UTC+0", "currency": "USD"},
}


class EnrichStage(StageBase):
    """Enriches validated records with external reference data."""

    @property
    def name(self) -> str:
        return "enrich"

    def process(
        self, records: list[Record], config: PipelineConfig
    ) -> list[Record]:
        enriched: list[Record] = []
        for r in records:
            if not r.is_validated:
                if config.strict_mode:
                    raise ValueError(
                        f"Record {r.id} has not been validated. "
                        "Enrichment requires validated input."
                    )
                continue
            region = r.payload.get("region", "default")
            extra = _ENRICHMENT_TABLE.get(region, _ENRICHMENT_TABLE["default"])
            enriched_record = Record(
                id=r.id,
                payload={**r.payload, **extra},
                metadata={**r.metadata, "enriched_by": self.name},
                is_validated=r.is_validated,
            )
            enriched.append(enriched_record)
        return enriched
