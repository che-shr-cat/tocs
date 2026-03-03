"""Abstract base class for pipeline stages."""

from abc import ABC, abstractmethod

from .models import PipelineConfig, Record


class StageBase(ABC):
    """All pipeline stages must implement this interface.

    Stages are chained by the runner via the registry; they must NOT
    import or call each other directly.
    """

    @abstractmethod
    def process(
        self, records: list[Record], config: PipelineConfig
    ) -> list[Record]:
        """Process a batch of records and return transformed records."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable stage name for logging."""
        ...
