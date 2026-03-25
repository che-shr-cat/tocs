"""Abstract base class for pipeline stages."""

from abc import ABC, abstractmethod
from typing import Any

from .models import BaseModel
from .types import StageProtocol


class StageBase(ABC):
    """All pipeline stages must inherit from this class.

    Provides a uniform interface for the registry and middleware.
    """

    name: str = ""

    @abstractmethod
    def process(self, data: list[Any], config: Any) -> list[Any]:
        """Process a batch of records.

        Args:
            data: Input records (typed per stage).
            config: PipelineConfig instance.

        Returns:
            Transformed records.
        """
        ...

    def validate_input(self, data: list[Any]) -> bool:
        """Optional input validation hook."""
        return len(data) > 0
