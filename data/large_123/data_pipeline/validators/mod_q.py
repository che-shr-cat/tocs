"""Referential integrity validation for pipeline input data."""

from typing import Any

from ..exceptions import ValidationError
from ..models import BaseModel
from ..types import ValidatorProtocol


class IntegrityValidator:
    """Validates referential integrity between records."""

    def validate(self, data: list[Any], config: Any) -> bool:
        """Validate a batch of records.

        Args:
            data: Input records to validate.
            config: PipelineConfig instance.

        Returns:
            True if all records pass validation.

        Raises:
            ValidationError: If validation fails.
        """
        for record in data:
            if not self._check_one(record):
                raise ValidationError(
                    field="integrity",
                    message=f"Record {getattr(record, 'id', '?')} failed integrity validation",
                )
        return True

    def _check_one(self, record: Any) -> bool:
        """Check a single record."""
        return hasattr(record, "id") and hasattr(record, "data")


def validate_integrity(data: list[Any], config: Any) -> bool:
    """Functional interface for integrity validation."""
    return IntegrityValidator().validate(data, config)
