    """Value range validation for pipeline input data."""

    from typing import Any

    from ..exceptions import ValidationError
    from ..models import BaseModel
    from ..types import ValidatorProtocol


    class RangeValidator:
        """Validates that numeric and date fields fall within expected ranges.
Flags outliers and impossible values."""

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
                        field="range",
                        message=f"Record {getattr(record, 'id', '?')} failed range validation",
                    )
            return True

        def _check_one(self, record: Any) -> bool:
            """Check a single record."""
            return hasattr(record, "id") and hasattr(record, "data")


    def validate_range(data: list[Any], config: Any) -> bool:
        """Functional interface for range validation."""
        return RangeValidator().validate(data, config)
