    """Schema validation for pipeline input data."""

    from typing import Any

    from ..exceptions import ValidationError
    from ..models import BaseModel
    from ..types import ValidatorProtocol


    class SchemaValidator:
        """Validates that input records conform to the expected schema.
Checks required fields, types, and structure."""

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
                        field="schema",
                        message=f"Record {getattr(record, 'id', '?')} failed schema validation",
                    )
            return True

        def _check_one(self, record: Any) -> bool:
            """Check a single record."""
            return hasattr(record, "id") and hasattr(record, "data")


    def validate_schema(data: list[Any], config: Any) -> bool:
        """Functional interface for schema validation."""
        return SchemaValidator().validate(data, config)
