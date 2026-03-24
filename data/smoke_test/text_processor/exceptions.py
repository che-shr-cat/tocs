"""Custom exception hierarchy for pipeline errors."""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""


class StageError(PipelineError):
    """Raised when a pipeline stage fails."""

    def __init__(self, stage_name: str, message: str) -> None:
        self.stage_name = stage_name
        super().__init__(f"Stage '{stage_name}': {message}")


class ValidationError(PipelineError):
    """Raised when input validation fails."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(f"Validation error on '{field}': {message}")
