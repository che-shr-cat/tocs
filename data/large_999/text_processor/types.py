"""Protocol definitions and type aliases for the pipeline."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StageProtocol(Protocol):
    """Protocol that all pipeline stages must satisfy."""

    name: str

    def process(self, data: list[Any], config: Any) -> list[Any]: ...

    def validate_input(self, data: list[Any]) -> bool: ...


@runtime_checkable
class ValidatorProtocol(Protocol):
    """Protocol for input validators."""

    def validate(self, data: list[Any], config: Any) -> bool: ...


@runtime_checkable
class HookProtocol(Protocol):
    """Protocol for lifecycle hooks."""

    def execute(self, context: dict[str, Any]) -> None: ...


# Type aliases
PipelineData = list[Any]
StageResult = list[Any]
ValidationResult = bool
