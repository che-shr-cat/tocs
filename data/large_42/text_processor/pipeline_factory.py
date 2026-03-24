"""Pipeline factory: constructs configured pipeline instances."""

from typing import Any

from .config import PipelineConfig
from .registry import get_registry
from .runner import run_pipeline


class PipelineBuilder:
    """Builder pattern for constructing configured pipelines.

    Wires together registry, middleware, hooks, and configuration
    into a ready-to-run pipeline instance.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._middleware: list[Any] = []
        self._hooks: list[Any] = []

    def with_middleware(self, middleware_fn: Any) -> "PipelineBuilder":
        """Add a middleware function to the pipeline."""
        self._middleware.append(middleware_fn)
        return self

    def with_hook(self, hook_fn: Any) -> "PipelineBuilder":
        """Add a lifecycle hook to the pipeline."""
        self._hooks.append(hook_fn)
        return self

    def build(self) -> Any:
        """Build and return the configured pipeline runner."""
        return {
            "config": self._config,
            "middleware": self._middleware,
            "hooks": self._hooks,
            "registry": get_registry(),
        }


def create_pipeline(config: PipelineConfig | None = None) -> list[Any]:
    """Create and execute a pipeline with default configuration."""
    return run_pipeline(config)
