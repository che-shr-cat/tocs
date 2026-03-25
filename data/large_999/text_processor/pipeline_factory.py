"""Pipeline factory: constructs configured pipeline instances."""

from typing import Any

from .config import PipelineConfig
from .registry import get_registry
from .runner import run_pipeline

from .middleware.mod_l import logging_middleware
from .middleware.mod_m import retry_middleware
from .middleware.mod_n import metrics_middleware
from .hooks.mod_s import pre_pipeline_hook
from .hooks.mod_t import post_pipeline_hook
from .hooks.mod_u import on_error_hook


class PipelineBuilder:
    """Builder pattern for constructing configured pipelines.

    Wires together registry, middleware, hooks, and configuration
    into a ready-to-run pipeline instance.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._middleware: list[Any] = [logging_middleware, retry_middleware, metrics_middleware]
        self._hooks: list[Any] = [pre_pipeline_hook, post_pipeline_hook, on_error_hook]

    def with_middleware(self, middleware_fn: Any) -> "PipelineBuilder":
        """Add a middleware function to the pipeline."""
        self._middleware.append(middleware_fn)
        return self

    def with_hook(self, hook_fn: Any) -> "PipelineBuilder":
        """Add a lifecycle hook to the pipeline."""
        self._hooks.append(hook_fn)
        return self

    def build(self) -> dict[str, Any]:
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
