"""Stage registry: maps stage names to classes via importlib."""

import importlib
from typing import Any

from .base import StageBase
from .config import PipelineConfig


class StageRegistry:
    """Discovers and instantiates pipeline stages by name.

    Stage modules are loaded dynamically using importlib, allowing
    the pipeline to be configured via pipeline_config.json without
    hard-coding stage imports.
    """

    def __init__(self) -> None:
        self._stages: dict[str, type[StageBase]] = {}

    def register(self, name: str, stage_cls: type[StageBase]) -> None:
        """Register a stage class by name."""
        self._stages[name] = stage_cls

    def discover(self, package: str, stage_names: list[str]) -> None:
        """Auto-discover stages from the stages subpackage."""
        for name in stage_names:
            module = importlib.import_module(f".stages.{name}", package)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, StageBase)
                    and attr is not StageBase
                ):
                    self._stages[name] = attr
                    break

    def get(self, name: str) -> type[StageBase]:
        """Retrieve a registered stage class."""
        if name not in self._stages:
            raise KeyError(f"Stage '{name}' not registered")
        return self._stages[name]

    def list_stages(self) -> list[str]:
        """Return names of all registered stages."""
        return list(self._stages.keys())


_registry: StageRegistry | None = None


def get_registry() -> StageRegistry:
    """Get or create the singleton stage registry."""
    global _registry
    if _registry is None:
        _registry = StageRegistry()
    return _registry
