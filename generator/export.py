"""Export codebase + ground_truth.json to a target directory.

Supports two export modes:
1. export_fixture() — copies the hand-authored fixture (for existing tests)
2. export_from_blueprint() — generates files from a PatternBlueprint (grammar-driven)
"""

from __future__ import annotations

import json
import shutil
import textwrap
import uuid
from pathlib import Path

from models import (
    CodebaseGroundTruth,
    ConstraintEvidenceType,
    DesignRationale,
    InvariantGroundTruth,
    ModuleGroundTruth,
)

from generator.grammar import PatternBlueprint, ModuleSpec

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_pipeline"


def load_ground_truth(codebase_dir: Path) -> CodebaseGroundTruth:
    """Load and validate ground_truth.json from a codebase directory."""
    gt_path = codebase_dir / "ground_truth.json"
    return CodebaseGroundTruth.model_validate_json(gt_path.read_text())


def export_fixture(output_dir: Path) -> CodebaseGroundTruth:
    """Copy the hand-authored fixture to *output_dir* and return its ground truth.

    The output directory will contain the codebase Python files,
    pipeline_config.json, test_boundaries.py, and ground_truth.json.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(FIXTURE_DIR, output_dir)

    # Remove __pycache__ if copied
    for pycache in output_dir.rglob("__pycache__"):
        shutil.rmtree(pycache)

    return load_ground_truth(output_dir)


def list_codebase_files(codebase_dir: Path) -> list[str]:
    """Return relative paths of all files in a codebase directory.

    Excludes ground_truth.json (metadata, not part of the codebase) and
    __pycache__ directories.
    """
    codebase_dir = Path(codebase_dir)
    files: list[str] = []
    for p in sorted(codebase_dir.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.name != "ground_truth.json":
            files.append(str(p.relative_to(codebase_dir)))
    return files


# ============================================================================
# Blueprint-driven export (grammar → files + ground_truth.json)
# ============================================================================


def export_from_blueprint(
    blueprint: PatternBlueprint,
    output_dir: Path,
) -> CodebaseGroundTruth:
    """Generate a codebase from a PatternBlueprint.

    Creates:
    - Directory structure with sub-packages
    - Python source files for each module (template-based)
    - pipeline_config.json for registry-wired codebases
    - ground_truth.json with typed edges, invariants, and contracts
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    pkg = blueprint.package_name

    # Create all directories needed
    dirs_needed: set[str] = set()
    for filepath in blueprint.modules:
        parts = filepath.split("/")
        if len(parts) > 2:  # has subdirectory
            dirs_needed.add("/".join(parts[:-1]))
        elif len(parts) == 2:
            dirs_needed.add(parts[0])
    for d in sorted(dirs_needed):
        (output_dir / d).mkdir(parents=True, exist_ok=True)

    # Generate each module's source file
    for filepath, spec in blueprint.modules.items():
        source = _generate_source(spec, blueprint)
        (output_dir / filepath).write_text(source)

    # Generate pipeline_config.json for medium codebases with registry
    has_registry = any(
        m.role == "registry" for m in blueprint.modules.values()
    )
    if has_registry:
        config = _generate_pipeline_config(blueprint)
        (output_dir / pkg / "pipeline_config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )

    # Generate test_smoke.py as test evidence
    test_path = output_dir / pkg / "test_smoke.py"
    test_path.write_text(_generate_test_evidence(blueprint))

    # Build and write ground_truth.json
    gt = _blueprint_to_ground_truth(blueprint)
    gt_path = output_dir / "ground_truth.json"
    gt_path.write_text(gt.model_dump_json(indent=2) + "\n")

    return gt


def _generate_source(spec: ModuleSpec, blueprint: PatternBlueprint) -> str:
    """Generate Python source code for a module based on its spec."""
    role = spec.role

    if role == "package_init":
        return _gen_init(spec, blueprint)
    elif role == "data_model":
        return _gen_models(spec)
    elif role == "base":
        return _gen_base(spec)
    elif role == "config":
        return _gen_config(spec)
    elif role == "exceptions":
        return _gen_exceptions(spec)
    elif role == "registry":
        return _gen_registry(spec, blueprint)
    elif role == "stage":
        return _gen_stage(spec)
    elif role == "adapter":
        return _gen_adapter(spec)
    elif role == "middleware":
        return _gen_middleware(spec)
    elif role == "utils":
        return _gen_utils(spec)
    elif role == "distractor":
        return _gen_distractor(spec)
    elif role == "orchestrator":
        return _gen_runner(spec, blueprint)
    elif role == "entry_point":
        return _gen_cli(spec)
    elif role == "source":
        return _gen_source_module(spec)
    elif role == "sink":
        return _gen_sink_module(spec)
    else:
        return f'"""Module: {spec.filepath} (role={role})"""\n'


def _rel_import(from_path: str, to_path: str) -> str:
    """Compute a relative import path between two module filepaths.

    E.g., from 'pkg/stages/mod_a.py' importing 'pkg/models.py'
    yields 'from ..models import ...'.
    """
    from_parts = from_path.replace(".py", "").split("/")
    to_parts = to_path.replace(".py", "").split("/")

    # Find common prefix
    common = 0
    for a, b in zip(from_parts, to_parts):
        if a == b:
            common += 1
        else:
            break

    ups = len(from_parts) - common - 1  # -1 for the filename itself
    dots = "." * (ups + 1)  # at least one dot for relative import
    remainder = ".".join(to_parts[common:])
    return f"{dots}{remainder}"


def _gen_init(spec: ModuleSpec, blueprint: PatternBlueprint) -> str:
    """Generate __init__.py content."""
    pkg = blueprint.package_name
    filepath = spec.filepath
    parts = filepath.split("/")

    if len(parts) == 2:
        # Top-level package init
        return f'"""{blueprint.domain_description}"""\n\n__version__ = "0.1.0"\n'
    else:
        subpkg = parts[-2]
        return f'"""{subpkg.capitalize()} subpackage."""\n'


def _gen_models(spec: ModuleSpec) -> str:
    """Generate models.py with Pydantic data types."""
    types = spec.generation_hints.get("types", "").split(",")
    lines = [
        '"""Shared data types for the pipeline."""',
        "",
        "from pydantic import BaseModel, Field",
        "",
    ]
    for t in types:
        t = t.strip()
        if not t:
            continue
        lines.extend([
            "",
            f"class {t}(BaseModel):",
            f'    """Data model: {t}."""',
            "",
            '    id: str = Field(default="", description="Record identifier")',
            "    data: dict = Field(default_factory=dict)",
            "",
        ])
    return "\n".join(lines) + "\n"


def _gen_base(spec: ModuleSpec) -> str:
    """Generate base.py with StageBase ABC."""
    return textwrap.dedent('''\
        """Abstract base class for pipeline stages."""

        from abc import ABC, abstractmethod
        from typing import Any

        from .models import BaseModel


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
    ''')


def _gen_config(spec: ModuleSpec) -> str:
    """Generate config.py."""
    fields = spec.generation_hints.get("config_fields", "batch_size").split(",")
    lines = [
        '"""Pipeline configuration."""',
        "",
        "from pathlib import Path",
        "from pydantic import BaseModel, Field",
        "",
        "",
        "class PipelineConfig(BaseModel):",
        '    """Runtime configuration for the pipeline."""',
        "",
    ]
    for f in fields:
        f = f.strip()
        if f == "batch_size":
            lines.append("    batch_size: int = Field(default=100)")
        elif f == "input_path":
            lines.append('    input_path: Path = Field(default=Path("data/input"))')
        elif f == "output_path":
            lines.append('    output_path: Path = Field(default=Path("data/output"))')
        elif f == "log_level":
            lines.append('    log_level: str = Field(default="INFO")')
        elif f == "stage_order":
            lines.append("    stage_order: list[str] = Field(default_factory=list)")
        else:
            lines.append(f'    {f}: str = Field(default="")')
    lines.append("")
    return "\n".join(lines) + "\n"


def _gen_exceptions(spec: ModuleSpec) -> str:
    """Generate exceptions.py."""
    return textwrap.dedent('''\
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
    ''')


def _gen_registry(spec: ModuleSpec, blueprint: PatternBlueprint) -> str:
    """Generate registry.py with importlib-based stage loading."""
    return textwrap.dedent('''\
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
    ''')


def _gen_stage(spec: ModuleSpec) -> str:
    """Generate a pipeline stage module."""
    func = spec.generation_hints.get("function_name", "process")
    in_type = spec.generation_hints.get("input_type", "Any")
    out_type = spec.generation_hints.get("output_type", "Any")
    desc = spec.generation_hints.get("description", spec.purpose)
    stage_name = spec.generation_hints.get("stage_name", "stage")

    return textwrap.dedent(f'''\
        """{desc}"""

        from typing import Any

        from ..base import StageBase
        from ..exceptions import StageError, ValidationError
        from ..models import {in_type}, {out_type}
        from ..utils.helpers import log_stage, timed


        class {stage_name.title().replace("_", "")}Stage(StageBase):
            """Pipeline stage: {desc.lower()}"""

            name = "{stage_name}"

            @timed
            def process(self, data: list[Any], config: Any) -> list[Any]:
                """Process batch: {in_type} -> {out_type}."""
                log_stage(self.name, len(data))
                results = []
                for record in data:
                    results.append(self._{func}_one(record, config))
                return results

            def _{func}_one(self, record: {in_type}, config: Any) -> {out_type}:
                """Process a single record."""
                return {out_type}(id=record.id, data=record.data)


        def {func}(data: list[{in_type}], config: Any) -> list[{out_type}]:
            """Functional interface for {stage_name}."""
            stage = {stage_name.title().replace("_", "")}Stage()
            return stage.process(data, config)
    ''')


def _gen_adapter(spec: ModuleSpec) -> str:
    """Generate an adapter module."""
    wrapped_func = spec.generation_hints.get("wrapped_function", "process")
    stage_name = spec.generation_hints.get("stage_name", "stage")

    return textwrap.dedent(f'''\
        """Adapter providing external interface for {stage_name} stage."""

        from typing import Any

        from ..base import StageBase


        class {stage_name.title().replace("_", "")}Adapter:
            """Wraps the {stage_name} stage for external consumers.

            Provides a stable interface that decouples external code from
            the stage's internal implementation.
            """

            def __init__(self, stage: StageBase) -> None:
                self._stage = stage

            def adapt_{wrapped_func}(self, data: list[Any], config: Any) -> list[Any]:
                """Adapt {wrapped_func} for external use."""
                if not self._stage.validate_input(data):
                    return []
                return self._stage.process(data, config)
    ''')


def _gen_middleware(spec: ModuleSpec) -> str:
    """Generate a middleware module."""
    kind = spec.generation_hints.get("middleware_type", "logging")

    if kind == "logging":
        return textwrap.dedent('''\
            """Logging middleware for pipeline stages."""

            import logging
            from typing import Any

            from ..base import StageBase

            logger = logging.getLogger(__name__)


            def logging_middleware(stage: StageBase) -> StageBase:
                """Wrap a stage with logging around process() calls."""
                original_process = stage.process

                def wrapped(data: list[Any], config: Any) -> list[Any]:
                    logger.info("Stage %s: processing %d records", stage.name, len(data))
                    result = original_process(data, config)
                    logger.info("Stage %s: produced %d records", stage.name, len(result))
                    return result

                stage.process = wrapped  # type: ignore[assignment]
                return stage
        ''')
    elif kind == "retry":
        return textwrap.dedent('''\
            """Retry middleware for pipeline stages."""

            import time
            from typing import Any

            from ..base import StageBase


            def retry_middleware(
                stage: StageBase, max_retries: int = 3, backoff: float = 0.1,
            ) -> StageBase:
                """Wrap a stage with retry logic around process() calls."""
                original_process = stage.process

                def wrapped(data: list[Any], config: Any) -> list[Any]:
                    last_exc: Exception | None = None
                    for attempt in range(max_retries):
                        try:
                            return original_process(data, config)
                        except Exception as exc:
                            last_exc = exc
                            time.sleep(backoff * (2 ** attempt))
                    raise RuntimeError(
                        f"Stage {stage.name} failed after {max_retries} retries"
                    ) from last_exc

                stage.process = wrapped  # type: ignore[assignment]
                return stage
        ''')
    else:  # metrics
        return textwrap.dedent('''\
            """Metrics middleware for pipeline stages."""

            import time
            from typing import Any

            from ..base import StageBase

            _stage_durations: dict[str, list[float]] = {}


            def metrics_middleware(stage: StageBase) -> StageBase:
                """Wrap a stage with timing metrics around process() calls."""
                original_process = stage.process

                def wrapped(data: list[Any], config: Any) -> list[Any]:
                    start = time.monotonic()
                    result = original_process(data, config)
                    elapsed = time.monotonic() - start
                    _stage_durations.setdefault(stage.name, []).append(elapsed)
                    return result

                stage.process = wrapped  # type: ignore[assignment]
                return stage
        ''')


def _gen_utils(spec: ModuleSpec) -> str:
    """Generate a utils module."""
    if "helpers" in spec.filepath:
        return textwrap.dedent('''\
            """Shared utilities: logging setup, timing, helpers."""

            import logging
            import functools
            import time
            from typing import Any, Callable


            def setup_logging(level: str = "INFO") -> None:
                """Configure pipeline-wide logging."""
                logging.basicConfig(
                    level=getattr(logging, level),
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                )


            def log_stage(stage_name: str, record_count: int) -> None:
                """Log stage execution."""
                logger = logging.getLogger(stage_name)
                logger.info("Processing %d records", record_count)


            def timed(func: Callable) -> Callable:
                """Decorator that logs function execution time."""

                @functools.wraps(func)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    start = time.monotonic()
                    result = func(*args, **kwargs)
                    elapsed = time.monotonic() - start
                    logging.getLogger(func.__qualname__).debug(
                        "Completed in %.3fs", elapsed,
                    )
                    return result

                return wrapper
        ''')
    else:  # formatters
        return textwrap.dedent('''\
            """Output formatting: JSON, CSV, table formatters."""

            import json
            from typing import Any


            def format_json(records: list[Any]) -> str:
                """Format records as JSON lines."""
                lines = []
                for r in records:
                    if hasattr(r, "model_dump"):
                        lines.append(json.dumps(r.model_dump()))
                    else:
                        lines.append(json.dumps(r))
                return "\\n".join(lines)


            def format_csv(records: list[Any], delimiter: str = ",") -> str:
                """Format records as CSV."""
                if not records:
                    return ""
                first = records[0]
                if hasattr(first, "model_dump"):
                    headers = list(first.model_dump().keys())
                else:
                    headers = list(first.keys()) if isinstance(first, dict) else []
                lines = [delimiter.join(headers)]
                for r in records:
                    d = r.model_dump() if hasattr(r, "model_dump") else r
                    lines.append(delimiter.join(str(d.get(h, "")) for h in headers))
                return "\\n".join(lines)


            def format_table(records: list[Any]) -> str:
                """Format records as a text table."""
                return format_csv(records, delimiter="\\t")
        ''')


def _gen_distractor(spec: ModuleSpec) -> str:
    """Generate a distractor (dead code) module."""
    if "old_pipeline" in spec.filepath:
        return textwrap.dedent('''\
            """DEPRECATED: Old monolithic pipeline implementation.

            This module is kept for historical reference only.
            It is NOT imported or used by any active code.

            The new pipeline uses a stage-based architecture with
            registry-driven stage resolution (see runner.py and registry.py).
            """

            # NOTE: This entire module is dead code.
            # It was the original implementation before the refactor to
            # the current stage-based architecture.


            def run_old_pipeline(input_path: str, output_path: str) -> None:
                """Run the old monolithic pipeline (deprecated)."""
                raise NotImplementedError(
                    "Old pipeline has been replaced. Use runner.run_pipeline() instead."
                )
        ''')
    else:  # compat
        return textwrap.dedent('''\
            """Compatibility shims for legacy pipeline format."""

            from ..models import *  # noqa: F401,F403


            def convert_legacy_format(data: dict) -> dict:
                """Convert legacy pipeline data format to current format.

                The old format used flat dicts with string keys.
                The new format uses typed Pydantic models.
                """
                return {
                    "id": data.get("record_id", data.get("id", "")),
                    "data": {k: v for k, v in data.items() if k not in ("record_id", "id")},
                }
        ''')


def _gen_runner(spec: ModuleSpec, blueprint: PatternBlueprint) -> str:
    """Generate runner.py (orchestrator)."""
    has_registry = any(
        m.role == "registry" for m in blueprint.modules.values()
    )
    if has_registry:
        return textwrap.dedent('''\
            """Pipeline runner: loads config, resolves stages, executes pipeline."""

            import json
            import logging
            from pathlib import Path
            from typing import Any

            from .config import PipelineConfig
            from .registry import get_registry
            from .models import *  # noqa: F401,F403
            from .base import StageBase

            logger = logging.getLogger(__name__)


            def run_pipeline(config: PipelineConfig | None = None) -> list[Any]:
                """Execute the full pipeline.

                Loads stages from the registry in config-specified order,
                applies middleware, and chains data through each stage.
                """
                if config is None:
                    config = PipelineConfig()

                registry = get_registry()
                registry.discover(__package__ or "", config.stage_order)

                data: list[Any] = []
                for stage_name in config.stage_order:
                    stage_cls = registry.get(stage_name)
                    stage = stage_cls()
                    data = stage.process(data, config)
                    logger.info(
                        "Stage %s: %d records", stage_name, len(data),
                    )

                return data
        ''')
    else:
        # Small blueprint: direct imports
        return textwrap.dedent('''\
            """Pipeline runner: chains all stages."""

            from typing import Any


            def run_pipeline(data: list[Any], config: Any) -> list[Any]:
                """Execute the pipeline stages in order."""
                return data
        ''')


def _gen_cli(spec: ModuleSpec) -> str:
    """Generate cli.py entry point."""
    return textwrap.dedent('''\
        """CLI entry point for the pipeline."""

        import argparse
        import sys

        from .config import PipelineConfig
        from .runner import run_pipeline


        def main(argv: list[str] | None = None) -> None:
            """Parse arguments and run the pipeline."""
            parser = argparse.ArgumentParser(description="Run the data pipeline")
            parser.add_argument("--input", type=str, default="data/input")
            parser.add_argument("--output", type=str, default="data/output")
            parser.add_argument("--batch-size", type=int, default=100)
            args = parser.parse_args(argv)

            config = PipelineConfig(
                input_path=args.input,
                output_path=args.output,
                batch_size=args.batch_size,
            )
            results = run_pipeline(config)
            print(f"Pipeline complete: {len(results)} records processed")


        if __name__ == "__main__":
            main()
    ''')


def _gen_source_module(spec: ModuleSpec) -> str:
    """Generate source.py for small blueprints."""
    out_type = spec.generation_hints.get("output_type", "Any")
    return textwrap.dedent(f'''\
        """{spec.purpose}"""

        from typing import Any


        def read_source(config: Any) -> list[Any]:
            """Read from the configured input source."""
            return []
    ''')


def _gen_sink_module(spec: ModuleSpec) -> str:
    """Generate sink.py for small blueprints."""
    in_type = spec.generation_hints.get("input_type", "Any")
    return textwrap.dedent(f'''\
        """{spec.purpose}"""

        from typing import Any


        def write_output(data: list[Any], config: Any) -> None:
            """Write processed data to the configured output sink."""
            pass
    ''')


def _generate_pipeline_config(blueprint: PatternBlueprint) -> dict:
    """Generate pipeline_config.json for registry-wired codebases."""
    stage_modules = []
    for m in blueprint.modules.values():
        if m.role == "stage":
            # Extract the neutral module name from path
            # e.g., "pkg/stages/mod_a.py" -> "mod_a"
            name = m.filepath.split("/")[-1].replace(".py", "")
            stage_modules.append({
                "module": name,
                "enabled": True,
            })
    return {
        "pipeline": {
            "name": blueprint.domain,
            "version": "0.1.0",
            "stages": stage_modules,
        },
        "settings": {
            "batch_size": 100,
            "log_level": "INFO",
        },
    }


def _generate_test_evidence(blueprint: PatternBlueprint) -> str:
    """Generate test_smoke.py as test evidence for invariants."""
    stage_paths = [
        m.filepath for m in blueprint.modules.values() if m.role == "stage"
    ]
    return textwrap.dedent(f'''\
        """Smoke tests verifying pipeline architectural invariants.

        IMPORTANT: These tests document and enforce critical architectural
        constraints. Stages must not import from each other directly.
        All stage access must go through the base interface.
        """

        import ast
        import importlib
        from pathlib import Path


        def test_no_inter_stage_imports():
            """Stages must not import from each other (boundary invariant)."""
            stages_dir = Path(__file__).parent / "stages"
            stage_files = sorted(stages_dir.glob("mod_*.py"))
            stage_names = {{f.stem for f in stage_files}}

            for sf in stage_files:
                tree = ast.parse(sf.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        imported = node.module.split(".")[-1]
                        assert imported not in stage_names or imported == sf.stem, (
                            f"{{sf.name}} imports from stage {{imported}}"
                        )


        def test_stages_inherit_base():
            """All stages must inherit from StageBase (interface invariant)."""
            # This test verifies the interface invariant is maintained
            stages_dir = Path(__file__).parent / "stages"
            for sf in sorted(stages_dir.glob("mod_*.py")):
                source = sf.read_text()
                assert "StageBase" in source, (
                    f"{{sf.name}} does not reference StageBase"
                )
    ''')


def _blueprint_to_ground_truth(blueprint: PatternBlueprint) -> CodebaseGroundTruth:
    """Convert a PatternBlueprint to CodebaseGroundTruth."""
    codebase_id = f"gen-{blueprint.pattern.value}-{blueprint.complexity.value}-{uuid.uuid4().hex[:8]}"

    # Build module ground truths
    gt_modules: dict[str, ModuleGroundTruth] = {}
    for filepath, spec in blueprint.modules.items():
        # Compute typed edges for this module
        module_edges = []
        for src, tgt, etype in blueprint.dependency_edges:
            if src == filepath:
                module_edges.append({
                    "target": tgt,
                    "type": etype,
                })
        gt_modules[filepath] = ModuleGroundTruth(
            filepath=filepath,
            purpose=spec.purpose,
            exports=spec.exports,
            edges=module_edges,
        )

    # Convert dependency edges to list of dicts
    gt_edges = [
        {"source": src, "target": tgt, "type": etype}
        for src, tgt, etype in blueprint.dependency_edges
    ]

    # Convert invariants
    gt_invariants = []
    for inv in blueprint.invariants:
        structured = _invariant_to_structured(inv)
        evidence_types = _infer_evidence_types(inv)
        gt_invariants.append(InvariantGroundTruth(
            id=inv.id,
            type=inv.type,
            description=inv.description,
            structured=structured,
            involved_modules=inv.involved_modules,
            rationale=inv.rationale,
            evidence_types=evidence_types,
            evidence_locations=inv.involved_modules[:2],
        ))

    # Convert design rationales
    gt_rationales = [
        DesignRationale(
            id=r.id,
            question=r.question,
            answer=r.answer,
            affected_modules=r.affected_modules,
            downstream_effects=r.downstream_effects,
        )
        for r in blueprint.design_rationales
    ]

    return CodebaseGroundTruth(
        codebase_id=codebase_id,
        pattern=blueprint.pattern,
        complexity=blueprint.complexity,
        language="python",
        modules=gt_modules,
        invariants=gt_invariants,
        design_rationales=gt_rationales,
        dependency_edges=gt_edges,
        contracts=[],
    )


def _invariant_to_structured(inv) -> dict:
    """Convert an InvariantSpec to a structured canonical form."""
    if inv.type.value == "BOUNDARY":
        if len(inv.involved_modules) == 2:
            return {
                "type": "FORBIDDEN_EDGE",
                "src": inv.involved_modules[0],
                "dst": inv.involved_modules[1],
            }
        return {
            "type": "FORBIDDEN_EDGE",
            "src": inv.involved_modules[0] if inv.involved_modules else None,
            "pattern": "no_import",
        }
    elif inv.type.value == "DATAFLOW":
        if len(inv.involved_modules) >= 2:
            return {
                "type": "VALIDATION_CHAIN",
                "src": inv.involved_modules[0],
                "dst": inv.involved_modules[-1],
            }
        return {"type": "VALIDATION_CHAIN"}
    elif inv.type.value == "INTERFACE":
        return {
            "type": "INTERFACE_ONLY",
            "via": inv.involved_modules[0] if inv.involved_modules else None,
        }
    elif inv.type.value == "INVARIANT":
        return {
            "type": "INVARIANT",
            "pattern": inv.description,
        }
    elif inv.type.value == "PURPOSE":
        return {
            "type": "PURPOSE",
            "pattern": inv.description,
        }
    return {"type": inv.type.value}


def _infer_evidence_types(inv) -> list[ConstraintEvidenceType]:
    """Infer evidence types from invariant type."""
    if inv.type.value == "BOUNDARY":
        return [ConstraintEvidenceType.STRUCTURAL, ConstraintEvidenceType.TEST]
    elif inv.type.value == "INTERFACE":
        return [ConstraintEvidenceType.STRUCTURAL]
    elif inv.type.value == "DATAFLOW":
        return [ConstraintEvidenceType.STRUCTURAL, ConstraintEvidenceType.DOCUMENTATION]
    elif inv.type.value == "INVARIANT":
        return [ConstraintEvidenceType.DOCUMENTATION]
    elif inv.type.value == "PURPOSE":
        return [ConstraintEvidenceType.DOCUMENTATION]
    return [ConstraintEvidenceType.STRUCTURAL]


# ── CLI entry point ────────────────────────────────────────────────
# Enables `python -m generator.export --pattern pipeline --complexity medium`

if __name__ == "__main__":
    from generator.__main__ import app
    app()
