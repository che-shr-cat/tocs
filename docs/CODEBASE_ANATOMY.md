# Generated Codebase Anatomy

Example: **codebase_42** (seed 42, domain: text processing)

## Data Flow

```
RawDocument → TokenizedDocument → NormalizedDocument → FilteredDocument
  → LemmatizedDocument → SummarizedDocument → IndexedDocument
```

Each step is a separate module with **neutral naming** (`mod_a.py`, not `tokenizer.py`) to prevent architecture inference from filenames.

## Architectural Layers

| Layer | Files | Role |
|-------|-------|------|
| **Infrastructure** | `config.py`, `models.py`, `base.py`, `exceptions.py` | Pydantic config, data types, ABC |
| **Stages** | `stages/mod_a..f.py` | Each inherits `StageBase`, implements `process()` |
| **Adapters** | `adapters/mod_g.py`, `mod_h.py` | Wrappers over stages for external consumers |
| **Middleware** | `middleware/mod_i.py`, `mod_j.py` | Logging + retry, decorate `process()` |
| **Utils** | `utils/helpers.py`, `formatters.py` | Helpers (`@timed`, `log_stage`) |
| **Orchestration** | `registry.py`, `runner.py`, `cli.py` | Dynamic loading + execution |
| **Distractors** | `legacy/old_pipeline.py`, `legacy/compat.py` | Dead code (not in any dependency edge) |

## How the 4 Edge Types Arise

1. `cli.py → runner.py` — regular `import` → **IMPORTS** (~67%)
2. `adapters/mod_g.py → stages/mod_a.py` — adapter calls `stage.process()` → **CALLS_API** (~17%)
3. `registry.py → stages/mod_a.py` — via `importlib.import_module()` driven by `pipeline_config.json` → **REGISTRY_WIRES** (~9%)
4. `stages/mod_a.py → stages/mod_b.py` — data flows sequentially through runner → **DATA_FLOWS_TO** (~7%)

~1/3 of edges are invisible to import-following. Code is runnable but function bodies are trivial — the benchmark tests architectural understanding, not code logic.

## Architectural Constraints (15-16 per codebase)

Five types of planted, checkable constraints:

- **BOUNDARY**: Forbidden dependencies. Example: "No stage may import another stage — stages communicate only via data passed by the runner." Evidence: `test_smoke.py` asserts no cross-stage imports via AST parsing.
- **INTERFACE**: Access only through ABC. Example: "All stages inherit StageBase and implement `process()`." Evidence: `test_smoke.py` checks StageBase reference in every stage file.
- **DATAFLOW**: Required processing chains. Example: "mod_a must produce TokenizedDocument before mod_b can consume it." Evidence: docstrings (`Process batch: RawDocument -> TokenizedDocument`) and type imports in stage files.
- **INVARIANT**: Structural conventions. Example: "PipelineConfig.batch_size must be respected by all stages." Evidence: docstrings in `config.py` and `runner.py`.
- **PURPOSE**: Design rationale. Example: "Each stage is a separate module to enforce single responsibility." Evidence: docstrings in stage files.

Each constraint has a structured canonical form `{type, src, dst, via, pattern}` for automated scoring — no LLM judge.

## Key Code Examples

### StageBase ABC (`base.py`)
```python
class StageBase(ABC):
    name: str = ""

    @abstractmethod
    def process(self, data: list[Any], config: Any) -> list[Any]: ...
```

### Dynamic loading (`registry.py`)
```python
def discover(self, package: str, stage_names: list[str]) -> None:
    for name in stage_names:
        module = importlib.import_module(f".stages.{name}", package)
        # find StageBase subclass and register it
```

### Pipeline orchestration (`runner.py`)
```python
def run_pipeline(config: PipelineConfig | None = None) -> list[Any]:
    registry = get_registry()
    registry.discover(__package__ or "", config.stage_order)
    data: list[Any] = []
    for stage_name in config.stage_order:
        stage_cls = registry.get(stage_name)
        stage = stage_cls()
        data = stage.process(data, config)  # DATA_FLOWS_TO
    return data
```

### Middleware (`middleware/mod_i.py` — logging)
```python
def logging_middleware(stage: StageBase) -> StageBase:
    original_process = stage.process
    def wrapped(data: list[Any], config: Any) -> list[Any]:
        logger.info("Stage %s: processing %d records", stage.name, len(data))
        result = original_process(data, config)
        logger.info("Stage %s: produced %d records", stage.name, len(result))
        return result
    stage.process = wrapped
    return stage
```

### Distractors (`legacy/old_pipeline.py`)
```python
"""DEPRECATED: Old monolithic pipeline implementation.
This module is kept for historical reference only.
It is NOT imported or used by any active code."""

def run_old_pipeline(input_path: str, output_path: str) -> None:
    raise NotImplementedError("Old pipeline has been replaced. Use runner.run_pipeline() instead.")
```

### Config (`pipeline_config.json`)
```json
{
  "pipeline": {
    "stages": [
      {"module": "mod_a", "enabled": true},
      {"module": "mod_b", "enabled": true},
      ...
    ]
  }
}
```
