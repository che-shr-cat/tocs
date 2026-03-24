# Scoring Details

## Dependency F1

Compare predicted edges from the cognitive map against ground truth. An edge matches if `(source, target, type)` all agree via exact string matching. Targets must be specific file paths: a directory-level edge (e.g., `registry.py → stages/`) does not match file-level ground truth (e.g., `registry.py → stages/mod_a.py`).

## Invariant F1

### Five constraint types

Each generated codebase contains 15-16 planted constraints:

| Type | What it encodes | Example | Structured form |
|------|----------------|---------|-----------------|
| **BOUNDARY** | Forbidden dependencies | "Stages must not import each other" | `{type: FORBIDDEN_EDGE, src: stages/mod_a.py, pattern: no_import}` |
| **INTERFACE** | Access only through ABC | "All stages inherit StageBase" | `{type: INTERFACE_ONLY, via: base.py}` |
| **DATAFLOW** | Required processing chains | "mod_a output feeds mod_b input" | `{type: VALIDATION_CHAIN, src: stages/mod_a.py, dst: stages/mod_b.py}` |
| **INVARIANT** | Structural conventions | "batch_size must be respected by all stages" | `{type: INVARIANT, pattern: "PipelineConfig.batch_size..."}` |
| **PURPOSE** | Design rationale | "Stages are separate for single responsibility" | `{type: PURPOSE, pattern: "Each processing stage..."}` |

### Strict matching (not used — all models score 0)

Normalizes `structured` to 5-tuple `(type, src, dst, via, pattern)` and compares exact-match. Fails because:

- **Path prefix mismatch**: model writes `stages/mod_a.py`, GT has `text_processor/stages/mod_a.py`
- **Field placement confusion**: model puts `via` in `dst` or vice versa
- **Free-text pattern**: INVARIANT/PURPOSE have NL `pattern` field — exact match impossible

### Relaxed matching (used in paper)

Implemented in `metrics/map_accuracy.py` (`invariant_prf_relaxed`).

**Normalization** (`_normalise_relaxed`):
1. Reduces to 3-tuple: `(type, src, dst)` — **`via` and `pattern` completely ignored**
2. Strips package prefix: `text_processor/stages/mod_a.py` → `stages/mod_a.py`
3. For `INTERFACE_ONLY`: if `dst` empty but `via` filled, promotes `via → dst`

**Comparison** (`_relaxed_match`):
1. `type` must match exactly (`FORBIDDEN_EDGE` == `FORBIDDEN_EDGE`)
2. `src` and `dst`: if GT field is non-empty, predicted must match. If GT field is empty, it acts as a **wildcard** (matches anything)
3. Greedy 1-to-1: each GT invariant matched at most once

### Known limitation: INVARIANT and PURPOSE are type-counted, not content-matched

BOUNDARY and DATAFLOW have real `src`/`dst` fields, so relaxed matching is meaningful:
```
inv-007 (BOUNDARY):  ("FORBIDDEN_EDGE", "stages/mod_a.py", "")       ← src checked
inv-001 (DATAFLOW):  ("VALIDATION_CHAIN", "stages/mod_a.py", "stages/mod_b.py")  ← both checked
```

INVARIANT and PURPOSE have empty `src`/`dst` in ground truth:
```
inv-011 (INVARIANT): ("INVARIANT", "", "")  ← both are wildcards
inv-013 (PURPOSE):   ("PURPOSE", "", "")    ← both are wildcards
```

Since empty GT fields act as wildcards, **any predicted invariant with `type=INVARIANT` matches any GT invariant of type INVARIANT**. Scoring reduces to: "did the model predict the right number of each type?" — content is not verified.

**In codebase_42 (15 invariants):**
- 11/15 have meaningful structural matching (5 DATAFLOW + 5 BOUNDARY + 1 INTERFACE)
- 4/15 are type-counting only (2 INVARIANT + 2 PURPOSE)

### Planned fix (v0.2)

Design machine-comparable canonical forms for the weak types:
- **INVARIANT** → e.g. `{type: CONFIG_BINDING, src: config.py, dst: runner.py, field: batch_size}`
- **PURPOSE** → e.g. `{type: DESIGN_RATIONALE, pattern: single_responsibility, scope: stages/}`

This would make invariant scoring fully structural across all 5 types, eliminating the type-counting loophole.
