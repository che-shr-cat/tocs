"""Cognitive map probing: prompt, parser, and differ.

1. PROBE_PROMPT — asks the model to externalize its architectural belief as JSON.
2. parse_cognitive_map() — fuzzy JSON extraction from model output.
3. diff_maps() — compares two CognitiveMaps for belief evolution tracking.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import ValidationError

from models import CognitiveMap


# ============================================================================
# Probe prompt
# ============================================================================

PROBE_PROMPT = """\
Based on everything you have observed so far, externalize your current \
architectural belief about this codebase as a JSON object matching the \
schema below.  Output ONLY the JSON object — no commentary before or after.

### Schema

```json
{
  "step": <int — current action step>,
  "components": {
    "<filepath>": {
      "filepath": "<filepath>",
      "status": "observed" | "inferred" | "unknown",
      "purpose": "<why this file exists>",
      "exports": [
        {
          "name": "<function or class name>",
          "signature": {
            "params": [
              {"name": "<param>", "type_hint": "<type>", "has_default": false}
            ],
            "return_type": "<type>"
          },
          "callers": ["<filepath of caller>"],
          "confidence": 0.0-1.0
        }
      ],
      "edges": [
        {
          "target": "<target filepath>",
          "type": "IMPORTS" | "CALLS_API" | "DATA_FLOWS_TO" | "REGISTRY_WIRES",
          "confidence": 0.0-1.0
        }
      ],
      "confidence": 0.0-1.0
    }
  },
  "invariants": [
    {
      "type": "BOUNDARY" | "DATAFLOW" | "INTERFACE" | "INVARIANT" | "PURPOSE",
      "description": "<what the constraint says>",
      "structured": {
        "type": "FORBIDDEN_EDGE" | "INTERFACE_ONLY" | "VALIDATION_CHAIN" | "NAMING_CONVENTION",
        "src": "<source file or null>",
        "dst": "<target file or null>",
        "via": "<intermediary or null>",
        "pattern": "<regex or null>"
      },
      "evidence": ["<filepath or observation that supports this>"],
      "confidence": 0.0-1.0
    }
  ],
  "unexplored": ["<filepath or directory you haven't examined>"],
  "uncertainty_summary": "<brief description of what you're unsure about>"
}
```

### Edge types
- **IMPORTS**: a Python `import` or `from X import Y` statement exists.
- **CALLS_API**: the file calls a public function exported by the target.
- **DATA_FLOWS_TO**: the output of this component is consumed by the target \
(e.g., return value passed as argument in an orchestrator).
- **REGISTRY_WIRES**: connected via a config file or registry, not a direct import.

### Example (partial)

```json
{
  "step": 6,
  "components": {
    "models.py": {
      "filepath": "models.py",
      "status": "observed",
      "purpose": "Shared data types used across pipeline stages",
      "exports": [
        {
          "name": "Record",
          "signature": {"params": [], "return_type": ""},
          "callers": ["mod_a.py", "mod_b.py"],
          "confidence": 0.9
        }
      ],
      "edges": [],
      "confidence": 0.95
    },
    "registry.py": {
      "filepath": "registry.py",
      "status": "observed",
      "purpose": "Loads pipeline stages from config at runtime",
      "exports": [
        {
          "name": "get_pipeline",
          "signature": {
            "params": [],
            "return_type": "list[StageBase]"
          },
          "callers": ["runner.py"],
          "confidence": 0.8
        }
      ],
      "edges": [
        {"target": "base.py", "type": "IMPORTS", "confidence": 0.9},
        {"target": "mod_a.py", "type": "REGISTRY_WIRES", "confidence": 0.7}
      ],
      "confidence": 0.8
    }
  },
  "invariants": [
    {
      "type": "BOUNDARY",
      "description": "Stages must not import each other directly",
      "structured": {
        "type": "FORBIDDEN_EDGE",
        "src": null,
        "dst": null,
        "via": null,
        "pattern": "no stage-to-stage imports"
      },
      "evidence": ["test_boundaries.py"],
      "confidence": 0.7
    }
  ],
  "unexplored": ["legacy.py", "helpers.py"],
  "uncertainty_summary": "Have not yet opened helpers.py or legacy.py. Unsure whether there are additional stages beyond mod_a."
}
```

Now output your belief as JSON:
"""


# ============================================================================
# Parser
# ============================================================================


@dataclass
class ParseResult:
    """Result of attempting to parse a CognitiveMap from model output."""

    success: bool
    map: Optional[CognitiveMap] = None
    raw_json: Optional[str] = None
    error: Optional[str] = None


def parse_cognitive_map(text: str, step: int = 0) -> ParseResult:
    """Extract a CognitiveMap from model output using fuzzy parsing.

    Tries in order:
    1. Direct json.loads on the full text
    2. Strip markdown code fences and parse
    3. Find the outermost JSON object boundaries and parse
    4. Give up and return error
    """
    best_error: ParseResult | None = None

    # 1. Try direct parse
    result = _try_parse(text, step)
    if result.success:
        return result
    if result.raw_json is not None:
        # JSON parsed but schema validation failed — most informative error
        return result
    best_error = result

    # 2. Strip markdown fences
    stripped = _strip_fences(text)
    if stripped != text:
        result = _try_parse(stripped, step)
        if result.success:
            return result
        if result.raw_json is not None:
            return result
        best_error = result

    # 3. Find JSON object boundaries
    extracted = _extract_json_object(text)
    if extracted:
        result = _try_parse(extracted, step)
        if result.success:
            return result
        if result.raw_json is not None:
            return result
        best_error = result

    # 4. Try to repair truncated JSON (model hit max_tokens)
    repaired = _repair_truncated_json(stripped if stripped != text else text)
    if repaired:
        result = _try_parse(repaired, step)
        if result.success:
            return result
        if result.raw_json is not None:
            return result
        if best_error is None:
            best_error = result

    # 5. Failure — return the most informative error we collected
    if best_error and best_error.error:
        return best_error
    return ParseResult(
        success=False,
        error=f"Could not extract valid CognitiveMap JSON from model output. "
        f"Text length: {len(text)} chars.",
    )


def _try_parse(text: str, step: int) -> ParseResult:
    """Try to parse text as JSON and validate as CognitiveMap."""
    text = text.strip()
    if not text:
        return ParseResult(success=False, error="Empty text")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return ParseResult(success=False, error=f"JSON decode error: {e}")

    if not isinstance(data, dict):
        return ParseResult(success=False, error="Parsed JSON is not an object")

    # Fill in step if missing
    if "step" not in data:
        data["step"] = step

    # Fix truncated invariants (from JSON repair)
    for inv in data.get("invariants", []):
        if isinstance(inv, dict) and not inv.get("description"):
            inv["description"] = "(truncated)"

    # Fix bare-string params: "stage_name" → {"name": "stage_name", ...}
    for comp in data.get("components", {}).values():
        if not isinstance(comp, dict):
            continue
        for exp in comp.get("exports", []):
            if not isinstance(exp, dict):
                continue
            sig = exp.get("signature")
            if not isinstance(sig, dict):
                continue
            params = sig.get("params", [])
            for i, p in enumerate(params):
                if isinstance(p, str):
                    params[i] = {
                        "name": p,
                        "type_hint": "",
                        "has_default": False,
                    }

    try:
        cmap = CognitiveMap.model_validate(data)
    except ValidationError as e:
        return ParseResult(
            success=False,
            raw_json=text,
            error=f"Pydantic validation error: {e}",
        )

    return ParseResult(success=True, map=cmap, raw_json=text)


def _strip_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ``` or ``` ... ```)."""
    # Match ```json\n...\n``` or ```\n...\n```
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Handle unclosed fences (truncated model output)
    unclosed = re.match(r"^\s*```(?:json)?\s*\n?(.*)", text, re.DOTALL)
    if unclosed:
        return unclosed.group(1).strip()

    return text


def _repair_truncated_json(text: str) -> str | None:
    """Attempt to repair JSON truncated by max_tokens.

    Strategy: find the opening '{', split into lines, and progressively
    remove trailing lines until the JSON can be closed with matching
    brackets/braces. This discards incomplete trailing entries but
    recovers the partial cognitive map.
    """
    start = text.find("{")
    if start == -1:
        return None

    json_text = text[start:]
    lines = json_text.split("\n")

    # Try progressively removing lines from the end
    for drop in range(min(len(lines), 20)):
        candidate_lines = lines[: len(lines) - drop] if drop > 0 else lines
        candidate = "\n".join(candidate_lines)

        # Strip trailing partial tokens (mid-value, trailing comma)
        candidate = re.sub(r',\s*$', '', candidate)

        # Count open/close delimiters outside of strings
        stack: list[str] = []
        in_string = False
        escape = False
        for ch in candidate:
            if escape:
                escape = False
                continue
            if ch == "\\":
                if in_string:
                    escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                stack.append("{")
            elif ch == "[":
                stack.append("[")
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

        if in_string:
            # Mid-string truncation — skip this cut point
            continue

        if not stack:
            return None  # Already balanced

        # Close all open delimiters
        suffix = ""
        for delim in reversed(stack):
            suffix += "}" if delim == "{" else "]"

        repaired = candidate + suffix
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            continue

    return None


def _extract_json_object(text: str) -> str | None:
    """Find the outermost { ... } in the text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ============================================================================
# Differ
# ============================================================================


@dataclass
class MapDiff:
    """Difference between two CognitiveMaps."""

    added_components: list[str] = field(default_factory=list)
    removed_components: list[str] = field(default_factory=list)
    changed_components: list[str] = field(default_factory=list)

    added_edges: list[tuple[str, str, str]] = field(default_factory=list)
    removed_edges: list[tuple[str, str, str]] = field(default_factory=list)

    added_invariants: int = 0
    removed_invariants: int = 0

    @property
    def has_changes(self) -> bool:
        return bool(
            self.added_components
            or self.removed_components
            or self.changed_components
            or self.added_edges
            or self.removed_edges
            or self.added_invariants
            or self.removed_invariants
        )


def diff_maps(old: CognitiveMap, new: CognitiveMap) -> MapDiff:
    """Compare two CognitiveMaps and return the differences."""
    diff = MapDiff()

    old_keys = set(old.components.keys())
    new_keys = set(new.components.keys())

    diff.added_components = sorted(new_keys - old_keys)
    diff.removed_components = sorted(old_keys - new_keys)

    # Changed components: same key, different content
    for key in old_keys & new_keys:
        old_comp = old.components[key]
        new_comp = new.components[key]
        if (
            old_comp.purpose != new_comp.purpose
            or old_comp.status != new_comp.status
            or old_comp.confidence != new_comp.confidence
            or _edge_set(old_comp.edges) != _edge_set(new_comp.edges)
        ):
            diff.changed_components.append(key)
    diff.changed_components.sort()

    # Edge diffs (across all components)
    old_edges = _all_edges(old)
    new_edges = _all_edges(new)
    diff.added_edges = sorted(new_edges - old_edges)
    diff.removed_edges = sorted(old_edges - new_edges)

    # Invariant count diff
    old_inv = {(inv.type, inv.description) for inv in old.invariants}
    new_inv = {(inv.type, inv.description) for inv in new.invariants}
    diff.added_invariants = len(new_inv - old_inv)
    diff.removed_invariants = len(old_inv - new_inv)

    return diff


def _edge_set(edges: list) -> set[tuple[str, str]]:
    """Convert a list of BeliefEdge to a set of (target, type) tuples."""
    return {(e.target, e.type.value if hasattr(e.type, "value") else e.type) for e in edges}


def _all_edges(cmap: CognitiveMap) -> set[tuple[str, str, str]]:
    """Collect all (source, target, type) triples from a cognitive map."""
    edges: set[tuple[str, str, str]] = set()
    for filepath, comp in cmap.components.items():
        for e in comp.edges:
            etype = e.type.value if hasattr(e.type, "value") else e.type
            edges.add((filepath, e.target, etype))
    return edges
