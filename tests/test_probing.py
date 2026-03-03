"""Tests for Task 3: Cognitive map probing — parser and differ.

Validates:
- Clean JSON parses correctly
- JSON wrapped in ```json fences parses
- JSON with surrounding commentary parses
- Invalid JSON returns appropriate error
- Parsed result validates as CognitiveMap
- Differ correctly identifies added/removed/changed components and edges
"""

from __future__ import annotations

import json

import pytest

from harness.probing import (
    PROBE_PROMPT,
    MapDiff,
    ParseResult,
    diff_maps,
    parse_cognitive_map,
)
from models import (
    BeliefEdge,
    CognitiveMap,
    ComponentBelief,
    EdgeType,
    InvariantBelief,
    ModuleStatus,
)

# ── Helpers ─────────────────────────────────────────────────────────

MINIMAL_MAP = {
    "step": 3,
    "components": {
        "models.py": {
            "filepath": "models.py",
            "status": "observed",
            "purpose": "Shared data types",
            "exports": [],
            "edges": [],
            "confidence": 0.9,
        }
    },
    "invariants": [],
    "unexplored": ["legacy.py"],
    "uncertainty_summary": "Haven't seen everything yet.",
}

RICH_MAP = {
    "step": 6,
    "components": {
        "models.py": {
            "filepath": "models.py",
            "status": "observed",
            "purpose": "Shared data types",
            "exports": [
                {
                    "name": "Record",
                    "signature": {"params": [], "return_type": ""},
                    "callers": ["mod_a.py"],
                    "confidence": 0.9,
                }
            ],
            "edges": [],
            "confidence": 0.95,
        },
        "registry.py": {
            "filepath": "registry.py",
            "status": "observed",
            "purpose": "Loads pipeline stages from config",
            "exports": [],
            "edges": [
                {"target": "base.py", "type": "IMPORTS", "confidence": 0.9},
                {"target": "mod_a.py", "type": "REGISTRY_WIRES", "confidence": 0.7},
            ],
            "confidence": 0.8,
        },
    },
    "invariants": [
        {
            "type": "BOUNDARY",
            "description": "Stages must not import each other",
            "evidence": ["test_boundaries.py"],
            "confidence": 0.7,
        }
    ],
    "unexplored": ["helpers.py"],
    "uncertainty_summary": "Haven't opened helpers.py yet.",
}


def _json_str(data: dict) -> str:
    return json.dumps(data, indent=2)


# ── Parse: clean JSON ──────────────────────────────────────────────


class TestParseClean:
    def test_minimal_map(self) -> None:
        result = parse_cognitive_map(_json_str(MINIMAL_MAP))
        assert result.success
        assert result.map is not None
        assert result.map.step == 3
        assert "models.py" in result.map.components

    def test_rich_map(self) -> None:
        result = parse_cognitive_map(_json_str(RICH_MAP))
        assert result.success
        assert result.map is not None
        assert len(result.map.components) == 2
        assert len(result.map.invariants) == 1

    def test_validates_as_cognitive_map(self) -> None:
        result = parse_cognitive_map(_json_str(RICH_MAP))
        assert result.success
        cmap = result.map
        assert isinstance(cmap, CognitiveMap)
        reg = cmap.components["registry.py"]
        assert reg.edges[0].type == EdgeType.IMPORTS

    def test_step_filled_if_missing(self) -> None:
        data = {**MINIMAL_MAP}
        del data["step"]
        result = parse_cognitive_map(_json_str(data), step=7)
        assert result.success
        assert result.map is not None
        assert result.map.step == 7


# ── Parse: markdown fences ─────────────────────────────────────────


class TestParseFences:
    def test_json_fences(self) -> None:
        text = f"```json\n{_json_str(MINIMAL_MAP)}\n```"
        result = parse_cognitive_map(text)
        assert result.success
        assert result.map is not None
        assert result.map.step == 3

    def test_plain_fences(self) -> None:
        text = f"```\n{_json_str(MINIMAL_MAP)}\n```"
        result = parse_cognitive_map(text)
        assert result.success

    def test_fences_with_trailing_whitespace(self) -> None:
        text = f"```json  \n{_json_str(MINIMAL_MAP)}\n```  "
        result = parse_cognitive_map(text)
        assert result.success


# ── Parse: surrounding commentary ──────────────────────────────────


class TestParseCommentary:
    def test_commentary_before(self) -> None:
        text = f"Here is my belief state:\n\n{_json_str(MINIMAL_MAP)}"
        result = parse_cognitive_map(text)
        assert result.success

    def test_commentary_after(self) -> None:
        text = f"{_json_str(MINIMAL_MAP)}\n\nI'm still uncertain about helpers.py."
        result = parse_cognitive_map(text)
        assert result.success

    def test_commentary_both_sides(self) -> None:
        text = (
            "Based on my observations so far:\n\n"
            f"{_json_str(MINIMAL_MAP)}\n\n"
            "I'll need to explore more files."
        )
        result = parse_cognitive_map(text)
        assert result.success
        assert result.map is not None
        assert result.map.step == 3

    def test_fences_with_commentary(self) -> None:
        text = (
            "Here is my current map:\n\n"
            f"```json\n{_json_str(RICH_MAP)}\n```\n\n"
            "Some notes about what I found."
        )
        result = parse_cognitive_map(text)
        assert result.success
        assert result.map is not None
        assert len(result.map.components) == 2


# ── Parse: failures ────────────────────────────────────────────────


class TestParseFailures:
    def test_empty_string(self) -> None:
        result = parse_cognitive_map("")
        assert not result.success
        assert result.error is not None

    def test_no_json(self) -> None:
        result = parse_cognitive_map("I haven't observed anything yet.")
        assert not result.success

    def test_invalid_json(self) -> None:
        result = parse_cognitive_map('{"step": 1, "components": {broken}')
        assert not result.success

    def test_valid_json_wrong_schema(self) -> None:
        result = parse_cognitive_map('{"step": "not_an_int"}')
        assert not result.success
        assert result.error is not None
        assert "validation" in result.error.lower() or "Pydantic" in result.error

    def test_json_array_not_object(self) -> None:
        result = parse_cognitive_map("[1, 2, 3]")
        assert not result.success


# ── Differ ──────────────────────────────────────────────────────────


def _make_map(
    step: int,
    components: dict[str, ComponentBelief] | None = None,
    invariants: list[InvariantBelief] | None = None,
) -> CognitiveMap:
    return CognitiveMap(
        step=step,
        components=components or {},
        invariants=invariants or [],
    )


def _comp(
    filepath: str,
    status: ModuleStatus = ModuleStatus.OBSERVED,
    purpose: str = "",
    edges: list[BeliefEdge] | None = None,
    confidence: float = 0.5,
) -> ComponentBelief:
    return ComponentBelief(
        filepath=filepath,
        status=status,
        purpose=purpose,
        edges=edges or [],
        confidence=confidence,
    )


def _edge(target: str, etype: EdgeType, conf: float = 0.5) -> BeliefEdge:
    return BeliefEdge(target=target, type=etype, confidence=conf)


class TestDiffer:
    def test_no_changes(self) -> None:
        m = _make_map(1, {"a.py": _comp("a.py")})
        diff = diff_maps(m, m)
        assert not diff.has_changes

    def test_added_component(self) -> None:
        old = _make_map(1, {"a.py": _comp("a.py")})
        new = _make_map(2, {
            "a.py": _comp("a.py"),
            "b.py": _comp("b.py"),
        })
        diff = diff_maps(old, new)
        assert diff.added_components == ["b.py"]
        assert diff.removed_components == []
        assert diff.has_changes

    def test_removed_component(self) -> None:
        old = _make_map(1, {"a.py": _comp("a.py"), "b.py": _comp("b.py")})
        new = _make_map(2, {"a.py": _comp("a.py")})
        diff = diff_maps(old, new)
        assert diff.removed_components == ["b.py"]

    def test_changed_component_purpose(self) -> None:
        old = _make_map(1, {"a.py": _comp("a.py", purpose="old")})
        new = _make_map(2, {"a.py": _comp("a.py", purpose="new")})
        diff = diff_maps(old, new)
        assert "a.py" in diff.changed_components

    def test_changed_component_status(self) -> None:
        old = _make_map(1, {"a.py": _comp("a.py", status=ModuleStatus.INFERRED)})
        new = _make_map(2, {"a.py": _comp("a.py", status=ModuleStatus.OBSERVED)})
        diff = diff_maps(old, new)
        assert "a.py" in diff.changed_components

    def test_added_edge(self) -> None:
        old = _make_map(1, {"a.py": _comp("a.py")})
        new = _make_map(2, {
            "a.py": _comp("a.py", edges=[_edge("b.py", EdgeType.IMPORTS)]),
        })
        diff = diff_maps(old, new)
        assert ("a.py", "b.py", "IMPORTS") in diff.added_edges

    def test_removed_edge(self) -> None:
        old = _make_map(1, {
            "a.py": _comp("a.py", edges=[_edge("b.py", EdgeType.IMPORTS)]),
        })
        new = _make_map(2, {"a.py": _comp("a.py")})
        diff = diff_maps(old, new)
        assert ("a.py", "b.py", "IMPORTS") in diff.removed_edges

    def test_added_invariant(self) -> None:
        inv = InvariantBelief(type="BOUNDARY", description="no cross imports")
        old = _make_map(1)
        new = _make_map(2, invariants=[inv])
        diff = diff_maps(old, new)
        assert diff.added_invariants == 1
        assert diff.removed_invariants == 0

    def test_complex_diff(self) -> None:
        """Multiple changes at once."""
        old = _make_map(3, {
            "a.py": _comp("a.py", edges=[_edge("b.py", EdgeType.IMPORTS)]),
            "b.py": _comp("b.py"),
        })
        new = _make_map(6, {
            "a.py": _comp("a.py", edges=[
                _edge("b.py", EdgeType.IMPORTS),
                _edge("c.py", EdgeType.CALLS_API),
            ]),
            "c.py": _comp("c.py"),
        })
        diff = diff_maps(old, new)
        assert diff.added_components == ["c.py"]
        assert diff.removed_components == ["b.py"]
        assert ("a.py", "c.py", "CALLS_API") in diff.added_edges
        assert diff.has_changes


# ── Probe prompt ────────────────────────────────────────────────────


class TestProbePrompt:
    def test_prompt_not_empty(self) -> None:
        assert len(PROBE_PROMPT) > 100

    def test_prompt_mentions_edge_types(self) -> None:
        assert "IMPORTS" in PROBE_PROMPT
        assert "CALLS_API" in PROBE_PROMPT
        assert "DATA_FLOWS_TO" in PROBE_PROMPT
        assert "REGISTRY_WIRES" in PROBE_PROMPT

    def test_prompt_has_example(self) -> None:
        assert '"step"' in PROBE_PROMPT
        assert '"components"' in PROBE_PROMPT
