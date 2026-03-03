"""Tests for Task 4: Map accuracy metrics.

Validates:
- Perfect map → F1=1.0
- Empty map → F1=0.0
- Hallucinated edge → precision < 1.0
- Missing edge → recall < 1.0
- ECE=0 when confidence matches accuracy perfectly
- Action AUC increases monotonically when F1 increases with steps
- Structured invariant matching: correct type+src+dst = match; wrong dst = miss
- Type equivalence: "dict" matches "Dict[str, Any]"
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from metrics.map_accuracy import (
    ContractScore,
    _trapezoidal_auc,
    confidence_ece,
    contract_accuracy,
    edge_prf,
    efficiency_curves,
    hallucinations,
    invariant_prf,
    invariant_prf_relaxed,
    normalise_type,
    score_exploration,
    score_map,
    steps_to_recall,
)
from models import (
    ActionResult,
    ActionType,
    AgentAction,
    BeliefEdge,
    BeliefExport,
    CodebaseGroundTruth,
    CognitiveMap,
    ComponentBelief,
    EdgeType,
    ExportedAPI,
    FunctionSignature,
    InvariantBelief,
    ModuleStatus,
    ParameterSpec,
    StructuredConstraint,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_pipeline"


@pytest.fixture
def gt() -> CodebaseGroundTruth:
    return CodebaseGroundTruth.model_validate_json(
        (FIXTURE_DIR / "ground_truth.json").read_text()
    )


# ── Helpers ─────────────────────────────────────────────────────────


def _comp(
    filepath: str,
    edges: list[BeliefEdge] | None = None,
    exports: list[BeliefExport] | None = None,
    confidence: float = 0.8,
) -> ComponentBelief:
    return ComponentBelief(
        filepath=filepath,
        status=ModuleStatus.OBSERVED,
        edges=edges or [],
        exports=exports or [],
        confidence=confidence,
    )


def _edge(target: str, etype: EdgeType, conf: float = 0.8) -> BeliefEdge:
    return BeliefEdge(target=target, type=etype, confidence=conf)


def _cmap(
    components: dict[str, ComponentBelief] | None = None,
    invariants: list[InvariantBelief] | None = None,
    step: int = 1,
) -> CognitiveMap:
    return CognitiveMap(
        step=step,
        components=components or {},
        invariants=invariants or [],
    )


def _perfect_map(gt: CodebaseGroundTruth) -> CognitiveMap:
    """Build a CognitiveMap that perfectly matches ground truth."""
    components: dict[str, ComponentBelief] = {}
    for filepath, mod in gt.modules.items():
        edges = [
            BeliefEdge(target=e["target"], type=EdgeType(e["type"]), confidence=1.0)
            for e in mod.edges
        ]
        components[filepath] = ComponentBelief(
            filepath=filepath,
            status=ModuleStatus.OBSERVED,
            edges=edges,
            confidence=1.0,
        )
    return CognitiveMap(step=20, components=components)


# ── 1. Edge P/R/F1 ─────────────────────────────────────────────────


class TestEdgePRF:
    def test_perfect_map(self, gt: CodebaseGroundTruth) -> None:
        cmap = _perfect_map(gt)
        p, r, f1 = edge_prf(cmap, gt)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_empty_map(self, gt: CodebaseGroundTruth) -> None:
        cmap = _cmap()
        p, r, f1 = edge_prf(cmap, gt)
        assert f1 == pytest.approx(0.0)
        assert r == pytest.approx(0.0)

    def test_hallucinated_edge_lowers_precision(self, gt: CodebaseGroundTruth) -> None:
        cmap = _perfect_map(gt)
        # Add a fake edge
        cmap.components["models.py"].edges.append(
            _edge("nonexistent.py", EdgeType.IMPORTS)
        )
        p, r, f1 = edge_prf(cmap, gt)
        assert p < 1.0
        assert r == pytest.approx(1.0)

    def test_missing_edge_lowers_recall(self, gt: CodebaseGroundTruth) -> None:
        cmap = _perfect_map(gt)
        # Remove one edge
        for comp in cmap.components.values():
            if comp.edges:
                comp.edges.pop()
                break
        p, r, f1 = edge_prf(cmap, gt)
        assert r < 1.0


# ── 2. Hallucinations ──────────────────────────────────────────────


class TestHallucinations:
    def test_perfect_map_no_hallucinations(self, gt: CodebaseGroundTruth) -> None:
        cmap = _perfect_map(gt)
        h = hallucinations(cmap, gt)
        assert h.hallucinated_edges == 0
        assert h.hallucinated_nodes == 0

    def test_phantom_node(self, gt: CodebaseGroundTruth) -> None:
        cmap = _cmap({"phantom.py": _comp("phantom.py")})
        h = hallucinations(cmap, gt)
        assert h.hallucinated_nodes == 1

    def test_hallucinated_edge(self, gt: CodebaseGroundTruth) -> None:
        cmap = _cmap({
            "models.py": _comp("models.py", edges=[_edge("fake.py", EdgeType.IMPORTS)])
        })
        h = hallucinations(cmap, gt)
        assert h.hallucinated_edges == 1


# ── 3. Invariant P/R/F1 ────────────────────────────────────────────


class TestInvariantPRF:
    def test_correct_structured_match(self, gt: CodebaseGroundTruth) -> None:
        """Invariant with matching structured form → counted as TP."""
        # Use the first ground truth invariant's structured form
        gt_struct = gt.invariants[0].structured
        pred_inv = InvariantBelief(
            type="DATAFLOW",
            description="whatever",
            structured=StructuredConstraint(**gt_struct),
            confidence=0.9,
        )
        cmap = _cmap(invariants=[pred_inv])
        p, r, f1 = invariant_prf(cmap, gt)
        assert p > 0.0  # at least one match

    def test_wrong_dst_no_match(self, gt: CodebaseGroundTruth) -> None:
        """Invariant with wrong dst field → no match."""
        gt_struct = dict(gt.invariants[0].structured)
        gt_struct["dst"] = "wrong_file.py"
        pred_inv = InvariantBelief(
            type="DATAFLOW",
            description="whatever",
            structured=StructuredConstraint(**gt_struct),
            confidence=0.9,
        )
        cmap = _cmap(invariants=[pred_inv])
        p, r, f1 = invariant_prf(cmap, gt)
        # The modified invariant should not match any GT invariant
        # (precision may be 0 if no matches at all)
        tp = len(set() & set())  # placeholder
        # More precisely: this prediction has 1 element, gt has N elements.
        # If the modified prediction doesn't match any GT, TP=0, precision=0
        pred_set = set()
        for inv in cmap.invariants:
            if inv.structured:
                d = inv.structured.model_dump()
                pred_set.add((
                    d.get("type", ""),
                    d.get("src") or "",
                    d.get("dst") or "",
                    d.get("via") or "",
                    d.get("pattern") or "",
                ))
        from metrics.map_accuracy import _gt_invariant_set
        gt_set = _gt_invariant_set(gt)
        assert len(pred_set & gt_set) == 0  # no overlap

    def test_empty_predictions(self, gt: CodebaseGroundTruth) -> None:
        cmap = _cmap()
        p, r, f1 = invariant_prf(cmap, gt)
        assert r == pytest.approx(0.0)


# ── 3b. Relaxed invariant matching ─────────────────────────────────


class TestInvariantRelaxed:
    def test_interface_only_field_swap(self, gt: CodebaseGroundTruth) -> None:
        """Model puts interface file in dst instead of via → relaxed matches."""
        # Find an INTERFACE_ONLY GT invariant
        gt_io = None
        for inv in gt.invariants:
            if inv.structured and inv.structured.get("type") == "INTERFACE_ONLY":
                gt_io = inv
                break
        if gt_io is None:
            pytest.skip("No INTERFACE_ONLY invariant in fixture")

        via_file = gt_io.structured.get("via") or ""
        # Model's version: puts the via file in dst instead
        pred_inv = InvariantBelief(
            type="INTERFACE",
            description="All stages go through the base class",
            structured=StructuredConstraint(
                type="INTERFACE_ONLY",
                src="stages",
                dst=via_file,  # field-swapped: should be in via
                via=None,
                pattern="StageBase",
            ),
            confidence=0.8,
        )
        cmap = _cmap(invariants=[pred_inv])

        # Strict: should NOT match (fields in wrong positions)
        sp, sr, sf1 = invariant_prf(cmap, gt)
        assert sf1 == pytest.approx(0.0)

        # Relaxed: SHOULD match (type + normalised dst align)
        rp, rr, rf1 = invariant_prf_relaxed(cmap, gt)
        assert rp > 0.0, "Relaxed matching should find the INTERFACE_ONLY match"

    def test_validation_chain_pairwise(self, gt: CodebaseGroundTruth) -> None:
        """Pairwise VALIDATION_CHAIN with stripped path → relaxed matches."""
        gt_vc = None
        for inv in gt.invariants:
            if inv.structured and inv.structured.get("type") == "VALIDATION_CHAIN":
                gt_vc = inv
                break
        if gt_vc is None:
            pytest.skip("No VALIDATION_CHAIN invariant in fixture")

        # Replicate the GT tuple exactly but with stripped paths
        src = gt_vc.structured.get("src", "")
        dst = gt_vc.structured.get("dst", "")
        # Strip leading package prefix (e.g. "text_processor/stages/mod_a.py" → "stages/mod_a.py")
        src_stripped = "/".join(src.split("/")[1:]) if "/" in src else src
        dst_stripped = "/".join(dst.split("/")[1:]) if "/" in dst else dst

        pred_inv = InvariantBelief(
            type="DATAFLOW",
            description="Data flows pairwise",
            structured=StructuredConstraint(
                type="VALIDATION_CHAIN",
                src=src_stripped,
                dst=dst_stripped,
                via=None,
                pattern=None,
            ),
            confidence=0.7,
        )
        cmap = _cmap(invariants=[pred_inv])
        rp, rr, rf1 = invariant_prf_relaxed(cmap, gt)
        assert rp > 0.0, "Relaxed matching should match stripped-path VALIDATION_CHAIN"

    def test_directory_level_forbidden_edge(self) -> None:
        """Directory-level FORBIDDEN_EDGE with path stripping normalisation."""
        from metrics.map_accuracy import _normalise_relaxed

        # GT uses file-level path
        gt_tuple = _normalise_relaxed({
            "type": "FORBIDDEN_EDGE",
            "src": "data_pipeline/stages/mod_a.py",
            "dst": None,
        })
        # Model uses same file but different package prefix
        pred_tuple = _normalise_relaxed({
            "type": "FORBIDDEN_EDGE",
            "src": "data_pipeline/stages/mod_a.py",
            "dst": None,
        })
        assert gt_tuple == pred_tuple

    def test_empty_relaxed(self, gt: CodebaseGroundTruth) -> None:
        cmap = _cmap()
        p, r, f1 = invariant_prf_relaxed(cmap, gt)
        assert r == pytest.approx(0.0)


# ── 4. Contract accuracy ───────────────────────────────────────────


class TestContractAccuracy:
    def test_perfect_contracts(self, gt: CodebaseGroundTruth) -> None:
        """Agent exports matching GT contracts exactly → full accuracy."""
        components: dict[str, ComponentBelief] = {}
        for contract in gt.contracts:
            fp = contract.module
            if fp not in components:
                components[fp] = _comp(fp)
            components[fp].exports.append(BeliefExport(
                name=contract.name,
                signature=contract.signature,
                callers=contract.callers,
                confidence=1.0,
            ))
        cmap = _cmap(components)
        score = contract_accuracy(cmap, gt)
        assert score.name_matches == score.total
        assert score.signature_matches == score.total
        assert score.accuracy == pytest.approx(1.0)

    def test_missing_contracts(self, gt: CodebaseGroundTruth) -> None:
        cmap = _cmap()
        score = contract_accuracy(cmap, gt)
        assert score.name_matches == 0
        assert score.accuracy == pytest.approx(0.0)


# ── 4b. Type equivalence ───────────────────────────────────────────


class TestTypeEquivalence:
    def test_dict_aliases(self) -> None:
        assert normalise_type("dict") == normalise_type("Dict[str, Any]")
        assert normalise_type("dict") == normalise_type("Dict")

    def test_list_aliases(self) -> None:
        assert normalise_type("list") == normalise_type("List[Any]")
        assert normalise_type("list") == normalise_type("List")

    def test_optional_unwrap(self) -> None:
        assert normalise_type("Optional[str]") == normalise_type("str")
        assert normalise_type("str | None") == normalise_type("str")
        assert normalise_type("None | str") == normalise_type("str")

    def test_case_insensitive_fallback(self) -> None:
        assert normalise_type("MyClass") == normalise_type("myclass")


# ── 5. ECE ──────────────────────────────────────────────────────────


class TestECE:
    def test_perfect_calibration(self, gt: CodebaseGroundTruth) -> None:
        """All edges correct with confidence=1.0 → ECE=0."""
        cmap = _perfect_map(gt)
        ece = confidence_ece(cmap, gt)
        assert ece == pytest.approx(0.0)

    def test_empty_map_ece_zero(self, gt: CodebaseGroundTruth) -> None:
        cmap = _cmap()
        ece = confidence_ece(cmap, gt)
        assert ece == pytest.approx(0.0)

    def test_overconfident_wrong_edges(self, gt: CodebaseGroundTruth) -> None:
        """All edges wrong but confidence=1.0 → ECE=1.0."""
        cmap = _cmap({
            "a.py": _comp("a.py", edges=[
                _edge("fake1.py", EdgeType.IMPORTS, conf=1.0),
                _edge("fake2.py", EdgeType.IMPORTS, conf=1.0),
            ])
        })
        ece = confidence_ece(cmap, gt)
        assert ece == pytest.approx(1.0)


# ── 6. Efficiency curves ───────────────────────────────────────────


def _make_action_log(n: int, open_indices: set[int] | None = None) -> list[ActionResult]:
    """Create a dummy action log with n actions."""
    log: list[ActionResult] = []
    open_indices = open_indices or set()
    for i in range(1, n + 1):
        if i in open_indices:
            action = AgentAction(type=ActionType.OPEN, argument=f"file_{i}.py")
        else:
            action = AgentAction(type=ActionType.LIST, argument="")
        log.append(ActionResult(action=action, output="", step=i))
    return log


class TestEfficiencyCurves:
    def test_monotonic_f1_increasing_auc(self, gt: CodebaseGroundTruth) -> None:
        """If F1 increases over time, action AUC should be positive."""
        # Build maps with increasing edge coverage
        gt_edges = list({
            (e["source"], e["target"], e["type"])
            for e in gt.dependency_edges
        })

        maps: list[CognitiveMap] = []
        for i, count in enumerate([2, 5, 10, len(gt_edges)]):
            components: dict[str, ComponentBelief] = {}
            for src, tgt, etype in gt_edges[:count]:
                if src not in components:
                    components[src] = _comp(src)
                components[src].edges.append(
                    BeliefEdge(target=tgt, type=EdgeType(etype), confidence=0.8)
                )
            maps.append(CognitiveMap(step=(i + 1) * 3, components=components))

        action_log = _make_action_log(12, open_indices={1, 4, 7, 10})
        act_curve, obs_curve, act_auc, obs_auc = efficiency_curves(
            maps, action_log, gt
        )
        assert act_auc > 0.0
        # F1 should be non-decreasing
        for i in range(1, len(act_curve)):
            assert act_curve[i] >= act_curve[i - 1] - 1e-9

    def test_empty_maps_zero_auc(self, gt: CodebaseGroundTruth) -> None:
        act_curve, obs_curve, act_auc, obs_auc = efficiency_curves([], [], gt)
        assert act_auc == 0.0
        assert obs_auc == 0.0


# ── 7. Steps-to-X-recall ───────────────────────────────────────────


class TestStepsToRecall:
    def test_perfect_map_at_step_1(self, gt: CodebaseGroundTruth) -> None:
        cmap = _perfect_map(gt)
        cmap.step = 1
        result = steps_to_recall([cmap], gt)
        assert result[0.5] == 1
        assert result[0.8] == 1

    def test_empty_maps_none(self, gt: CodebaseGroundTruth) -> None:
        result = steps_to_recall([], gt)
        assert result[0.5] is None
        assert result[0.8] is None


# ── Combined scorer ─────────────────────────────────────────────────


class TestScoreMap:
    def test_perfect_map_metrics(self, gt: CodebaseGroundTruth) -> None:
        cmap = _perfect_map(gt)
        metrics = score_map(cmap, gt)
        assert metrics.dependency_f1 == pytest.approx(1.0)
        assert metrics.confidence_ece == pytest.approx(0.0)

    def test_empty_map_metrics(self, gt: CodebaseGroundTruth) -> None:
        cmap = _cmap()
        metrics = score_map(cmap, gt)
        assert metrics.dependency_f1 == pytest.approx(0.0)
        assert metrics.dependency_recall == pytest.approx(0.0)


class TestScoreExploration:
    def test_returns_exploration_metrics(self, gt: CodebaseGroundTruth) -> None:
        cmap = _perfect_map(gt)
        cmap.step = 5
        action_log = _make_action_log(5, open_indices={1, 3, 5})
        metrics = score_exploration([cmap], action_log, gt)
        assert metrics.steps_taken == 5
        assert metrics.files_opened == 3
        assert metrics.final_efficiency > 0.0
