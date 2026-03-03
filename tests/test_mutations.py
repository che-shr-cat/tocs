"""Tests for Task 8: Belief Revision (REVISE phase).

Validates:
- MutationEngine modifies actual files on disk
- ShamMutationEngine does NOT modify any files
- EvidenceGenerator produces realistic output
- Revision scoring: BRS, inertia_proper, impact_discovery, gullibility
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from harness.mutations import (
    EvidenceGenerator,
    MutationEngine,
    MutationResult,
    ShamMutationEngine,
    score_revision,
)
from models import (
    BeliefEdge,
    BeliefExport,
    BeliefRevisionMetrics,
    CodebaseGroundTruth,
    CognitiveMap,
    ComponentBelief,
    EdgeType,
    FunctionSignature,
    ModuleStatus,
    MutationType,
    ParameterSpec,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_pipeline"


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """Copy fixture to a temp directory for safe mutation testing."""
    dst = tmp_path / "codebase"
    shutil.copytree(FIXTURE_DIR, dst)
    return dst


@pytest.fixture
def gt(workdir: Path) -> CodebaseGroundTruth:
    return CodebaseGroundTruth.model_validate_json(
        (workdir / "ground_truth.json").read_text()
    )


# ── Helper to build synthetic beliefs ────────────────────────────────


def _belief(
    edges: list[tuple[str, str, str]] | None = None,
    exports: dict[str, list[BeliefExport]] | None = None,
) -> CognitiveMap:
    """Build a CognitiveMap from edge triples and optional exports."""
    components: dict[str, ComponentBelief] = {}
    for src, tgt, etype in (edges or []):
        if src not in components:
            components[src] = ComponentBelief(
                filepath=src,
                status=ModuleStatus.OBSERVED,
                purpose="",
                edges=[],
                exports=[],
                confidence=0.9,
            )
        components[src].edges.append(
            BeliefEdge(target=tgt, type=EdgeType(etype), confidence=0.9)
        )
        if tgt not in components:
            components[tgt] = ComponentBelief(
                filepath=tgt,
                status=ModuleStatus.INFERRED,
                purpose="",
                edges=[],
                exports=[],
                confidence=0.5,
            )
    if exports:
        for module, export_list in exports.items():
            if module not in components:
                components[module] = ComponentBelief(
                    filepath=module,
                    status=ModuleStatus.OBSERVED,
                    purpose="",
                    edges=[],
                    exports=[],
                    confidence=0.9,
                )
            components[module].exports = export_list
    return CognitiveMap(step=0, components=components)


# ── MutationEngine: file modifications ──────────────────────────────


class TestMutationEngine:
    def test_interface_break_modifies_file(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        engine = MutationEngine(workdir, gt)
        result = engine.apply_interface_break()

        content = (workdir / result.mutation.target_module).read_text()
        assert "strict" in content

    def test_interface_break_returns_valid_mutation(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        engine = MutationEngine(workdir, gt)
        result = engine.apply_interface_break()

        assert result.mutation.type == MutationType.INTERFACE_BREAK
        assert not result.mutation.is_sham
        assert len(result.mutation.affected_modules) >= 1

    def test_interface_break_updates_post_gt(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        engine = MutationEngine(workdir, gt)
        result = engine.apply_interface_break()

        # Post GT contract should have new param
        found = False
        for contract in result.post_gt.contracts:
            if contract.module == result.mutation.target_module:
                for param in contract.signature.params:
                    if param.name == "strict":
                        found = True
        assert found, "Post GT missing the new 'strict' parameter"

    def test_dependency_shift_modifies_files(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        engine = MutationEngine(workdir, gt)
        result = engine.apply_dependency_shift()

        # Importer should now reference dst_module
        imp_content = (workdir / "mod_b.py").read_text()
        assert "mod_d" in imp_content

        # Destination should have the function
        dst_content = (workdir / "mod_d.py").read_text()
        assert "compute_checksum" in dst_content

    def test_boundary_breach_modifies_file(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        engine = MutationEngine(workdir, gt)
        result = engine.apply_boundary_breach()

        content = (workdir / "mod_c.py").read_text()
        assert "mod_a" in content
        assert "IngestStage" in content

    def test_boundary_breach_has_affected_invariants(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        engine = MutationEngine(workdir, gt)
        result = engine.apply_boundary_breach()

        # Should affect the "no stage-to-stage imports" invariant
        assert len(result.mutation.affected_invariants) > 0
        assert "inv-002" in result.mutation.affected_invariants


# ── ShamMutationEngine: no modifications ─────────────────────────────


class TestShamMutation:
    def test_sham_does_not_modify_files(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        originals: dict[str, str] = {}
        for f in workdir.glob("*.py"):
            originals[f.name] = f.read_text()

        engine = ShamMutationEngine(workdir, gt)
        engine.generate_sham(MutationType.INTERFACE_BREAK)

        for f in workdir.glob("*.py"):
            assert f.read_text() == originals[f.name], f"{f.name} was modified!"

    def test_sham_is_sham_flag(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        engine = ShamMutationEngine(workdir, gt)
        result = engine.generate_sham(MutationType.BOUNDARY_BREACH)
        assert result.mutation.is_sham is True

    def test_sham_has_empty_affected(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        engine = ShamMutationEngine(workdir, gt)
        result = engine.generate_sham(MutationType.INTERFACE_BREAK)
        assert result.mutation.affected_modules == []
        assert result.mutation.affected_invariants == []

    def test_sham_post_gt_unchanged(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        engine = ShamMutationEngine(workdir, gt)
        result = engine.generate_sham(MutationType.DEPENDENCY_SHIFT)
        assert len(result.post_gt.dependency_edges) == len(gt.dependency_edges)


# ── EvidenceGenerator ────────────────────────────────────────────────


class TestEvidenceGenerator:
    def test_interface_break_evidence_has_error(self) -> None:
        evidence = EvidenceGenerator.for_interface_break(
            "mod_b.py", "process", ["runner.py"]
        )
        assert "TypeError" in evidence
        assert "process" in evidence
        assert "FAILURES" in evidence

    def test_dependency_shift_evidence_has_import_error(self) -> None:
        evidence = EvidenceGenerator.for_dependency_shift(
            "helpers.py", "mod_d.py", "compute_checksum", "mod_b.py"
        )
        assert "ImportError" in evidence
        assert "compute_checksum" in evidence
        assert "helpers" in evidence

    def test_boundary_breach_evidence_has_assertion(self) -> None:
        evidence = EvidenceGenerator.for_boundary_breach(
            "mod_c.py", "mod_a.py", "stages must not import directly"
        )
        assert "FAILURES" in evidence
        assert "mod_c.py" in evidence
        assert "mod_a.py" in evidence

    def test_evidence_is_nonempty_string(self) -> None:
        for method, args in [
            (EvidenceGenerator.for_interface_break, ("m.py", "fn", ["c.py"])),
            (EvidenceGenerator.for_dependency_shift, ("a.py", "b.py", "fn", "c.py")),
            (EvidenceGenerator.for_boundary_breach, ("a.py", "b.py", "desc")),
        ]:
            result = method(*args)
            assert isinstance(result, str)
            assert len(result) > 50


# ── Revision scoring ────────────────────────────────────────────────


class TestScoreRevision:
    def test_brs_1_when_all_correctly_updated(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        """BRS=1.0 when post-belief correctly reflects the mutation."""
        engine = MutationEngine(workdir, gt)
        result = engine.apply_boundary_breach(
            from_module="mod_c.py", to_module="mod_a.py"
        )

        # Pre-belief: correctly does NOT have mod_c→mod_a edge
        pre = _belief([
            ("mod_c.py", "base.py", "IMPORTS"),
            ("mod_c.py", "models.py", "IMPORTS"),
        ])
        # Post-belief: correctly HAS the new edge
        post = _belief([
            ("mod_c.py", "base.py", "IMPORTS"),
            ("mod_c.py", "models.py", "IMPORTS"),
            ("mod_c.py", "mod_a.py", "IMPORTS"),
        ])

        metrics = score_revision(pre, post, result, gt)
        assert metrics.revision_score == pytest.approx(1.0)

    def test_gullibility_0_when_sham_unchanged(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        """Gullibility=0 when belief doesn't change in sham condition."""
        engine = ShamMutationEngine(workdir, gt)
        result = engine.generate_sham(MutationType.INTERFACE_BREAK)

        belief = _belief([("mod_b.py", "models.py", "IMPORTS")])
        metrics = score_revision(belief, belief, result, gt)
        assert metrics.gullibility_rate == pytest.approx(0.0)

    def test_inertia_1_when_not_updated(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        """inertia_proper=1.0 when pre-known elements NOT updated after evidence."""
        engine = MutationEngine(workdir, gt)
        result = engine.apply_boundary_breach(
            from_module="mod_c.py", to_module="mod_a.py"
        )

        # Pre-belief: correctly no mod_c→mod_a (matches pre_gt)
        pre = _belief([("mod_c.py", "models.py", "IMPORTS")])
        # Post-belief: STILL no mod_c→mod_a (should have it but doesn't → not updated)
        post = _belief([("mod_c.py", "models.py", "IMPORTS")])

        metrics = score_revision(pre, post, result, gt)
        assert metrics.inertia_proper == pytest.approx(1.0)

    def test_inertia_0_when_all_updated(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        """inertia_proper=0.0 when all pre-known elements correctly updated."""
        engine = MutationEngine(workdir, gt)
        result = engine.apply_boundary_breach(
            from_module="mod_c.py", to_module="mod_a.py"
        )

        pre = _belief([("mod_c.py", "models.py", "IMPORTS")])
        post = _belief([
            ("mod_c.py", "models.py", "IMPORTS"),
            ("mod_c.py", "mod_a.py", "IMPORTS"),
        ])

        metrics = score_revision(pre, post, result, gt)
        assert metrics.inertia_proper == pytest.approx(0.0)

    def test_impact_discovery_1_when_all_discovered(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        """impact_discovery=1.0 when pre-unknown elements found post-evidence."""
        engine = MutationEngine(workdir, gt)
        result = engine.apply_boundary_breach(
            from_module="mod_c.py", to_module="mod_a.py"
        )

        # Pre-belief: INCORRECTLY has mod_c→mod_a (hallucinated in pre_gt)
        # This makes it pre-unknown (incorrect before mutation)
        pre = _belief([
            ("mod_c.py", "models.py", "IMPORTS"),
            ("mod_c.py", "mod_a.py", "IMPORTS"),
        ])
        # Post-belief: CORRECTLY has mod_c→mod_a (matches post_gt)
        post = _belief([
            ("mod_c.py", "models.py", "IMPORTS"),
            ("mod_c.py", "mod_a.py", "IMPORTS"),
        ])

        metrics = score_revision(pre, post, result, gt)
        assert metrics.impact_discovery == pytest.approx(1.0)

    def test_sham_gullibility_positive_when_changed(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        """Gullibility > 0 when belief changes in sham condition."""
        engine = ShamMutationEngine(workdir, gt)
        result = engine.generate_sham(MutationType.BOUNDARY_BREACH)

        pre = _belief([("mod_c.py", "models.py", "IMPORTS")])
        # Post adds a spurious edge
        post = _belief([
            ("mod_c.py", "models.py", "IMPORTS"),
            ("mod_c.py", "mod_a.py", "IMPORTS"),
        ])

        metrics = score_revision(pre, post, result, gt)
        assert metrics.gullibility_rate > 0

    def test_interface_break_scoring(self, workdir: Path, gt: CodebaseGroundTruth) -> None:
        """Interface break scoring works against contract layer."""
        engine = MutationEngine(workdir, gt)
        result = engine.apply_interface_break(
            module="registry.py",
            function_name="get_pipeline",
        )

        # Pre-belief: has old signature (0 params)
        pre = _belief(
            exports={
                "registry.py": [
                    BeliefExport(
                        name="get_pipeline",
                        signature=FunctionSignature(params=[], return_type="list"),
                        callers=["runner.py"],
                        confidence=0.9,
                    )
                ]
            }
        )
        # Post-belief: has NEW signature (1 param: strict)
        post = _belief(
            exports={
                "registry.py": [
                    BeliefExport(
                        name="get_pipeline",
                        signature=FunctionSignature(
                            params=[ParameterSpec(name="strict", type_hint="bool")],
                            return_type="list",
                        ),
                        callers=["runner.py"],
                        confidence=0.9,
                    )
                ]
            }
        )

        metrics = score_revision(pre, post, result, gt)
        assert isinstance(metrics, BeliefRevisionMetrics)
        assert metrics.mutation_type == MutationType.INTERFACE_BREAK
