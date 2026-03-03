"""Tests for Task 9: Constraint Discovery Probes.

Validates:
- Probe generation produces valid 4-option MCQs
- Correct answer is actually a violation
- Distractors don't violate constraints
- ComplianceChecker catches a forbidden import
- ComplianceChecker passes compliant code
"""

from __future__ import annotations

from pathlib import Path

import pytest

from generator.export import load_ground_truth
from metrics.constraint_discovery import (
    ComplianceChecker,
    ConstraintProbe,
    ProbeGenerator,
    score_constraint_discovery,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_pipeline"


@pytest.fixture
def gt():
    return load_ground_truth(FIXTURE_DIR)


# ── Probe generation ──────────────────────────────────────────────


class TestProbeGeneration:
    def test_generates_probes_for_all_invariants(self, gt) -> None:
        gen = ProbeGenerator(gt, seed=42)
        probes = gen.generate_all()
        assert len(probes) == len(gt.invariants)

    def test_each_probe_has_4_options(self, gt) -> None:
        gen = ProbeGenerator(gt, seed=42)
        probes = gen.generate_all()
        for probe in probes:
            assert len(probe.options) == 4, (
                f"Probe {probe.invariant_id} has {len(probe.options)} options"
            )

    def test_each_probe_has_exactly_one_correct(self, gt) -> None:
        gen = ProbeGenerator(gt, seed=42)
        probes = gen.generate_all()
        for probe in probes:
            correct_count = sum(1 for o in probe.options if o.is_correct)
            assert correct_count == 1, (
                f"Probe {probe.invariant_id} has {correct_count} correct options"
            )

    def test_correct_index_matches_correct_option(self, gt) -> None:
        gen = ProbeGenerator(gt, seed=42)
        probes = gen.generate_all()
        for probe in probes:
            assert probe.options[probe.correct_index].is_correct

    def test_probe_has_question(self, gt) -> None:
        gen = ProbeGenerator(gt, seed=42)
        probes = gen.generate_all()
        for probe in probes:
            assert "constraint" in probe.question.lower()
            assert probe.question.endswith("?")

    def test_probe_has_explanation(self, gt) -> None:
        gen = ProbeGenerator(gt, seed=42)
        probes = gen.generate_all()
        for probe in probes:
            assert len(probe.explanation) > 10

    def test_probe_has_invariant_id(self, gt) -> None:
        gen = ProbeGenerator(gt, seed=42)
        probes = gen.generate_all()
        inv_ids = {inv.id for inv in gt.invariants}
        for probe in probes:
            assert probe.invariant_id in inv_ids

    def test_forbidden_edge_violation_mentions_import(self, gt) -> None:
        """Correct answer for FORBIDDEN_EDGE should mention importing."""
        gen = ProbeGenerator(gt, seed=42)
        probes = gen.generate_all()
        forbidden_probes = [
            p for p in probes
            if any(
                inv.structured.get("type") == "FORBIDDEN_EDGE"
                for inv in gt.invariants
                if inv.id == p.invariant_id
            )
        ]
        assert len(forbidden_probes) > 0
        for probe in forbidden_probes:
            correct = probe.options[probe.correct_index]
            assert "import" in correct.label.lower()

    def test_deterministic_with_seed(self, gt) -> None:
        gen1 = ProbeGenerator(gt, seed=42)
        gen2 = ProbeGenerator(gt, seed=42)
        probes1 = gen1.generate_all()
        probes2 = gen2.generate_all()

        for p1, p2 in zip(probes1, probes2):
            assert p1.correct_index == p2.correct_index
            assert p1.invariant_id == p2.invariant_id
            for o1, o2 in zip(p1.options, p2.options):
                assert o1.label == o2.label

    def test_different_seeds_may_differ(self, gt) -> None:
        gen1 = ProbeGenerator(gt, seed=1)
        gen2 = ProbeGenerator(gt, seed=999)
        probes1 = gen1.generate_all()
        probes2 = gen2.generate_all()

        # At least some probes should have different option ordering
        any_different = False
        for p1, p2 in zip(probes1, probes2):
            if p1.correct_index != p2.correct_index:
                any_different = True
                break
        # Not strictly guaranteed but very likely with 6 probes
        assert any_different or len(probes1) == 0


# ── Compliance checker ─────────────────────────────────────────────


class TestComplianceChecker:
    def test_catches_forbidden_import(self, gt) -> None:
        checker = ComplianceChecker(gt)
        # mod_c.py importing mod_a.py is forbidden (inv-002: no stage-to-stage)
        code = "from .mod_a import IngestStage\n"
        result = checker.check_forbidden_edge(
            code, "mod_c.py", forbidden_targets=["mod_a.py"]
        )
        assert not result.compliant
        assert "mod_a" in result.details

    def test_passes_compliant_import(self, gt) -> None:
        checker = ComplianceChecker(gt)
        # Importing from models.py is always fine
        code = "from .models import Record\n"
        result = checker.check_forbidden_edge(
            code, "mod_a.py", forbidden_targets=["mod_b.py", "mod_c.py"]
        )
        assert result.compliant

    def test_forbidden_import_from_gt(self, gt) -> None:
        """Auto-derive forbidden targets from ground truth."""
        checker = ComplianceChecker(gt)
        # mod_a is a stage; inv-002 forbids stage-to-stage imports
        code = "from .mod_b import ValidateStage\n"
        result = checker.check_forbidden_edge(code, "mod_a.py")
        assert not result.compliant

    def test_passes_non_forbidden_import(self, gt) -> None:
        checker = ComplianceChecker(gt)
        # Importing from models.py is fine for any module
        code = "from .models import Record, PipelineConfig\n"
        result = checker.check_forbidden_edge(code, "mod_a.py")
        assert result.compliant

    def test_catches_runner_import(self, gt) -> None:
        """inv-004: no stage/helper may import runner.py."""
        checker = ComplianceChecker(gt)
        code = "from .runner import Pipeline\n"
        result = checker.check_forbidden_edge(code, "mod_b.py")
        assert not result.compliant

    def test_interface_only_passes_clean_code(self, gt) -> None:
        checker = ComplianceChecker(gt)
        code = "from .base import StageBase\nclass MyStage(StageBase): pass\n"
        result = checker.check_interface_only(code)
        assert result.compliant

    def test_interface_only_catches_direct_stage_access(self, gt) -> None:
        checker = ComplianceChecker(gt)
        code = (
            "from .mod_a import IngestStage\n"
            "result = mod_a._internal_method()\n"
        )
        result = checker.check_interface_only(code)
        assert not result.compliant

    def test_validation_chain_passes_code_with_check(self, gt) -> None:
        checker = ComplianceChecker(gt)
        code = (
            "for r in records:\n"
            "    if not r.is_validated:\n"
            "        raise ValueError('not validated')\n"
        )
        result = checker.check_validation_chain(code)
        assert result.compliant

    def test_validation_chain_passes_unrelated_code(self, gt) -> None:
        checker = ComplianceChecker(gt)
        code = "x = 1 + 2\nprint(x)\n"
        result = checker.check_validation_chain(code)
        assert result.compliant

    def test_check_all_returns_three_results(self, gt) -> None:
        checker = ComplianceChecker(gt)
        code = "from .models import Record\n"
        results = checker.check_all(code, "mod_a.py")
        assert len(results) == 3
        types = {r.constraint_type for r in results}
        assert types == {"FORBIDDEN_EDGE", "INTERFACE_ONLY", "VALIDATION_CHAIN"}


# ── Scoring ────────────────────────────────────────────────────────


class TestScoring:
    def test_perfect_score(self, gt) -> None:
        gen = ProbeGenerator(gt, seed=42)
        probes = gen.generate_all()
        answers = [p.correct_index for p in probes]
        metrics = score_constraint_discovery(probes, answers)
        assert metrics.counterfactual_probe_accuracy == pytest.approx(1.0)

    def test_zero_score(self, gt) -> None:
        gen = ProbeGenerator(gt, seed=42)
        probes = gen.generate_all()
        # Always pick wrong answer
        answers = [(p.correct_index + 1) % 4 for p in probes]
        metrics = score_constraint_discovery(probes, answers)
        assert metrics.counterfactual_probe_accuracy == pytest.approx(0.0)

    def test_partial_score(self, gt) -> None:
        gen = ProbeGenerator(gt, seed=42)
        probes = gen.generate_all()
        # Get half right
        answers = []
        for i, p in enumerate(probes):
            if i % 2 == 0:
                answers.append(p.correct_index)
            else:
                answers.append((p.correct_index + 1) % 4)
        metrics = score_constraint_discovery(probes, answers)
        assert 0.0 < metrics.counterfactual_probe_accuracy < 1.0

    def test_empty_probes(self) -> None:
        metrics = score_constraint_discovery([], [])
        assert metrics.counterfactual_probe_accuracy == pytest.approx(0.0)

    def test_behavioral_compliance_is_none(self, gt) -> None:
        gen = ProbeGenerator(gt, seed=42)
        probes = gen.generate_all()
        answers = [p.correct_index for p in probes]
        metrics = score_constraint_discovery(probes, answers)
        assert metrics.behavioral_compliance is None
