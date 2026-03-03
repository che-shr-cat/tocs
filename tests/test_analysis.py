"""Tests for Task 10: Gap Analysis + Figures.

Validates:
- Gap analysis produces correct APG values from synthetic results
- Figures generate without errors (save to /tmp, verify files exist)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from metrics.gap_analysis import (
    APGEntry,
    GapAnalysisResult,
    compute_gap,
    load_results,
)
from analysis.figures import (
    generate_all,
    plot_baseline_comparison,
    plot_edge_type_discovery,
    plot_f1_vs_steps,
)
from models import (
    BeliefRevisionMetrics,
    CognitiveMap,
    ComponentBelief,
    ConstraintDiscoveryMetrics,
    EvalResult,
    ExplorationMetrics,
    MapAccuracyMetrics,
    ModuleStatus,
    MutationType,
)


# ── Synthetic result helpers ───────────────────────────────────────


def _make_result(
    model: str = "test-model",
    codebase: str = "test-codebase",
    mode: str = "active",
    condition: str | None = None,
    dep_f1: float = 0.5,
    inv_f1: float = 0.3,
    action_auc: float = 0.4,
    revision: BeliefRevisionMetrics | None = None,
    constraint: ConstraintDiscoveryMetrics | None = None,
) -> EvalResult:
    return EvalResult(
        model_name=model,
        codebase_id=codebase,
        mode=mode,
        passive_condition=condition,
        exploration=ExplorationMetrics(
            steps_taken=10,
            files_opened=5,
            unique_files=5,
            final_efficiency=action_auc,
            action_auc=action_auc,
            information_gain_curve=[0.1, 0.2, 0.3, 0.4, 0.5],
            action_efficiency_curve=[0.0, 0.1, 0.2, 0.3, dep_f1],
        ),
        map_accuracy=MapAccuracyMetrics(
            dependency_precision=dep_f1,
            dependency_recall=dep_f1,
            dependency_f1=dep_f1,
            invariant_precision=inv_f1,
            invariant_recall=inv_f1,
            invariant_f1=inv_f1,
            confidence_ece=0.1,
        ),
        belief_revision=revision,
        constraint_discovery=constraint,
        cognitive_maps=[
            CognitiveMap(
                step=3,
                components={
                    "mod_a.py": ComponentBelief(
                        filepath="mod_a.py",
                        status=ModuleStatus.OBSERVED,
                        purpose="Test module",
                        edges=[],
                        exports=[],
                        confidence=0.8,
                    ),
                },
            ),
        ],
    )


# ── Gap analysis tests ─────────────────────────────────────────────


class TestAPGEntry:
    def test_apg_total(self) -> None:
        entry = APGEntry(
            model_name="m",
            codebase_id="c",
            metric="dependency_f1",
            active_value=0.5,
            passive_full_value=0.8,
        )
        assert entry.apg_total == pytest.approx(0.3)

    def test_apg_selection(self) -> None:
        entry = APGEntry(
            model_name="m",
            codebase_id="c",
            metric="dependency_f1",
            active_value=0.5,
            passive_oracle_value=0.7,
        )
        assert entry.apg_selection == pytest.approx(0.2)

    def test_apg_decision(self) -> None:
        entry = APGEntry(
            model_name="m",
            codebase_id="c",
            metric="dependency_f1",
            active_value=0.5,
            passive_replay_value=0.65,
        )
        assert entry.apg_decision == pytest.approx(0.15)

    def test_apg_none_when_missing(self) -> None:
        entry = APGEntry(
            model_name="m",
            codebase_id="c",
            metric="dependency_f1",
            active_value=0.5,
        )
        assert entry.apg_total is None
        assert entry.apg_selection is None
        assert entry.apg_decision is None


class TestComputeGap:
    def test_basic_gap(self) -> None:
        results = [
            _make_result(model="m", mode="active", dep_f1=0.5),
            _make_result(model="m", mode="passive", condition="full", dep_f1=0.8),
        ]
        gap = compute_gap(results)
        dep_entries = [e for e in gap.entries if e.metric == "dependency_f1"]
        assert len(dep_entries) == 1
        assert dep_entries[0].apg_total == pytest.approx(0.3)

    def test_multiple_models(self) -> None:
        results = [
            _make_result(model="a", mode="active", dep_f1=0.3),
            _make_result(model="b", mode="active", dep_f1=0.5),
            _make_result(model="a", mode="passive", condition="full", dep_f1=0.7),
            _make_result(model="b", mode="passive", condition="full", dep_f1=0.9),
        ]
        gap = compute_gap(results)
        dep_entries = {
            e.model_name: e
            for e in gap.entries
            if e.metric == "dependency_f1"
        }
        assert dep_entries["a"].apg_total == pytest.approx(0.4)
        assert dep_entries["b"].apg_total == pytest.approx(0.4)

    def test_full_decomposition(self) -> None:
        results = [
            _make_result(model="m", mode="active", dep_f1=0.4),
            _make_result(model="m", mode="passive", condition="full", dep_f1=0.9),
            _make_result(model="m", mode="passive", condition="oracle", dep_f1=0.7),
            _make_result(model="m", mode="passive", condition="replay", dep_f1=0.6),
        ]
        gap = compute_gap(results)
        dep = [e for e in gap.entries if e.metric == "dependency_f1"][0]
        assert dep.apg_total == pytest.approx(0.5)
        assert dep.apg_selection == pytest.approx(0.3)
        assert dep.apg_decision == pytest.approx(0.2)

    def test_produces_three_metrics(self) -> None:
        results = [_make_result(model="m", mode="active")]
        gap = compute_gap(results)
        metrics = {e.metric for e in gap.entries}
        assert metrics == {"dependency_f1", "invariant_f1", "action_auc"}

    def test_no_active_no_entries(self) -> None:
        results = [
            _make_result(model="m", mode="passive", condition="full"),
        ]
        gap = compute_gap(results)
        assert len(gap.entries) == 0


class TestGapOutput:
    def test_markdown_output(self) -> None:
        results = [
            _make_result(model="m", mode="active", dep_f1=0.5),
            _make_result(model="m", mode="passive", condition="full", dep_f1=0.8),
        ]
        gap = compute_gap(results)
        md = gap.to_markdown()
        assert "| m |" in md
        assert "dependency_f1" in md
        assert "0.300" in md  # APG total

    def test_csv_output(self) -> None:
        results = [
            _make_result(model="m", mode="active", dep_f1=0.5),
        ]
        gap = compute_gap(results)
        csv_text = gap.to_csv()
        assert "model" in csv_text
        assert "dependency_f1" in csv_text

    def test_empty_results(self) -> None:
        gap = compute_gap([])
        assert gap.to_markdown() == "No data."


class TestLoadResults:
    def test_load_from_dir(self, tmp_path: Path) -> None:
        result = _make_result()
        path = tmp_path / "test.json"
        path.write_text(result.model_dump_json(indent=2))

        loaded = load_results(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].model_name == "test-model"

    def test_skips_action_logs(self, tmp_path: Path) -> None:
        result = _make_result()
        (tmp_path / "test.json").write_text(result.model_dump_json())
        (tmp_path / "test_action_log.json").write_text("[]")

        loaded = load_results(tmp_path)
        assert len(loaded) == 1

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("not valid json")
        result = _make_result()
        (tmp_path / "good.json").write_text(result.model_dump_json())

        loaded = load_results(tmp_path)
        assert len(loaded) == 1


# ── Figure generation tests ────────────────────────────────────────


class TestFigures:
    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "figures"

    @pytest.fixture
    def sample_results(self) -> list[EvalResult]:
        return [
            _make_result(
                model="baseline-a",
                mode="active",
                dep_f1=0.4,
            ),
            _make_result(
                model="baseline-b",
                mode="active",
                dep_f1=0.6,
            ),
            _make_result(
                model="baseline-a",
                mode="passive",
                condition="full",
                dep_f1=0.8,
            ),
            _make_result(
                model="baseline-b",
                mode="passive",
                condition="full",
                dep_f1=0.9,
            ),
        ]

    def test_f1_vs_steps_creates_file(self, sample_results, output_dir) -> None:
        path = plot_f1_vs_steps(sample_results, output_dir)
        assert path.exists()
        assert path.suffix == ".png"

    def test_baseline_comparison_creates_file(self, sample_results, output_dir) -> None:
        path = plot_baseline_comparison(sample_results, output_dir)
        assert path.exists()
        assert path.suffix == ".png"

    def test_edge_type_discovery_creates_file(self, sample_results, output_dir) -> None:
        path = plot_edge_type_discovery(sample_results, output_dir)
        assert path.exists()
        assert path.suffix == ".png"

    def test_generate_all_creates_3_files(self, sample_results, output_dir) -> None:
        paths = generate_all(sample_results, output_dir)
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_empty_results_still_generates(self, output_dir) -> None:
        """Empty results produce placeholder figures."""
        paths = generate_all([], output_dir)
        assert len(paths) == 3
        for p in paths:
            assert p.exists()

    def test_output_directory_created(self, tmp_path) -> None:
        nested = tmp_path / "deep" / "nested" / "figures"
        results = [_make_result(mode="active")]
        paths = generate_all(results, nested)
        assert nested.exists()
        assert len(paths) == 3
