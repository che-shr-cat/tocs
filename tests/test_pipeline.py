"""Tests for Task 7: End-to-end evaluation pipeline.

Smoke tests that run baselines through the full pipeline and verify
the output validates as EvalResult with reasonable metrics.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from evaluation.run_eval import run_single, save_result
from models import ComplexityTier, EvalResult

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_pipeline"


# ── BFS-Import through the pipeline ─────────────────────────────────


class TestBFSImportPipeline:
    def test_produces_eval_result(self) -> None:
        result = run_single(
            model="bfs-import",
            codebase=FIXTURE_DIR,
            mode="active",
            budget=15,
            probe_interval=3,
        )
        assert isinstance(result, EvalResult)
        assert result.model_name == "bfs-import"
        assert result.codebase_id == "fixture-pipeline-001"
        assert result.mode == "active"

    def test_has_cognitive_maps(self) -> None:
        result = run_single(
            model="bfs-import",
            codebase=FIXTURE_DIR,
            budget=15,
            probe_interval=3,
        )
        assert len(result.cognitive_maps) >= 1

    def test_dependency_f1_positive(self) -> None:
        result = run_single(
            model="bfs-import",
            codebase=FIXTURE_DIR,
            budget=15,
            probe_interval=3,
        )
        assert result.map_accuracy.dependency_f1 > 0
        assert result.map_accuracy.dependency_precision > 0
        assert result.map_accuracy.dependency_recall > 0

    def test_exploration_metrics_populated(self) -> None:
        result = run_single(
            model="bfs-import",
            codebase=FIXTURE_DIR,
            budget=15,
            probe_interval=3,
        )
        assert result.exploration.steps_taken > 0
        assert result.exploration.files_opened > 0
        assert result.exploration.unique_files > 0
        assert result.exploration.action_auc > 0


# ── Config-Aware through the pipeline ────────────────────────────────


class TestConfigAwarePipeline:
    def test_produces_eval_result(self) -> None:
        result = run_single(
            model="config-aware",
            codebase=FIXTURE_DIR,
            budget=15,
            probe_interval=3,
        )
        assert isinstance(result, EvalResult)
        assert result.model_name == "config-aware"

    def test_f1_positive(self) -> None:
        result = run_single(
            model="config-aware",
            codebase=FIXTURE_DIR,
            budget=15,
            probe_interval=3,
        )
        assert result.map_accuracy.dependency_f1 > 0


# ── Oracle through the pipeline ──────────────────────────────────────


class TestOraclePipeline:
    def test_perfect_f1(self) -> None:
        result = run_single(
            model="oracle",
            codebase=FIXTURE_DIR,
        )
        assert result.map_accuracy.dependency_f1 == pytest.approx(1.0)
        assert result.map_accuracy.dependency_precision == pytest.approx(1.0)
        assert result.map_accuracy.dependency_recall == pytest.approx(1.0)

    def test_mode_is_active(self) -> None:
        result = run_single(model="oracle", codebase=FIXTURE_DIR)
        assert result.mode == "active"
        assert result.passive_condition is None


# ── Random through the pipeline ──────────────────────────────────────


class TestRandomPipeline:
    def test_produces_eval_result(self) -> None:
        result = run_single(
            model="random",
            codebase=FIXTURE_DIR,
            budget=10,
            probe_interval=3,
        )
        assert isinstance(result, EvalResult)
        assert result.model_name == "random"


# ── Save & reload ────────────────────────────────────────────────────


class TestSaveReload:
    def test_save_and_reload(self, tmp_path: Path) -> None:
        result = run_single(
            model="bfs-import",
            codebase=FIXTURE_DIR,
            budget=10,
            probe_interval=3,
            output=tmp_path,
        )

        # Find saved file
        saved_files = list(tmp_path.glob("*.json"))
        result_files = [f for f in saved_files if "action_log" not in f.name]
        assert len(result_files) == 1

        # Reload and validate
        loaded = EvalResult.model_validate_json(result_files[0].read_text())
        assert loaded.model_name == result.model_name
        assert loaded.codebase_id == result.codebase_id
        assert loaded.map_accuracy.dependency_f1 == pytest.approx(
            result.map_accuracy.dependency_f1
        )

    def test_action_log_saved(self, tmp_path: Path) -> None:
        run_single(
            model="bfs-import",
            codebase=FIXTURE_DIR,
            budget=10,
            probe_interval=3,
            output=tmp_path,
        )

        log_files = list(tmp_path.glob("*_action_log.json"))
        assert len(log_files) == 1

        log_data = json.loads(log_files[0].read_text())
        assert isinstance(log_data, list)
        assert len(log_data) > 0
        assert "action" in log_data[0]
        assert "output" in log_data[0]

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested"
        run_single(
            model="bfs-import",
            codebase=FIXTURE_DIR,
            budget=5,
            probe_interval=3,
            output=nested,
        )
        assert nested.exists()
        assert len(list(nested.glob("*.json"))) >= 1


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_unknown_mode_raises(self) -> None:
        """Unknown mode raises ValueError for non-baseline models."""
        with pytest.raises(ValueError, match="Unknown mode"):
            run_single(
                model="claude-sonnet-4-5-20250929",
                codebase=FIXTURE_DIR,
                mode="nonexistent",
            )

    def test_passive_replay_without_log_raises(self) -> None:
        with pytest.raises(ValueError, match="replay-log"):
            run_single(
                model="claude-sonnet-4-5-20250929",
                codebase=FIXTURE_DIR,
                mode="passive-replay",
            )

    def test_config_aware_beats_bfs_import_in_pipeline(self) -> None:
        """End-to-end ranking matches baseline-level ranking."""
        budget = 15
        r_bfs = run_single(model="bfs-import", codebase=FIXTURE_DIR, budget=budget)
        r_cfg = run_single(model="config-aware", codebase=FIXTURE_DIR, budget=budget)

        assert r_cfg.map_accuracy.dependency_f1 > r_bfs.map_accuracy.dependency_f1


# ── Medium codebase smoke tests ──────────────────────────────────────


class TestMediumCodebasePipeline:
    """Full pipeline smoke tests on a generated medium codebase.

    Validates Task 13 Step 2-4: generate medium codebase, run baselines,
    validate EvalResult output.
    """

    @pytest.fixture(scope="class")
    def medium_codebase(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """Generate a medium codebase for testing."""
        from generator.export import export_from_blueprint
        from generator.grammar import PipelineTemplate

        tmpdir = tmp_path_factory.mktemp("medium_codebase")
        codebase_dir = tmpdir / "test_codebase"
        template = PipelineTemplate()
        bp = template.generate(complexity=ComplexityTier.MEDIUM, seed=99)
        export_from_blueprint(bp, codebase_dir)
        return codebase_dir

    def test_oracle_perfect_f1_medium(self, medium_codebase: Path) -> None:
        result = run_single(model="oracle", codebase=medium_codebase)
        assert result.map_accuracy.dependency_f1 == pytest.approx(1.0)

    def test_bfs_import_medium(self, medium_codebase: Path) -> None:
        result = run_single(
            model="bfs-import", codebase=medium_codebase,
            budget=20, probe_interval=3,
        )
        assert isinstance(result, EvalResult)
        assert len(result.cognitive_maps) >= 1
        assert result.map_accuracy.dependency_f1 > 0
        assert result.exploration.steps_taken > 0

    def test_config_aware_medium(self, medium_codebase: Path) -> None:
        result = run_single(
            model="config-aware", codebase=medium_codebase,
            budget=20, probe_interval=3,
        )
        assert isinstance(result, EvalResult)
        assert len(result.cognitive_maps) >= 1
        assert result.map_accuracy.dependency_f1 > 0

    def test_random_medium(self, medium_codebase: Path) -> None:
        result = run_single(
            model="random", codebase=medium_codebase,
            budget=20, probe_interval=3,
        )
        assert isinstance(result, EvalResult)
        assert len(result.cognitive_maps) >= 1
        # Random may or may not find edges

    def test_config_aware_beats_bfs_medium(self, medium_codebase: Path) -> None:
        """Config-aware should outperform BFS-Import on medium codebases."""
        r_bfs = run_single(
            model="bfs-import", codebase=medium_codebase,
            budget=20, probe_interval=3,
        )
        r_cfg = run_single(
            model="config-aware", codebase=medium_codebase,
            budget=20, probe_interval=3,
        )
        assert r_cfg.map_accuracy.dependency_f1 >= r_bfs.map_accuracy.dependency_f1

    def test_save_and_reload_medium(self, medium_codebase: Path, tmp_path: Path) -> None:
        """Results save and reload as valid EvalResult JSON."""
        result = run_single(
            model="bfs-import", codebase=medium_codebase,
            budget=10, probe_interval=3, output=tmp_path,
        )
        result_files = [
            f for f in tmp_path.glob("*.json") if "action_log" not in f.name
        ]
        assert len(result_files) == 1

        loaded = EvalResult.model_validate_json(result_files[0].read_text())
        assert loaded.model_name == result.model_name
        assert loaded.map_accuracy.dependency_f1 == pytest.approx(
            result.map_accuracy.dependency_f1
        )

    def test_mock_model_exploration_medium(self, medium_codebase: Path) -> None:
        """Mock adapter runs through the full pipeline on medium codebase."""
        from evaluation.model_adapters.base import AdapterResult, BaseAdapter
        from generator.export import load_ground_truth
        from harness.environment import Environment
        from models import CognitiveMap

        gt = load_ground_truth(medium_codebase)
        filepaths = sorted(gt.modules.keys())[:5]

        # Build mock responses: LIST root, LIST package dir, OPEN 5 files, DONE
        pkg_name = filepaths[0].split("/")[0] if "/" in filepaths[0] else ""
        responses = ["LIST()"]
        if pkg_name:
            responses.append(f"LIST({pkg_name})")
            # LIST subdirs
            subdirs = set()
            for fp in filepaths:
                parts = fp.split("/")
                if len(parts) > 2:
                    subdirs.add("/".join(parts[:2]))
            for sd in sorted(subdirs):
                responses.append(f"LIST({sd})")

        for fp in filepaths:
            responses.append(f"OPEN({fp})")

        # Add probe JSON at step 3 intervals
        probe_json = json.dumps({
            "step": 3,
            "components": {
                fp: {
                    "filepath": fp,
                    "status": "observed",
                    "purpose": "test",
                    "edges": [],
                    "exports": [],
                    "confidence": 0.8,
                }
                for fp in filepaths[:2]
            },
            "invariants": [],
            "unexplored": filepaths[2:],
            "uncertainty_summary": "mock test",
        })
        # Insert probe responses at the right positions
        responses_with_probes = []
        action_count = 0
        for r in responses:
            responses_with_probes.append(r)
            action_count += 1
            if action_count % 3 == 0:
                responses_with_probes.append(probe_json)

        responses_with_probes.append("DONE()")
        responses_with_probes.append(probe_json)  # final probe

        class _MockAdapter(BaseAdapter):
            def __init__(self, resps):
                super().__init__(model="mock")
                self._resps = list(resps)
                self._idx = 0

            def _call_model(self, messages, system):
                if self._idx >= len(self._resps):
                    return "DONE()"
                r = self._resps[self._idx]
                self._idx += 1
                return r

        adapter = _MockAdapter(responses_with_probes)
        env = Environment(medium_codebase, budget=20)
        result = adapter.run_exploration(env, probe_interval=3)

        assert isinstance(result, AdapterResult)
        assert len(result.action_log) > 0
        assert len(result.maps) >= 1
