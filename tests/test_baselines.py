"""Tests for Task 5: Rule-based baselines.

Validates:
- BFS-Import finds IMPORTS edges
- Config-aware finds REGISTRY_WIRES edges
- Random explorer uses exactly B actions
- Oracle scores F1=1.0
- Config-aware > BFS-Import on F1 (registry-wired codebase)
- BFS-Import > Random on F1
"""

from __future__ import annotations

from pathlib import Path

import pytest

from baselines import bfs_import, config_aware, oracle, random_explorer
from harness.environment import Environment
from metrics.map_accuracy import edge_prf, score_map
from models import CodebaseGroundTruth, EdgeType

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_pipeline"


@pytest.fixture
def gt() -> CodebaseGroundTruth:
    return CodebaseGroundTruth.model_validate_json(
        (FIXTURE_DIR / "ground_truth.json").read_text()
    )


def _env(budget: int = 20) -> Environment:
    return Environment(FIXTURE_DIR, budget=budget)


# ── BFS-Import ──────────────────────────────────────────────────────


class TestBFSImport:
    def test_finds_imports_edges(self, gt: CodebaseGroundTruth) -> None:
        env = _env(budget=20)
        maps = bfs_import.run(env, probe_interval=3)
        assert len(maps) >= 1
        final = maps[-1]

        # Should find at least some IMPORTS edges
        found_imports = False
        for comp in final.components.values():
            for e in comp.edges:
                if e.type == EdgeType.IMPORTS:
                    found_imports = True
                    break
        assert found_imports, "BFS-Import found no IMPORTS edges"

    def test_returns_multiple_maps(self, gt: CodebaseGroundTruth) -> None:
        env = _env(budget=15)
        maps = bfs_import.run(env, probe_interval=3)
        # Should have at least a periodic probe + final map
        assert len(maps) >= 2

    def test_respects_budget(self, gt: CodebaseGroundTruth) -> None:
        budget = 5
        env = _env(budget=budget)
        bfs_import.run(env, probe_interval=3)
        assert env.actions_taken <= budget


# ── Config-Aware ────────────────────────────────────────────────────


class TestConfigAware:
    def test_finds_registry_wires(self, gt: CodebaseGroundTruth) -> None:
        env = _env(budget=20)
        maps = config_aware.run(env, probe_interval=3)
        final = maps[-1]

        found_rw = False
        for comp in final.components.values():
            for e in comp.edges:
                if e.type == EdgeType.REGISTRY_WIRES:
                    found_rw = True
                    break
        assert found_rw, "Config-aware found no REGISTRY_WIRES edges"

    def test_opens_config_files_early(self, gt: CodebaseGroundTruth) -> None:
        env = _env(budget=20)
        config_aware.run(env, probe_interval=3)
        # Config file should be among the first files opened
        config_opened = False
        for i, filepath in enumerate(env.opened_files[:5]):
            if filepath.endswith(".json"):
                config_opened = True
                break
        assert config_opened, "Config file not opened in first 5 actions"


# ── Random Explorer ─────────────────────────────────────────────────


class TestRandomExplorer:
    def test_uses_budget(self, gt: CodebaseGroundTruth) -> None:
        budget = 8
        env = _env(budget=budget)
        random_explorer.run(env, probe_interval=3, seed=42)
        # Should use all budget (LIST + OPENs)
        assert env.actions_taken == budget

    def test_returns_maps(self, gt: CodebaseGroundTruth) -> None:
        env = _env(budget=10)
        maps = random_explorer.run(env, probe_interval=3, seed=42)
        assert len(maps) >= 1
        final = maps[-1]
        # Should have some observed components
        observed = [c for c in final.components.values() if c.status.value == "observed"]
        assert len(observed) >= 1


# ── Oracle ──────────────────────────────────────────────────────────


class TestOracle:
    def test_f1_equals_1(self, gt: CodebaseGroundTruth) -> None:
        maps = oracle.run(gt)
        assert len(maps) == 1
        metrics = score_map(maps[0], gt)
        assert metrics.dependency_f1 == pytest.approx(1.0)
        assert metrics.dependency_precision == pytest.approx(1.0)
        assert metrics.dependency_recall == pytest.approx(1.0)

    def test_all_components_observed(self, gt: CodebaseGroundTruth) -> None:
        maps = oracle.run(gt)
        cmap = maps[0]
        assert set(cmap.components.keys()) == set(gt.modules.keys())
        for comp in cmap.components.values():
            assert comp.status.value == "observed"
            assert comp.confidence == 1.0


# ── Ranking: Config-Aware > BFS-Import > Random ────────────────────


class TestRanking:
    def test_config_aware_beats_bfs_import(self, gt: CodebaseGroundTruth) -> None:
        """Config-aware should score higher F1 because it finds REGISTRY_WIRES."""
        budget = 15

        env_bfs = _env(budget=budget)
        maps_bfs = bfs_import.run(env_bfs, probe_interval=3)
        _, _, f1_bfs = edge_prf(maps_bfs[-1], gt)

        env_cfg = _env(budget=budget)
        maps_cfg = config_aware.run(env_cfg, probe_interval=3)
        _, _, f1_cfg = edge_prf(maps_cfg[-1], gt)

        assert f1_cfg > f1_bfs, (
            f"Config-aware F1={f1_cfg:.3f} should beat BFS-Import F1={f1_bfs:.3f}"
        )

    def test_bfs_import_beats_random(self, gt: CodebaseGroundTruth) -> None:
        """BFS-Import should score higher F1 than random (on average)."""
        budget = 7  # LIST + 6 OPENs out of 10 files — tight enough that strategy matters

        env_bfs = _env(budget=budget)
        maps_bfs = bfs_import.run(env_bfs, probe_interval=3)
        _, _, f1_bfs = edge_prf(maps_bfs[-1], gt)

        # Average over a few random seeds
        f1_random_sum = 0.0
        n_trials = 5
        for seed in range(n_trials):
            env_rnd = _env(budget=budget)
            maps_rnd = random_explorer.run(env_rnd, probe_interval=3, seed=seed)
            _, _, f1_rnd = edge_prf(maps_rnd[-1], gt)
            f1_random_sum += f1_rnd
        f1_random_avg = f1_random_sum / n_trials

        assert f1_bfs > f1_random_avg, (
            f"BFS-Import F1={f1_bfs:.3f} should beat Random avg F1={f1_random_avg:.3f}"
        )
