"""Tests for Task 2: Partial observability harness.

Validates:
- LIST returns filenames only (no contents)
- OPEN returns contents and increments action count
- SEARCH returns paths + line numbers but NO content
- INSPECT returns signature + docstring but not function body
- Budget enforcement (BudgetExhausted after B actions)
- DONE doesn't cost an action
- ground_truth.json is hidden from all actions
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.environment import BudgetExhausted, Environment
from models import ActionType, AgentAction

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_pipeline"


@pytest.fixture
def env() -> Environment:
    return Environment(FIXTURE_DIR, budget=20)


@pytest.fixture
def tiny_env() -> Environment:
    """Environment with budget=3 for budget tests."""
    return Environment(FIXTURE_DIR, budget=3)


def _action(typ: ActionType, arg: str | None = None, sec: str | None = None) -> AgentAction:
    return AgentAction(type=typ, argument=arg, secondary_argument=sec)


# ── LIST ────────────────────────────────────────────────────────────


class TestList:
    def test_returns_filenames(self, env: Environment) -> None:
        result = env.step(_action(ActionType.LIST, ""))
        names = result.output.splitlines()
        assert "models.py" in names
        assert "mod_a.py" in names
        assert "pipeline_config.json" in names

    def test_no_file_contents_in_list(self, env: Environment) -> None:
        result = env.step(_action(ActionType.LIST, ""))
        # Should not contain any Python code
        assert "import" not in result.output
        assert "class " not in result.output
        assert "def " not in result.output

    def test_ground_truth_hidden(self, env: Environment) -> None:
        result = env.step(_action(ActionType.LIST, ""))
        assert "ground_truth.json" not in result.output

    def test_no_pycache(self, env: Environment) -> None:
        result = env.step(_action(ActionType.LIST, ""))
        assert "__pycache__" not in result.output

    def test_nonexistent_directory(self, env: Environment) -> None:
        result = env.step(_action(ActionType.LIST, "nonexistent"))
        assert "Error" in result.output

    def test_costs_one_action(self, env: Environment) -> None:
        env.step(_action(ActionType.LIST, ""))
        assert env.actions_taken == 1


# ── OPEN ────────────────────────────────────────────────────────────


class TestOpen:
    def test_returns_file_contents(self, env: Environment) -> None:
        result = env.step(_action(ActionType.OPEN, "models.py"))
        assert "class Record" in result.output
        assert "PipelineConfig" in result.output

    def test_increments_action_count(self, env: Environment) -> None:
        assert env.actions_taken == 0
        env.step(_action(ActionType.OPEN, "models.py"))
        assert env.actions_taken == 1
        env.step(_action(ActionType.OPEN, "base.py"))
        assert env.actions_taken == 2

    def test_tracks_opened_files(self, env: Environment) -> None:
        env.step(_action(ActionType.OPEN, "models.py"))
        env.step(_action(ActionType.OPEN, "mod_a.py"))
        assert env.opened_files == ["models.py", "mod_a.py"]

    def test_nonexistent_file(self, env: Environment) -> None:
        result = env.step(_action(ActionType.OPEN, "does_not_exist.py"))
        assert "Error" in result.output

    def test_ground_truth_hidden(self, env: Environment) -> None:
        result = env.step(_action(ActionType.OPEN, "ground_truth.json"))
        assert "Error" in result.output
        assert "not found" in result.output


# ── SEARCH ──────────────────────────────────────────────────────────


class TestSearch:
    def test_returns_filepaths_and_lines(self, env: Environment) -> None:
        result = env.step(_action(ActionType.SEARCH, "StageBase"))
        assert "base.py" in result.output
        # Should have line numbers
        for line in result.output.splitlines():
            parts = line.split(":")
            assert len(parts) == 2, f"Expected filepath:lines, got: {line}"
            # Second part should be comma-separated numbers
            for num in parts[1].split(","):
                assert num.strip().isdigit(), f"Expected line number, got: {num}"

    def test_no_content_in_search(self, env: Environment) -> None:
        """CRITICAL: SEARCH must NOT return content — only paths and line numbers."""
        result = env.step(_action(ActionType.SEARCH, "is_validated"))
        output = result.output
        # These strings appear in file contents but must NOT appear in search output
        assert "Record" not in output
        assert "class " not in output
        assert "def " not in output
        assert "import" not in output
        assert "True" not in output
        assert "False" not in output

    def test_search_finds_across_files(self, env: Environment) -> None:
        result = env.step(_action(ActionType.SEARCH, "StageBase"))
        files_found = {line.split(":")[0] for line in result.output.splitlines()}
        assert "base.py" in files_found
        # StageBase is also imported in mod_a, mod_b, mod_c, mod_d
        assert len(files_found) >= 2

    def test_search_no_match(self, env: Environment) -> None:
        result = env.step(_action(ActionType.SEARCH, "xyzzy_nonexistent_12345"))
        assert "No matches" in result.output

    def test_ground_truth_hidden_from_search(self, env: Environment) -> None:
        result = env.step(_action(ActionType.SEARCH, "fixture-pipeline"))
        # "fixture-pipeline" appears in ground_truth.json but not in code files
        assert "ground_truth" not in result.output

    def test_costs_one_action(self, env: Environment) -> None:
        env.step(_action(ActionType.SEARCH, "import"))
        assert env.actions_taken == 1


# ── INSPECT ─────────────────────────────────────────────────────────


class TestInspect:
    def test_returns_signature(self, env: Environment) -> None:
        result = env.step(_action(ActionType.INSPECT, "helpers.py", "compute_checksum"))
        assert "def" in result.output
        assert "compute_checksum" in result.output
        assert "data" in result.output  # parameter name

    def test_returns_docstring(self, env: Environment) -> None:
        result = env.step(_action(ActionType.INSPECT, "helpers.py", "compute_checksum"))
        assert "checksum" in result.output.lower()

    def test_no_function_body(self, env: Environment) -> None:
        result = env.step(_action(ActionType.INSPECT, "helpers.py", "compute_checksum"))
        # The function body contains "hashlib.md5" — should NOT appear
        assert "md5" not in result.output
        assert "hexdigest" not in result.output

    def test_inspect_class(self, env: Environment) -> None:
        result = env.step(_action(ActionType.INSPECT, "base.py", "StageBase"))
        assert "class StageBase" in result.output
        assert "process" in result.output  # method signature

    def test_inspect_class_no_body(self, env: Environment) -> None:
        """Class INSPECT shows method signatures but not implementations."""
        result = env.step(_action(ActionType.INSPECT, "mod_a.py", "IngestStage"))
        assert "class IngestStage" in result.output
        assert "def process" in result.output
        # Body contains "strip().lower()" — should not be in output
        assert "strip().lower()" not in result.output

    def test_symbol_not_found(self, env: Environment) -> None:
        result = env.step(_action(ActionType.INSPECT, "helpers.py", "nonexistent"))
        assert "not found" in result.output

    def test_file_not_found(self, env: Environment) -> None:
        result = env.step(_action(ActionType.INSPECT, "nope.py", "foo"))
        assert "Error" in result.output

    def test_costs_one_action(self, env: Environment) -> None:
        env.step(_action(ActionType.INSPECT, "helpers.py", "compute_checksum"))
        assert env.actions_taken == 1


# ── DONE ────────────────────────────────────────────────────────────


class TestDone:
    def test_does_not_cost_action(self, env: Environment) -> None:
        env.step(_action(ActionType.OPEN, "models.py"))
        assert env.actions_taken == 1
        env.step(_action(ActionType.DONE))
        assert env.actions_taken == 1  # DONE is free

    def test_terminates(self, env: Environment) -> None:
        env.step(_action(ActionType.DONE))
        assert env.terminated

    def test_no_actions_after_done(self, env: Environment) -> None:
        env.step(_action(ActionType.DONE))
        with pytest.raises(BudgetExhausted):
            env.step(_action(ActionType.OPEN, "models.py"))


# ── Budget enforcement ──────────────────────────────────────────────


class TestBudget:
    def test_budget_exhausted(self, tiny_env: Environment) -> None:
        """Budget of 3 → 3 actions allowed, 4th raises."""
        tiny_env.step(_action(ActionType.LIST, ""))
        tiny_env.step(_action(ActionType.OPEN, "models.py"))
        tiny_env.step(_action(ActionType.SEARCH, "import"))
        assert tiny_env.actions_taken == 3
        with pytest.raises(BudgetExhausted):
            tiny_env.step(_action(ActionType.OPEN, "base.py"))

    def test_remaining_budget(self, tiny_env: Environment) -> None:
        assert tiny_env.remaining_budget == 3
        tiny_env.step(_action(ActionType.OPEN, "models.py"))
        assert tiny_env.remaining_budget == 2

    def test_done_after_budget_still_works(self, tiny_env: Environment) -> None:
        """DONE should work even after budget is exhausted."""
        for _ in range(3):
            tiny_env.step(_action(ActionType.LIST, ""))
        # Budget exhausted, but DONE is free
        tiny_env.step(_action(ActionType.DONE))
        assert tiny_env.terminated


# ── Action log ──────────────────────────────────────────────────────


class TestActionLog:
    def test_log_records_actions(self, env: Environment) -> None:
        env.step(_action(ActionType.LIST, ""))
        env.step(_action(ActionType.OPEN, "models.py"))
        env.step(_action(ActionType.DONE))
        assert len(env.action_log) == 3
        assert env.action_log[0].action.type == ActionType.LIST
        assert env.action_log[1].action.type == ActionType.OPEN
        assert env.action_log[2].action.type == ActionType.DONE

    def test_step_numbers_correct(self, env: Environment) -> None:
        env.step(_action(ActionType.LIST, ""))
        env.step(_action(ActionType.OPEN, "models.py"))
        env.step(_action(ActionType.DONE))
        assert env.action_log[0].step == 1
        assert env.action_log[1].step == 2
        assert env.action_log[2].step == 2  # DONE doesn't increment
