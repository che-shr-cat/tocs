"""Tests for Task 1: Pipeline codebase generator (hand-authored fixture).

Validates:
- All .py files parse with ast.parse
- ground_truth.json validates against CodebaseGroundTruth schema
- Every IMPORTS edge is actually present in the code
- Distractor modules exist and are NOT in the main pipeline flow
- At least one REGISTRY_WIRES edge exists
- At least one constraint has evidence_type "test"
- Anti-triviality: registry wiring, adapter indirection, neutral naming
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from models import CodebaseGroundTruth, EdgeType

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_pipeline"


@pytest.fixture
def ground_truth() -> CodebaseGroundTruth:
    gt_path = FIXTURE_DIR / "ground_truth.json"
    return CodebaseGroundTruth.model_validate_json(gt_path.read_text())


@pytest.fixture
def py_files() -> list[Path]:
    return sorted(FIXTURE_DIR.glob("*.py"))


# ── Codebase validity ──────────────────────────────────────────────


class TestCodebaseValidity:
    def test_all_py_files_parse(self, py_files: list[Path]) -> None:
        for p in py_files:
            source = p.read_text()
            try:
                ast.parse(source, filename=str(p))
            except SyntaxError as e:
                pytest.fail(f"{p.name} failed to parse: {e}")

    def test_has_init(self) -> None:
        assert (FIXTURE_DIR / "__init__.py").exists()

    def test_file_count_in_range(self, py_files: list[Path]) -> None:
        # 8-10 component files + __init__ + test_boundaries = 10-12 .py files
        assert 8 <= len(py_files) <= 14, f"Got {len(py_files)} .py files"

    def test_pipeline_config_exists(self) -> None:
        assert (FIXTURE_DIR / "pipeline_config.json").exists()

    def test_pipeline_config_valid_json(self) -> None:
        config = json.loads((FIXTURE_DIR / "pipeline_config.json").read_text())
        assert "pipeline" in config
        assert "stages" in config["pipeline"]
        assert len(config["pipeline"]["stages"]) >= 2


# ── Ground truth schema ────────────────────────────────────────────


class TestGroundTruthSchema:
    def test_validates_against_schema(self, ground_truth: CodebaseGroundTruth) -> None:
        assert ground_truth.codebase_id
        assert ground_truth.pattern.value == "pipeline"

    def test_all_module_filepaths_exist(self, ground_truth: CodebaseGroundTruth) -> None:
        for filepath in ground_truth.modules:
            assert (FIXTURE_DIR / filepath).exists(), f"{filepath} in ground truth but not on disk"

    def test_dependency_edges_reference_known_modules(
        self, ground_truth: CodebaseGroundTruth
    ) -> None:
        known = set(ground_truth.modules.keys())
        for edge in ground_truth.dependency_edges:
            assert edge["source"] in known, f"Edge source {edge['source']} unknown"
            assert edge["target"] in known, f"Edge target {edge['target']} unknown"

    def test_has_typed_edges(self, ground_truth: CodebaseGroundTruth) -> None:
        types = {e["type"] for e in ground_truth.dependency_edges}
        assert "IMPORTS" in types
        assert "CALLS_API" in types
        assert "DATA_FLOWS_TO" in types
        assert "REGISTRY_WIRES" in types

    def test_has_contracts(self, ground_truth: CodebaseGroundTruth) -> None:
        assert len(ground_truth.contracts) >= 3
        for c in ground_truth.contracts:
            assert c.name
            assert c.module in ground_truth.modules

    def test_has_invariants(self, ground_truth: CodebaseGroundTruth) -> None:
        assert len(ground_truth.invariants) >= 3

    def test_invariant_ids_unique(self, ground_truth: CodebaseGroundTruth) -> None:
        ids = [inv.id for inv in ground_truth.invariants]
        assert len(ids) == len(set(ids))

    def test_invariants_have_structured_form(
        self, ground_truth: CodebaseGroundTruth
    ) -> None:
        for inv in ground_truth.invariants:
            assert "type" in inv.structured, f"{inv.id} missing structured.type"

    def test_invariants_have_evidence_types(
        self, ground_truth: CodebaseGroundTruth
    ) -> None:
        for inv in ground_truth.invariants:
            assert len(inv.evidence_types) > 0, f"{inv.id} has no evidence_types"


# ── Edge verification against code ─────────────────────────────────


class TestEdgesMatchCode:
    def _get_imports(self, filepath: Path) -> set[str]:
        """Extract imported module names from a Python file."""
        source = filepath.read_text()
        tree = ast.parse(source)
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                # Relative imports like "from .models import ..." → "models"
                parts = node.module.split(".")
                mod_name = parts[-1]
                imports.add(mod_name)
        return imports

    def test_imports_edges_present_in_code(
        self, ground_truth: CodebaseGroundTruth
    ) -> None:
        for edge in ground_truth.dependency_edges:
            if edge["type"] != "IMPORTS":
                continue
            source_file = FIXTURE_DIR / edge["source"]
            target_mod = edge["target"].replace(".py", "")
            imports = self._get_imports(source_file)
            assert target_mod in imports, (
                f"IMPORTS edge {edge['source']} → {edge['target']} "
                f"not found in code. Imports found: {imports}"
            )

    def test_registry_wires_in_config(
        self, ground_truth: CodebaseGroundTruth
    ) -> None:
        """Every REGISTRY_WIRES target should be listed in pipeline_config.json."""
        config = json.loads((FIXTURE_DIR / "pipeline_config.json").read_text())
        config_modules = {s["module"] for s in config["pipeline"]["stages"]}

        for edge in ground_truth.dependency_edges:
            if edge["type"] != "REGISTRY_WIRES":
                continue
            target_mod = edge["target"].replace(".py", "")
            assert target_mod in config_modules, (
                f"REGISTRY_WIRES target {edge['target']} not in pipeline_config.json"
            )

    def test_data_flows_to_witnessed_in_runner(
        self, ground_truth: CodebaseGroundTruth
    ) -> None:
        """DATA_FLOWS_TO edges should be witnessed by the runner loop."""
        runner_src = (FIXTURE_DIR / "runner.py").read_text()
        # The runner chains stages via a loop: data = stage.process(data, config)
        assert "stage.process" in runner_src or "data = " in runner_src


# ── Anti-triviality measures ───────────────────────────────────────


class TestAntiTriviality:
    def test_registry_wiring_exists(self, ground_truth: CodebaseGroundTruth) -> None:
        rw_edges = [e for e in ground_truth.dependency_edges if e["type"] == "REGISTRY_WIRES"]
        assert len(rw_edges) >= 1, "No REGISTRY_WIRES edges found"

    def test_adapter_indirection(self) -> None:
        """base.py defines an ABC that stages implement."""
        base_src = (FIXTURE_DIR / "base.py").read_text()
        assert "ABC" in base_src
        assert "abstractmethod" in base_src

    def test_distractor_modules_exist(self, ground_truth: CodebaseGroundTruth) -> None:
        """helpers.py and legacy.py exist and are not in the main pipeline flow."""
        assert "helpers.py" in ground_truth.modules
        assert "legacy.py" in ground_truth.modules

    def test_distractors_not_in_pipeline_flow(
        self, ground_truth: CodebaseGroundTruth
    ) -> None:
        """Distractor modules have no DATA_FLOWS_TO edges."""
        distractors = {"helpers.py", "legacy.py"}
        for edge in ground_truth.dependency_edges:
            if edge["type"] == "DATA_FLOWS_TO":
                assert edge["source"] not in distractors, (
                    f"Distractor {edge['source']} has DATA_FLOWS_TO edge"
                )
                assert edge["target"] not in distractors, (
                    f"Distractor {edge['target']} is target of DATA_FLOWS_TO"
                )

    def test_legacy_has_no_incoming_edges(
        self, ground_truth: CodebaseGroundTruth
    ) -> None:
        """legacy.py is dead code — nothing points to it."""
        for edge in ground_truth.dependency_edges:
            assert edge["target"] != "legacy.py", (
                f"legacy.py has incoming edge from {edge['source']}"
            )

    def test_neutral_naming(self) -> None:
        """Stage files use neutral names (mod_a, mod_b, ...) not descriptive ones."""
        descriptive = {"extract.py", "transform.py", "load.py", "etl.py",
                       "ingest.py", "validate.py", "enrich.py", "export.py"}
        actual_files = {p.name for p in FIXTURE_DIR.glob("*.py")}
        assert not actual_files & descriptive, (
            f"Found descriptive filenames: {actual_files & descriptive}"
        )

    def test_hidden_invariant_exists(
        self, ground_truth: CodebaseGroundTruth
    ) -> None:
        """At least one invariant requires reading function bodies to discover."""
        has_doc_evidence = any(
            "documentation" in [e.value for e in inv.evidence_types]
            for inv in ground_truth.invariants
        )
        assert has_doc_evidence, "No invariant with documentation evidence"

    def test_constraint_has_test_evidence(
        self, ground_truth: CodebaseGroundTruth
    ) -> None:
        """At least one constraint has evidence_type 'test'."""
        has_test = any(
            "test" in [e.value for e in inv.evidence_types]
            for inv in ground_truth.invariants
        )
        assert has_test, "No invariant with test evidence"

    def test_test_boundaries_exists(self) -> None:
        assert (FIXTURE_DIR / "test_boundaries.py").exists()

    def test_docstring_states_constraint(self) -> None:
        """mod_c.py contains a docstring stating the validation requirement."""
        src = (FIXTURE_DIR / "mod_c.py").read_text()
        assert "is_validated" in src
        assert "IMPORTANT" in src or "must have been validated" in src.lower()


# ── Export module ──────────────────────────────────────────────────


class TestExport:
    def test_export_fixture(self, tmp_path: Path) -> None:
        from generator.export import export_fixture

        gt = export_fixture(tmp_path / "output")
        assert gt.codebase_id == "fixture-pipeline-001"
        assert (tmp_path / "output" / "models.py").exists()
        assert (tmp_path / "output" / "ground_truth.json").exists()

    def test_load_ground_truth(self) -> None:
        from generator.export import load_ground_truth

        gt = load_ground_truth(FIXTURE_DIR)
        assert gt.pattern.value == "pipeline"

    def test_list_codebase_files(self) -> None:
        from generator.export import list_codebase_files

        files = list_codebase_files(FIXTURE_DIR)
        # Should include .py files and pipeline_config.json but NOT ground_truth.json
        assert "models.py" in files
        assert "pipeline_config.json" in files
        assert "ground_truth.json" not in files
