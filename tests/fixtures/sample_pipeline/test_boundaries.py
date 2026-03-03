"""Architectural boundary tests for the pipeline.

These tests verify critical architectural constraints that must hold
across the pipeline. They serve as documentation AND enforcement.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent


def test_validation_before_enrichment():
    """Constraint: mod_b (validation) must precede mod_c (enrichment) in config.

    This ordering is critical because mod_c assumes records have
    is_validated=True.  Reordering stages to put enrichment before
    validation would cause data integrity violations in strict mode.
    """
    config_path = PIPELINE_DIR / "pipeline_config.json"
    with open(config_path) as f:
        config = json.load(f)

    stage_modules = [s["module"] for s in config["pipeline"]["stages"]]
    assert "mod_b" in stage_modules, "Validation stage (mod_b) missing from pipeline"
    assert "mod_c" in stage_modules, "Enrichment stage (mod_c) missing from pipeline"

    validate_idx = stage_modules.index("mod_b")
    enrich_idx = stage_modules.index("mod_c")
    assert validate_idx < enrich_idx, (
        f"Validation (mod_b, index={validate_idx}) must come before "
        f"enrichment (mod_c, index={enrich_idx}) in the pipeline"
    )


def test_no_direct_stage_imports():
    """Constraint: stages (mod_a through mod_d) must not import each other.

    Stages communicate only through the runner/registry mechanism.
    Direct imports between stages would bypass the pipeline contract.
    """
    stage_files = ["mod_a.py", "mod_b.py", "mod_c.py", "mod_d.py"]
    stage_modules = {"mod_a", "mod_b", "mod_c", "mod_d"}

    for fname in stage_files:
        filepath = PIPELINE_DIR / fname
        source = filepath.read_text()
        tree = ast.parse(source)
        current_mod = fname.replace(".py", "")

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_mod = node.module.split(".")[-1]
                if imported_mod in stage_modules and imported_mod != current_mod:
                    raise AssertionError(
                        f"{fname} imports from {imported_mod} — "
                        f"stages must not import each other directly"
                    )
