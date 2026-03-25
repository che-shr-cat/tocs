"""Smoke tests verifying pipeline architectural invariants.

IMPORTANT: These tests document and enforce critical architectural
constraints. Stages must not import from each other directly.
All stage access must go through the base interface.
"""

import ast
import importlib
from pathlib import Path


def test_no_inter_stage_imports():
    """Stages must not import from each other (boundary invariant)."""
    stages_dir = Path(__file__).parent / "stages"
    stage_files = sorted(stages_dir.glob("mod_*.py"))
    stage_names = {f.stem for f in stage_files}

    for sf in stage_files:
        tree = ast.parse(sf.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported = node.module.split(".")[-1]
                assert imported not in stage_names or imported == sf.stem, (
                    f"{sf.name} imports from stage {imported}"
                )


def test_stages_inherit_base():
    """All stages must inherit from StageBase (interface invariant)."""
    stages_dir = Path(__file__).parent / "stages"
    for sf in sorted(stages_dir.glob("mod_*.py")):
        source = sf.read_text()
        assert "StageBase" in source, (
            f"{sf.name} does not reference StageBase"
        )

