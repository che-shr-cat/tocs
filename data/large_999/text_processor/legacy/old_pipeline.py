"""DEPRECATED: Old monolithic pipeline implementation.

This module is kept for historical reference only.
It is NOT imported or used by any active code.

The new pipeline uses a stage-based architecture with
registry-driven stage resolution (see runner.py and registry.py).
"""

# NOTE: This entire module is dead code.
# It was the original implementation before the refactor to
# the current stage-based architecture.


def run_old_pipeline(input_path: str, output_path: str) -> None:
    """Run the old monolithic pipeline (deprecated)."""
    raise NotImplementedError(
        "Old pipeline has been replaced. Use runner.run_pipeline() instead."
    )
