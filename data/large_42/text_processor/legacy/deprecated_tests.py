"""DEPRECATED: Old test file from the original monolithic pipeline.

These tests are no longer valid — the pipeline architecture has been
refactored to use registry-driven stage resolution. Kept for
historical reference only. See test_smoke.py for current tests.
"""

from ..models import *  # noqa: F401,F403


def test_old_pipeline_runs():
    """DEPRECATED: No longer valid after refactor."""
    # This test called the old monolithic run_old_pipeline()
    # which has been replaced by runner.run_pipeline()
    pass


def test_old_format_output():
    """DEPRECATED: Output format has changed."""
    # The old pipeline produced flat dicts; the new pipeline
    # produces typed Pydantic models via serializers
    pass
