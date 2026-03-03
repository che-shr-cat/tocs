"""Legacy data converter — retained for backwards compatibility.

This module was part of an earlier version of the pipeline that used
a different record format. It is no longer used by any active stage
but is kept in the repository in case historical data needs reprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LegacyRecord:
    """Old-format record from v0 pipeline."""

    key: str
    value: str
    timestamp: float = 0.0


def convert_legacy(records: list[LegacyRecord]) -> list[dict]:
    """Convert legacy records to dict format."""
    return [{"id": r.key, "payload": {"value": r.value}} for r in records]
