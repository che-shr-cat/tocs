"""Shared helper functions.

Note: This module provides general-purpose utilities. It is NOT part of
the pipeline stage sequence — it only provides support functions used by
individual stages as needed.
"""

from __future__ import annotations

import hashlib
from typing import Any


def compute_checksum(data: dict[str, Any]) -> str:
    """Compute a deterministic checksum for a data payload."""
    content = str(sorted(data.items()))
    return hashlib.md5(content.encode()).hexdigest()[:8]


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary with dot-separated keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, key).items())
        else:
            items.append((key, v))
    return dict(items)
