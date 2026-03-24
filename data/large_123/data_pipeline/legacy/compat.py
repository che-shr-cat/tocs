"""Compatibility shims for legacy pipeline format."""

from ..models import *  # noqa: F401,F403


def convert_legacy_format(data: dict) -> dict:
    """Convert legacy pipeline data format to current format.

    The old format used flat dicts with string keys.
    The new format uses typed Pydantic models.
    """
    return {
        "id": data.get("record_id", data.get("id", "")),
        "data": {k: v for k, v in data.items() if k not in ("record_id", "id")},
    }
