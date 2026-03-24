"""Output formatting: JSON, CSV, table formatters."""

import json
from typing import Any


def format_json(records: list[Any]) -> str:
    """Format records as JSON lines."""
    lines = []
    for r in records:
        if hasattr(r, "model_dump"):
            lines.append(json.dumps(r.model_dump()))
        else:
            lines.append(json.dumps(r))
    return "\n".join(lines)


def format_csv(records: list[Any], delimiter: str = ",") -> str:
    """Format records as CSV."""
    if not records:
        return ""
    first = records[0]
    if hasattr(first, "model_dump"):
        headers = list(first.model_dump().keys())
    else:
        headers = list(first.keys()) if isinstance(first, dict) else []
    lines = [delimiter.join(headers)]
    for r in records:
        d = r.model_dump() if hasattr(r, "model_dump") else r
        lines.append(delimiter.join(str(d.get(h, "")) for h in headers))
    return "\n".join(lines)


def format_table(records: list[Any]) -> str:
    """Format records as a text table."""
    return format_csv(records, delimiter="\t")
