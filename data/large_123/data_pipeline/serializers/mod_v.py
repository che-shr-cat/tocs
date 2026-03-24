"""JSON serializer for pipeline output."""

import json
from typing import Any

from ..models import BaseModel
from ..utils.formatters import format_json


def serialize_json(records: list[Any], output_path: str = "") -> str:
    """Serialize pipeline output as JSON.

    Uses the shared format_json utility for consistent formatting,
    then optionally writes to the output path.
    """
    result = format_json(records)
    if output_path:
        with open(output_path, "w") as f:
            f.write(result)
    return result
