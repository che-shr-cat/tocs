"""CSV serializer for pipeline output."""

from typing import Any

from ..models import BaseModel
from ..utils.formatters import format_csv


def serialize_csv(records: list[Any], output_path: str = "") -> str:
    """Serialize pipeline output as CSV.

    Uses the shared format_csv utility for consistent formatting,
    then optionally writes to the output path.
    """
    result = format_csv(records)
    if output_path:
        with open(output_path, "w") as f:
            f.write(result)
    return result
