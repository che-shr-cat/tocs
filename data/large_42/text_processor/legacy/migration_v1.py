"""One-time migration script: pipeline v0 -> v1 format.

This script was used during the architecture refactor to convert
old-format data files to the new Pydantic model format. It is
no longer needed for normal operation.
"""

from ..models import *  # noqa: F401,F403


def migrate_v0_to_v1(input_path: str, output_path: str) -> int:
    """Migrate data files from v0 flat-dict format to v1 Pydantic format.

    Returns:
        Number of records migrated.
    """
    # In production, this read old JSON files line-by-line,
    # converted each record, and wrote new typed JSON.
    # Kept for reference in case a rollback is needed.
    return 0
