"""Data models shared across all pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Record:
    """Base record flowing through the pipeline.

    Attributes:
        id: Unique record identifier.
        payload: Arbitrary key-value data.
        metadata: Processing metadata added by stages.
        is_validated: Set to True by the validation stage.
    """

    id: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    is_validated: bool = False


@dataclass
class PipelineConfig:
    """Runtime configuration for the pipeline."""

    batch_size: int = 100
    input_path: str = "input.csv"
    output_path: str = "output.json"
    strict_mode: bool = True
