"""Pipeline configuration."""

from pathlib import Path
from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Runtime configuration for the pipeline."""

    batch_size: int = Field(default=100)
    input_path: Path = Field(default=Path("data/input"))
    output_path: Path = Field(default=Path("data/output"))
    log_level: str = Field(default="INFO")
    stage_order: list[str] = Field(default_factory=list)

