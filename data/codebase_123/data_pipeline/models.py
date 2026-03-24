"""Shared data types for the pipeline."""

from pydantic import BaseModel, Field


class RawRecord(BaseModel):
    """Data model: RawRecord."""

    id: str = Field(default="", description="Record identifier")
    data: dict = Field(default_factory=dict)


class IngestedRecord(BaseModel):
    """Data model: IngestedRecord."""

    id: str = Field(default="", description="Record identifier")
    data: dict = Field(default_factory=dict)


class ValidatedRecord(BaseModel):
    """Data model: ValidatedRecord."""

    id: str = Field(default="", description="Record identifier")
    data: dict = Field(default_factory=dict)


class DeduplicatedRecord(BaseModel):
    """Data model: DeduplicatedRecord."""

    id: str = Field(default="", description="Record identifier")
    data: dict = Field(default_factory=dict)


class EnrichedRecord(BaseModel):
    """Data model: EnrichedRecord."""

    id: str = Field(default="", description="Record identifier")
    data: dict = Field(default_factory=dict)


class TransformedRecord(BaseModel):
    """Data model: TransformedRecord."""

    id: str = Field(default="", description="Record identifier")
    data: dict = Field(default_factory=dict)


class AggregatedRecord(BaseModel):
    """Data model: AggregatedRecord."""

    id: str = Field(default="", description="Record identifier")
    data: dict = Field(default_factory=dict)


class ExportedRecord(BaseModel):
    """Data model: ExportedRecord."""

    id: str = Field(default="", description="Record identifier")
    data: dict = Field(default_factory=dict)

