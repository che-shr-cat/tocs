"""Shared data types for the pipeline."""

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    """Data model: RawDocument."""

    id: str = Field(default="", description="Record identifier")
    data: dict = Field(default_factory=dict)


class TokenizedDocument(BaseModel):
    """Data model: TokenizedDocument."""

    id: str = Field(default="", description="Record identifier")
    data: dict = Field(default_factory=dict)


class FilteredDocument(BaseModel):
    """Data model: FilteredDocument."""

    id: str = Field(default="", description="Record identifier")
    data: dict = Field(default_factory=dict)


class VectorizedDocument(BaseModel):
    """Data model: VectorizedDocument."""

    id: str = Field(default="", description="Record identifier")
    data: dict = Field(default_factory=dict)


class ClassifiedDocument(BaseModel):
    """Data model: ClassifiedDocument."""

    id: str = Field(default="", description="Record identifier")
    data: dict = Field(default_factory=dict)


class SummarizedDocument(BaseModel):
    """Data model: SummarizedDocument."""

    id: str = Field(default="", description="Record identifier")
    data: dict = Field(default_factory=dict)


class IndexedDocument(BaseModel):
    """Data model: IndexedDocument."""

    id: str = Field(default="", description="Record identifier")
    data: dict = Field(default_factory=dict)

