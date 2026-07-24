"""Typed retrieval, index, and API models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ModelIdentity(BaseModel):
    name: str
    revision: str | None
    dimension: int = Field(gt=0)
    maximum_sequence_length: int | None = Field(default=None, gt=0)


class SourceSummary(BaseModel):
    document_id: str
    processing_profile_id: str
    manifest_sha256: str
    chunk_count: int = Field(ge=0)
    total_characters: int = Field(ge=0)
    filename: str
    title: str


class IndexRecord(BaseModel):
    row: int = Field(ge=0)
    document_id: str
    processing_profile_id: str
    filename: str
    title: str
    chunk_id: str
    chunk_order: int = Field(ge=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_heading: str | None
    text: str = Field(min_length=1)
    character_count: int = Field(gt=0)


class IndexArtifact(BaseModel):
    filename: str
    sha256: str
    byte_size: int = Field(ge=0)


class IndexMetadata(BaseModel):
    index_profile_id: str
    vector_profile_id: str
    corpus_fingerprint: str
    index_format_version: str
    model: ModelIdentity
    index_type: Literal["IndexFlatIP", "IndexFlatL2"]
    similarity_metric: Literal["cosine", "inverse_squared_l2"]
    embedding_normalization: bool
    vector_count: int = Field(ge=1)
    indexed_chunk_count: int = Field(ge=1)
    indexed_character_count: int = Field(ge=1)
    source_document_count: int = Field(ge=1)
    zero_chunk_document_ids: list[str]
    sources: list[SourceSummary]
    built_at: datetime


class IndexManifest(IndexMetadata):
    manifest_format_version: Literal["1"]
    version_id: str
    artifacts: dict[str, IndexArtifact]


class ActivePointer(BaseModel):
    version_id: str
    manifest_sha256: str


class IndexBuildResponse(BaseModel):
    rebuilt: bool
    index: IndexMetadata


class IndexPublication(BaseModel):
    """Validated stored state selected for active publication."""

    created: bool
    manifest: IndexManifest


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None
    document_ids: list[str] | None = None
    minimum_score: float | None = None


class SearchResult(BaseModel):
    rank: int = Field(ge=1)
    score: float
    document_id: str
    filename: str
    title: str
    chunk_id: str
    chunk_order: int
    page_start: int
    page_end: int
    section_heading: str | None
    text: str
    character_count: int


class SearchResponse(BaseModel):
    query: str
    index_profile_id: str
    corpus_fingerprint: str
    similarity_metric: str
    requested_top_k: int
    returned_count: int
    results: list[SearchResult]
