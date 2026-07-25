"""Typed ingestion, persistence, and API models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DocumentId = str
ChunkId = str
ExtractionWarning = Literal["no_extractable_text"]


class ProcessingProfile(BaseModel):
    """Versioned settings that determine persisted extraction and chunks."""

    processing_profile_id: str
    processing_format_version: str
    extraction_format_version: str
    chunking_format_version: str
    chunk_max_characters: int = Field(gt=0)
    chunk_overlap_characters: int = Field(ge=0)


class DocumentMetadata(BaseModel):
    """Traceable metadata for an ingested PDF."""

    document_id: DocumentId
    filename: str
    sha256: str
    title: str
    author: str | None
    subject: str | None
    page_count: int = Field(ge=1)
    extracted_character_count: int = Field(ge=0)
    ingested_at: datetime
    extraction_format_version: str
    chunking_format_version: str
    processing_format_version: str
    processing_profile_id: str
    chunk_max_characters: int = Field(gt=0)
    chunk_overlap_characters: int = Field(ge=0)
    encrypted: Literal[False]
    file_size_bytes: int = Field(gt=0)


class ArtifactDescriptor(BaseModel):
    """Integrity information for one persisted artifact."""

    filename: str
    sha256: str
    byte_size: int = Field(ge=0)


class DocumentManifest(BaseModel):
    """Versioned completeness and integrity record written last."""

    manifest_format_version: Literal["1"]
    document_id: DocumentId
    source_sha256: str
    processing_profile_id: str
    processing_format_version: str
    extraction_format_version: str
    chunking_format_version: str
    chunk_max_characters: int = Field(gt=0)
    chunk_overlap_characters: int = Field(ge=0)
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=0)
    artifacts: dict[str, ArtifactDescriptor]


class PageRecord(BaseModel):
    """Normalized text and provenance for one PDF page."""

    document_id: DocumentId
    page_number: int = Field(ge=1)
    text: str
    character_count: int = Field(ge=0)
    warnings: list[ExtractionWarning] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    """Deterministic text chunk with source-page context."""

    chunk_id: ChunkId
    document_id: DocumentId
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_heading: str | None
    text: str = Field(min_length=1)
    character_count: int = Field(gt=0)
    chunk_order: int = Field(ge=1)


class DocumentIngestResponse(BaseModel):
    """Result of a new or duplicate document upload."""

    document: DocumentMetadata
    duplicate: bool


class DocumentSummary(BaseModel):
    """Bounded safe metadata for document discovery."""

    document_id: DocumentId
    filename: str
    title: str
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=0)
    processing_profile: ProcessingProfile
    ingested_at: datetime


class DocumentListResponse(BaseModel):
    """Deterministically ordered public document summaries."""

    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    total: int = Field(ge=0)
    documents: list[DocumentSummary]


class PageListResponse(BaseModel):
    """Deterministically ordered page records."""

    document_id: DocumentId
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    pages: list[PageRecord]


class ChunkListResponse(BaseModel):
    """Deterministically ordered chunk records."""

    document_id: DocumentId
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    chunks: list[ChunkRecord]
