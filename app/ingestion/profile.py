"""Deterministic document-processing provenance."""

import hashlib
import json

from app.core.config import Settings
from app.ingestion.models import ProcessingProfile


def create_processing_profile(settings: Settings) -> ProcessingProfile:
    """Create a deterministic processing profile from content-affecting settings."""
    return ProcessingProfile(
        processing_profile_id=calculate_processing_profile_id(
            processing_format_version=settings.processing_format_version,
            extraction_format_version=settings.extraction_format_version,
            chunking_format_version=settings.chunking_format_version,
            chunk_max_characters=settings.chunk_max_characters,
            chunk_overlap_characters=settings.chunk_overlap_characters,
        ),
        processing_format_version=settings.processing_format_version,
        extraction_format_version=settings.extraction_format_version,
        chunking_format_version=settings.chunking_format_version,
        chunk_max_characters=settings.chunk_max_characters,
        chunk_overlap_characters=settings.chunk_overlap_characters,
    )


def calculate_processing_profile_id(
    *,
    processing_format_version: str,
    extraction_format_version: str,
    chunking_format_version: str,
    chunk_max_characters: int,
    chunk_overlap_characters: int,
) -> str:
    """Hash all configuration that changes persisted processing results."""
    values = {
        "processing_format_version": processing_format_version,
        "extraction_format_version": extraction_format_version,
        "chunking_format_version": chunking_format_version,
        "chunk_max_characters": chunk_max_characters,
        "chunk_overlap_characters": chunk_overlap_characters,
    }
    encoded = json.dumps(values, separators=(",", ":"), sort_keys=True).encode()
    return f"prf_{hashlib.sha256(encoded).hexdigest()}"
