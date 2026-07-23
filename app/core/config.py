"""Application configuration loaded from environment variables."""

from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


class Settings(BaseSettings):
    """AeroDocs settings with safe local defaults."""

    model_config = SettingsConfigDict(
        env_prefix="AERODOCS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    app_name: str = "AeroDocs"
    app_version: str = "0.1.0"
    log_level: LogLevel = "INFO"
    document_storage_root: Path = Path("data/runtime")
    max_pdf_upload_bytes: int = 25 * 1024 * 1024
    max_request_body_bytes: int = 27 * 1024 * 1024
    max_pdf_pages: int = 1_000
    max_extracted_characters_per_page: int = 500_000
    max_extracted_characters_total: int = 10_000_000
    allowed_pdf_media_type: str = "application/pdf"
    extraction_format_version: str = "1"
    chunking_format_version: str = "1"
    processing_format_version: str = "1"
    chunk_max_characters: int = 2_000
    chunk_overlap_characters: int = 200

    @model_validator(mode="after")
    def validate_ingestion_limits(self) -> "Settings":
        """Validate bounded ingestion and chunking configuration."""
        if self.max_pdf_upload_bytes <= 0:
            raise ValueError("max_pdf_upload_bytes must be positive")
        if self.max_request_body_bytes <= self.max_pdf_upload_bytes:
            raise ValueError(
                "max_request_body_bytes must be greater than max_pdf_upload_bytes"
            )
        if self.max_pdf_pages <= 0:
            raise ValueError("max_pdf_pages must be positive")
        if self.max_extracted_characters_per_page <= 0:
            raise ValueError("max_extracted_characters_per_page must be positive")
        if self.max_extracted_characters_total <= 0:
            raise ValueError("max_extracted_characters_total must be positive")
        if self.chunk_max_characters <= 0:
            raise ValueError("chunk_max_characters must be positive")
        if not 0 <= self.chunk_overlap_characters < self.chunk_max_characters:
            raise ValueError(
                "chunk_overlap_characters must be non-negative and smaller than "
                "chunk_max_characters"
            )
        if not self.allowed_pdf_media_type.strip():
            raise ValueError("allowed_pdf_media_type must not be empty")
        if not self.extraction_format_version.strip():
            raise ValueError("extraction_format_version must not be empty")
        if not self.chunking_format_version.strip():
            raise ValueError("chunking_format_version must not be empty")
        if not self.processing_format_version.strip():
            raise ValueError("processing_format_version must not be empty")
        return self
