"""Application configuration loaded from environment variables."""

import re
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.generation.limits import (
    MAX_DRAFT_CLAIM_CHARACTERS,
    MAX_DRAFT_CLAIMS,
    MAX_DRAFT_CONFLICT_EVIDENCE_IDS,
    MAX_DRAFT_QUOTE_CHARACTERS,
    MAX_DRAFT_SUPPORTS_PER_CLAIM,
)

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
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
    embedding_torch_num_threads: int = 1
    retrieval_default_top_k: int = 5
    retrieval_maximum_top_k: int = 20
    minimum_similarity_score: float = 0.0
    index_format_version: str = "1"
    embedding_normalization: bool = True
    index_runtime_root: Path = Path("data/runtime/indexes")
    maximum_chunks_per_index_build: int = 50_000
    maximum_total_embedding_characters_per_build: int = 50_000_000
    maximum_query_characters: int = 2_000
    index_build_lock_timeout_seconds: float = 30.0
    generation_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    generation_model_revision: str = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    generation_device: str = "cpu"
    answer_default_top_k: int = 5
    answer_maximum_top_k: int = 10
    answer_minimum_similarity_score: float = 0.35
    maximum_answer_question_characters: int = 2_000
    maximum_answer_context_chunks: int = 8
    maximum_answer_context_characters: int = 16_000
    maximum_answer_evidence_characters: int = 3_000
    maximum_generation_prompt_tokens: int = 4_096
    maximum_generation_output_tokens: int = 256
    maximum_generation_claims: int = 8
    maximum_generation_claim_characters: int = 800
    maximum_generation_supports_per_claim: int = 8
    maximum_generation_quote_characters: int = 500
    maximum_generation_conflict_evidence_ids: int = 8
    minimum_claim_lexical_support: float = 0.60
    generation_timeout_seconds: float = 120.0
    generation_format_version: str = "1"
    generation_audit_format_version: str = "1"
    generation_audit_root: Path = Path("data/runtime/generation/audit")

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
        if not self.embedding_model_name.strip():
            raise ValueError("embedding_model_name must not be empty")
        if not re.fullmatch(r"[0-9a-f]{40}", self.embedding_model_revision):
            raise ValueError(
                "embedding_model_revision must be an immutable 40-character SHA"
            )
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")
        if not self.embedding_device.strip():
            raise ValueError("embedding_device must not be empty")
        if self.embedding_torch_num_threads <= 0:
            raise ValueError("embedding_torch_num_threads must be positive")
        if self.retrieval_default_top_k <= 0:
            raise ValueError("retrieval_default_top_k must be positive")
        if self.retrieval_maximum_top_k < self.retrieval_default_top_k:
            raise ValueError(
                "retrieval_maximum_top_k must be at least retrieval_default_top_k"
            )
        minimum_score = -1.0 if self.embedding_normalization else 0.0
        if not minimum_score <= self.minimum_similarity_score <= 1.0:
            raise ValueError("minimum_similarity_score is outside the metric range")
        if not self.index_format_version.strip():
            raise ValueError("index_format_version must not be empty")
        if self.maximum_chunks_per_index_build <= 0:
            raise ValueError("maximum_chunks_per_index_build must be positive")
        if self.maximum_total_embedding_characters_per_build <= 0:
            raise ValueError(
                "maximum_total_embedding_characters_per_build must be positive"
            )
        if self.maximum_query_characters <= 0:
            raise ValueError("maximum_query_characters must be positive")
        if self.index_build_lock_timeout_seconds <= 0:
            raise ValueError("index_build_lock_timeout_seconds must be positive")
        if not self.generation_model_name.strip():
            raise ValueError("generation_model_name must not be empty")
        if not re.fullmatch(r"[0-9a-f]{40}", self.generation_model_revision):
            raise ValueError(
                "generation_model_revision must be an immutable 40-character SHA"
            )
        if self.generation_device != "cpu":
            raise ValueError("generation_device must be cpu")
        if self.answer_default_top_k <= 0:
            raise ValueError("answer_default_top_k must be positive")
        if self.answer_maximum_top_k < self.answer_default_top_k:
            raise ValueError(
                "answer_maximum_top_k must be at least answer_default_top_k"
            )
        if self.answer_maximum_top_k > self.retrieval_maximum_top_k:
            raise ValueError(
                "answer_maximum_top_k must not exceed retrieval_maximum_top_k"
            )
        structured_limits = (
            (
                "maximum_generation_claim_characters",
                self.maximum_generation_claim_characters,
                MAX_DRAFT_CLAIM_CHARACTERS,
            ),
            (
                "maximum_generation_claims",
                self.maximum_generation_claims,
                MAX_DRAFT_CLAIMS,
            ),
            (
                "maximum_generation_supports_per_claim",
                self.maximum_generation_supports_per_claim,
                MAX_DRAFT_SUPPORTS_PER_CLAIM,
            ),
            (
                "maximum_generation_quote_characters",
                self.maximum_generation_quote_characters,
                MAX_DRAFT_QUOTE_CHARACTERS,
            ),
            (
                "maximum_generation_conflict_evidence_ids",
                self.maximum_generation_conflict_evidence_ids,
                MAX_DRAFT_CONFLICT_EVIDENCE_IDS,
            ),
        )
        for name, value, ceiling in structured_limits:
            if not 0 < value <= ceiling:
                raise ValueError(f"{name} must be between 1 and {ceiling}")
        answer_minimum = -1.0 if self.embedding_normalization else 0.0
        if not answer_minimum <= self.answer_minimum_similarity_score <= 1.0:
            raise ValueError(
                "answer_minimum_similarity_score is outside the metric range"
            )
        positive_limits = {
            "maximum_answer_question_characters": (
                self.maximum_answer_question_characters
            ),
            "maximum_answer_context_chunks": self.maximum_answer_context_chunks,
            "maximum_answer_context_characters": (
                self.maximum_answer_context_characters
            ),
            "maximum_answer_evidence_characters": (
                self.maximum_answer_evidence_characters
            ),
            "maximum_generation_prompt_tokens": (self.maximum_generation_prompt_tokens),
            "maximum_generation_output_tokens": (self.maximum_generation_output_tokens),
            "maximum_generation_claims": self.maximum_generation_claims,
            "maximum_generation_claim_characters": (
                self.maximum_generation_claim_characters
            ),
            "maximum_generation_quote_characters": (
                self.maximum_generation_quote_characters
            ),
        }
        if any(value <= 0 for value in positive_limits.values()):
            raise ValueError("generation resource limits must be positive")
        if not 0.0 <= self.minimum_claim_lexical_support <= 1.0:
            raise ValueError("minimum_claim_lexical_support must be between 0 and 1")
        if self.generation_timeout_seconds <= 0:
            raise ValueError("generation_timeout_seconds must be positive")
        if not self.generation_format_version.strip():
            raise ValueError("generation_format_version must not be empty")
        if not self.generation_audit_format_version.strip():
            raise ValueError("generation_audit_format_version must not be empty")
        return self
