"""Application configuration loaded from environment variables."""

import re
import unicodedata
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.generation.limits import (
    MAX_DRAFT_CLAIM_CHARACTERS,
    MAX_DRAFT_CLAIMS,
    MAX_DRAFT_CONFLICT_EVIDENCE_IDS,
    MAX_DRAFT_QUOTE_CHARACTERS,
    MAX_DRAFT_SUPPORTS_PER_CLAIM,
)
from app.ingestion.limits import MAXIMUM_DOCUMENT_LIST_LIMIT

LogLevel = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
SecurityMode = Literal["disabled", "test", "oidc"]
_NON_PRODUCTION_OIDC_IDENTIFIERS = frozenset(
    {
        "*",
        "aerodocs-api",
        "demo",
        "demonstration",
        "example",
        "fixture",
        "fixture-operators",
        "fixture-viewers",
        "operator",
        "organisation-operator-group",
        "organisation-viewer-group",
        "placeholder",
        "sample",
        "test",
        "testing",
        "todo",
        "viewer",
        "wrong-api",
    }
)
_WILDCARD_IDENTIFIER_CHARACTERS = frozenset("*?[]")
PINNED_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PINNED_EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
PINNED_GENERATION_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
PINNED_GENERATION_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


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
    security_mode: SecurityMode = "disabled"
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_url: str | None = None
    oidc_viewer_group: str | None = None
    oidc_operator_group: str | None = None
    oidc_clock_skew_seconds: int = 60
    oidc_maximum_token_lifetime_seconds: int = 15 * 60
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
    embedding_model_name: str = PINNED_EMBEDDING_MODEL_NAME
    embedding_model_revision: str = PINNED_EMBEDDING_MODEL_REVISION
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
    generation_model_name: str = PINNED_GENERATION_MODEL_NAME
    generation_model_revision: str = PINNED_GENERATION_MODEL_REVISION
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
    ui_enabled: bool = True
    ui_title: str = "AeroDocs Evidence Review"
    ui_default_page_size: int = 20
    ui_maximum_page_size: int = 100
    mcp_server_name: str = "AeroDocs"
    mcp_server_version: str = "0.1.0"
    mcp_transport: Literal["stdio"] = "stdio"
    mcp_default_pagination_limit: int = 20
    mcp_maximum_pagination_limit: int = 100
    mcp_maximum_tool_result_characters: int = 131_072

    @model_validator(mode="after")
    def validate_ingestion_limits(self) -> "Settings":
        """Validate bounded ingestion and chunking configuration."""
        self.validate_http_security()
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
        if not 1 <= len(self.ui_title.strip()) <= 80:
            raise ValueError("ui_title must contain between 1 and 80 characters")
        if not 1 <= self.ui_default_page_size <= self.ui_maximum_page_size:
            raise ValueError(
                "ui_default_page_size must be between 1 and ui_maximum_page_size"
            )
        if not 1 <= self.ui_maximum_page_size <= MAXIMUM_DOCUMENT_LIST_LIMIT:
            raise ValueError("ui_maximum_page_size exceeds the document-list limit")
        if not self.mcp_server_name.strip():
            raise ValueError("mcp_server_name must not be empty")
        if not self.mcp_server_version.strip():
            raise ValueError("mcp_server_version must not be empty")
        if self.mcp_transport != "stdio":
            raise ValueError("mcp_transport must be stdio")
        if self.mcp_default_pagination_limit <= 0:
            raise ValueError("mcp_default_pagination_limit must be positive")
        if self.mcp_maximum_pagination_limit < self.mcp_default_pagination_limit:
            raise ValueError(
                "mcp_maximum_pagination_limit must be at least "
                "mcp_default_pagination_limit"
            )
        minimum = self._minimum_safe_mcp_tool_result_characters()
        if self.mcp_maximum_tool_result_characters < minimum:
            raise ValueError(
                f"mcp_maximum_tool_result_characters must be at least {minimum}"
            )
        return self

    def validate_http_security(self) -> None:
        """Validate only the closed HTTP-security configuration."""
        if self.oidc_clock_skew_seconds != 60:
            raise ValueError("oidc_clock_skew_seconds must be 60")
        if self.oidc_maximum_token_lifetime_seconds != 15 * 60:
            raise ValueError("oidc_maximum_token_lifetime_seconds must be 900")
        values = (
            self.oidc_issuer,
            self.oidc_audience,
            self.oidc_jwks_url,
            self.oidc_viewer_group,
            self.oidc_operator_group,
        )
        if self.security_mode != "oidc":
            if any(value is not None for value in values):
                raise ValueError("OIDC settings require security_mode=oidc")
            return
        if any(value is None for value in values):
            raise ValueError("oidc mode requires complete OIDC settings")
        issuer = self.oidc_issuer
        audience = self.oidc_audience
        jwks_url = self.oidc_jwks_url
        viewer = self.oidc_viewer_group
        operator = self.oidc_operator_group
        assert issuer is not None
        assert audience is not None
        assert jwks_url is not None
        assert viewer is not None
        assert operator is not None
        if len(issuer) > 2_048 or len(jwks_url) > 2_048:
            raise ValueError("OIDC URLs exceed the configured bound")
        if not _safe_deployment_identifier(audience, maximum_characters=256):
            raise ValueError("oidc_audience is invalid")
        if not _safe_group_mapping(viewer) or not _safe_group_mapping(operator):
            raise ValueError("OIDC group mappings are invalid")
        if (
            viewer != viewer.strip()
            or operator != operator.strip()
            or viewer == operator
        ):
            raise ValueError("OIDC group mappings must be distinct")
        issuer_url = urlsplit(issuer)
        jwks = urlsplit(jwks_url)
        if (
            issuer_url.scheme.lower() != "https"
            or jwks.scheme.lower() != "https"
            or not issuer_url.hostname
            or not jwks.hostname
            or issuer_url.username is not None
            or issuer_url.password is not None
            or jwks.username is not None
            or jwks.password is not None
            or issuer_url.fragment
            or jwks.fragment
            or jwks.query
        ):
            raise ValueError("OIDC issuer and JWKS URL must be exact HTTPS URLs")
        issuer_origin = (
            issuer_url.scheme.lower(),
            issuer_url.hostname.lower(),
            issuer_url.port or 443,
        )
        jwks_origin = (
            jwks.scheme.lower(),
            jwks.hostname.lower(),
            jwks.port or 443,
        )
        if issuer_origin != jwks_origin:
            raise ValueError("OIDC issuer and JWKS origins must match")

    @staticmethod
    def _minimum_safe_mcp_tool_result_characters() -> int:
        from app.mcp_server.results import MINIMUM_SAFE_TOOL_RESULT_CHARACTERS

        return MINIMUM_SAFE_TOOL_RESULT_CHARACTERS


def _safe_group_mapping(value: str) -> bool:
    return _safe_deployment_identifier(
        value,
        maximum_characters=128,
        maximum_bytes=512,
    )


def _safe_deployment_identifier(
    value: str,
    *,
    maximum_characters: int,
    maximum_bytes: int = 1_024,
) -> bool:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return (
        1 <= len(value) <= maximum_characters
        and len(value.encode("utf-8")) <= maximum_bytes
        and value == value.strip()
        and normalized not in _NON_PRODUCTION_OIDC_IDENTIFIERS
        and not (_WILDCARD_IDENTIFIER_CHARACTERS & set(value))
        and all(
            unicodedata.category(character) not in {"Cc", "Cf", "Cs", "Cn"}
            for character in value
        )
    )
