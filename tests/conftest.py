"""Shared pytest fixtures."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings

SETTING_ENVIRONMENT_VARIABLES = (
    "AERODOCS_APP_NAME",
    "AERODOCS_APP_VERSION",
    "AERODOCS_LOG_LEVEL",
    "AERODOCS_SECURITY_MODE",
    "AERODOCS_OIDC_ISSUER",
    "AERODOCS_OIDC_AUDIENCE",
    "AERODOCS_OIDC_JWKS_URL",
    "AERODOCS_OIDC_VIEWER_GROUP",
    "AERODOCS_OIDC_OPERATOR_GROUP",
    "AERODOCS_OIDC_CLOCK_SKEW_SECONDS",
    "AERODOCS_OIDC_MAXIMUM_TOKEN_LIFETIME_SECONDS",
    "AERODOCS_DOCUMENT_STORAGE_ROOT",
    "AERODOCS_MAX_PDF_UPLOAD_BYTES",
    "AERODOCS_MAX_REQUEST_BODY_BYTES",
    "AERODOCS_MAX_PDF_PAGES",
    "AERODOCS_MAX_EXTRACTED_CHARACTERS_PER_PAGE",
    "AERODOCS_MAX_EXTRACTED_CHARACTERS_TOTAL",
    "AERODOCS_ALLOWED_PDF_MEDIA_TYPE",
    "AERODOCS_EXTRACTION_FORMAT_VERSION",
    "AERODOCS_CHUNKING_FORMAT_VERSION",
    "AERODOCS_PROCESSING_FORMAT_VERSION",
    "AERODOCS_CHUNK_MAX_CHARACTERS",
    "AERODOCS_CHUNK_OVERLAP_CHARACTERS",
    "AERODOCS_EMBEDDING_MODEL_NAME",
    "AERODOCS_EMBEDDING_MODEL_REVISION",
    "AERODOCS_EMBEDDING_BATCH_SIZE",
    "AERODOCS_EMBEDDING_DEVICE",
    "AERODOCS_EMBEDDING_TORCH_NUM_THREADS",
    "AERODOCS_RETRIEVAL_DEFAULT_TOP_K",
    "AERODOCS_RETRIEVAL_MAXIMUM_TOP_K",
    "AERODOCS_MINIMUM_SIMILARITY_SCORE",
    "AERODOCS_INDEX_FORMAT_VERSION",
    "AERODOCS_EMBEDDING_NORMALIZATION",
    "AERODOCS_INDEX_RUNTIME_ROOT",
    "AERODOCS_MAXIMUM_CHUNKS_PER_INDEX_BUILD",
    "AERODOCS_MAXIMUM_TOTAL_EMBEDDING_CHARACTERS_PER_BUILD",
    "AERODOCS_MAXIMUM_QUERY_CHARACTERS",
    "AERODOCS_INDEX_BUILD_LOCK_TIMEOUT_SECONDS",
    "AERODOCS_GENERATION_MODEL_NAME",
    "AERODOCS_GENERATION_MODEL_REVISION",
    "AERODOCS_GENERATION_DEVICE",
    "AERODOCS_ANSWER_DEFAULT_TOP_K",
    "AERODOCS_ANSWER_MAXIMUM_TOP_K",
    "AERODOCS_ANSWER_MINIMUM_SIMILARITY_SCORE",
    "AERODOCS_MAXIMUM_ANSWER_QUESTION_CHARACTERS",
    "AERODOCS_MAXIMUM_ANSWER_CONTEXT_CHUNKS",
    "AERODOCS_MAXIMUM_ANSWER_CONTEXT_CHARACTERS",
    "AERODOCS_MAXIMUM_ANSWER_EVIDENCE_CHARACTERS",
    "AERODOCS_MAXIMUM_GENERATION_PROMPT_TOKENS",
    "AERODOCS_MAXIMUM_GENERATION_OUTPUT_TOKENS",
    "AERODOCS_MAXIMUM_GENERATION_CLAIMS",
    "AERODOCS_MAXIMUM_GENERATION_CLAIM_CHARACTERS",
    "AERODOCS_MAXIMUM_GENERATION_SUPPORTS_PER_CLAIM",
    "AERODOCS_MAXIMUM_GENERATION_QUOTE_CHARACTERS",
    "AERODOCS_MAXIMUM_GENERATION_CONFLICT_EVIDENCE_IDS",
    "AERODOCS_MINIMUM_CLAIM_LEXICAL_SUPPORT",
    "AERODOCS_GENERATION_TIMEOUT_SECONDS",
    "AERODOCS_GENERATION_FORMAT_VERSION",
    "AERODOCS_GENERATION_AUDIT_FORMAT_VERSION",
    "AERODOCS_GENERATION_AUDIT_ROOT",
    "AERODOCS_UI_ENABLED",
    "AERODOCS_UI_TITLE",
    "AERODOCS_UI_DEFAULT_PAGE_SIZE",
    "AERODOCS_UI_MAXIMUM_PAGE_SIZE",
    "AERODOCS_MCP_SERVER_NAME",
    "AERODOCS_MCP_SERVER_VERSION",
    "AERODOCS_MCP_TRANSPORT",
    "AERODOCS_MCP_DEFAULT_PAGINATION_LIMIT",
    "AERODOCS_MCP_MAXIMUM_PAGINATION_LIMIT",
    "AERODOCS_MCP_MAXIMUM_TOOL_RESULT_CHARACTERS",
)


@pytest.fixture
def default_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Settings:
    """Return settings isolated from local environment and dotenv files."""
    monkeypatch.chdir(tmp_path)
    for variable in SETTING_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    # `_env_file` is supported at runtime but absent from the generated signature.
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def client(default_settings: Settings) -> Iterator[TestClient]:
    """Return an HTTP client for an isolated application instance."""
    from app.factory import create_app

    with TestClient(create_app(default_settings)) as test_client:
        yield test_client


@pytest.fixture
def make_pdf() -> Callable[..., bytes]:
    """Return a generated-PDF factory; no binary fixture touches the repository."""

    def factory(
        pages: list[str],
        *,
        metadata: dict[str, str] | None = None,
        encrypted: bool = False,
    ) -> bytes:
        document = pymupdf.open()  # type: ignore[no-untyped-call]
        try:
            for text in pages:
                page = document.new_page()
                if text:
                    page.insert_text((72, 72), text)
            if metadata:
                document.set_metadata(metadata)
            if encrypted:
                return cast(
                    bytes,
                    document.tobytes(  # type: ignore[no-untyped-call]
                        encryption=pymupdf.PDF_ENCRYPT_AES_256,  # type: ignore[attr-defined]
                        owner_pw="owner-password",
                        user_pw="user-password",
                    ),
                )
            return cast(
                bytes,
                document.tobytes(),  # type: ignore[no-untyped-call]
            )
        finally:
            document.close()  # type: ignore[no-untyped-call]

    return factory
