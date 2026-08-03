"""Tests for application settings."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.generation.limits import (
    MAX_DRAFT_CLAIM_CHARACTERS,
    MAX_DRAFT_CLAIMS,
    MAX_DRAFT_CONFLICT_EVIDENCE_IDS,
    MAX_DRAFT_QUOTE_CHARACTERS,
    MAX_DRAFT_SUPPORTS_PER_CLAIM,
)
from app.mcp_server.results import MINIMUM_SAFE_TOOL_RESULT_CHARACTERS


def test_default_settings(default_settings: Settings) -> None:
    assert default_settings.app_name == "AeroDocs"
    assert default_settings.app_version == "0.1.0"
    assert default_settings.log_level == "INFO"
    assert default_settings.security_mode == "disabled"
    assert default_settings.document_storage_root == Path("data/runtime")
    assert default_settings.max_pdf_upload_bytes == 25 * 1024 * 1024
    assert default_settings.max_request_body_bytes == 27 * 1024 * 1024
    assert default_settings.max_pdf_pages == 1_000
    assert default_settings.max_extracted_characters_per_page == 500_000
    assert default_settings.max_extracted_characters_total == 10_000_000
    assert default_settings.allowed_pdf_media_type == "application/pdf"
    assert default_settings.extraction_format_version == "1"
    assert default_settings.chunking_format_version == "1"
    assert default_settings.processing_format_version == "1"
    assert default_settings.chunk_max_characters == 2_000
    assert default_settings.chunk_overlap_characters == 200
    assert default_settings.embedding_model_name.endswith("all-MiniLM-L6-v2")
    assert (
        default_settings.embedding_model_revision
        == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    assert default_settings.embedding_batch_size == 32
    assert default_settings.embedding_torch_num_threads == 1
    assert default_settings.retrieval_default_top_k == 5
    assert default_settings.retrieval_maximum_top_k == 20
    assert default_settings.embedding_normalization is True
    assert default_settings.index_runtime_root == Path("data/runtime/indexes")
    assert default_settings.maximum_chunks_per_index_build == 50_000
    assert default_settings.generation_model_name.endswith("Qwen2.5-1.5B-Instruct")
    assert (
        default_settings.generation_model_revision
        == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    )
    assert default_settings.generation_device == "cpu"
    assert default_settings.mcp_server_name == "AeroDocs"
    assert default_settings.mcp_server_version == "0.1.0"
    assert default_settings.mcp_transport == "stdio"
    assert default_settings.mcp_default_pagination_limit == 20
    assert default_settings.mcp_maximum_pagination_limit == 100
    assert default_settings.mcp_maximum_tool_result_characters == 131_072
    assert default_settings.answer_default_top_k == 5
    assert default_settings.answer_maximum_top_k == 10
    assert default_settings.generation_audit_root == Path(
        "data/runtime/generation/audit"
    )
    assert default_settings.ui_enabled is True
    assert default_settings.ui_title == "AeroDocs Evidence Review"
    assert default_settings.ui_default_page_size == 20
    assert default_settings.ui_maximum_page_size == 100


def test_environment_variables_override_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AERODOCS_APP_NAME", "AeroDocs Test")
    monkeypatch.setenv("AERODOCS_APP_VERSION", "9.9.9")
    monkeypatch.setenv("AERODOCS_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AERODOCS_DOCUMENT_STORAGE_ROOT", "custom/runtime")
    monkeypatch.setenv("AERODOCS_MAX_PDF_UPLOAD_BYTES", "12345")
    monkeypatch.setenv("AERODOCS_MAX_REQUEST_BODY_BYTES", "13000")
    monkeypatch.setenv("AERODOCS_MAX_PDF_PAGES", "12")
    monkeypatch.setenv("AERODOCS_MAX_EXTRACTED_CHARACTERS_PER_PAGE", "345")
    monkeypatch.setenv("AERODOCS_MAX_EXTRACTED_CHARACTERS_TOTAL", "678")
    monkeypatch.setenv("AERODOCS_ALLOWED_PDF_MEDIA_TYPE", "application/test-pdf")
    monkeypatch.setenv("AERODOCS_EXTRACTION_FORMAT_VERSION", "test-v2")
    monkeypatch.setenv("AERODOCS_CHUNKING_FORMAT_VERSION", "chunk-v2")
    monkeypatch.setenv("AERODOCS_PROCESSING_FORMAT_VERSION", "process-v2")
    monkeypatch.setenv("AERODOCS_CHUNK_MAX_CHARACTERS", "800")
    monkeypatch.setenv("AERODOCS_CHUNK_OVERLAP_CHARACTERS", "80")
    monkeypatch.setenv("AERODOCS_EMBEDDING_MODEL_NAME", "local/test-model")
    monkeypatch.setenv("AERODOCS_EMBEDDING_MODEL_REVISION", "a" * 40)
    monkeypatch.setenv("AERODOCS_EMBEDDING_BATCH_SIZE", "7")
    monkeypatch.setenv("AERODOCS_EMBEDDING_TORCH_NUM_THREADS", "2")
    monkeypatch.setenv("AERODOCS_RETRIEVAL_DEFAULT_TOP_K", "3")
    monkeypatch.setenv("AERODOCS_RETRIEVAL_MAXIMUM_TOP_K", "9")
    monkeypatch.setenv("AERODOCS_INDEX_RUNTIME_ROOT", "custom/indexes")
    monkeypatch.setenv("AERODOCS_GENERATION_MODEL_NAME", "local/generator")
    monkeypatch.setenv("AERODOCS_ANSWER_DEFAULT_TOP_K", "2")
    monkeypatch.setenv("AERODOCS_ANSWER_MAXIMUM_TOP_K", "6")
    monkeypatch.setenv("AERODOCS_GENERATION_AUDIT_ROOT", "custom/audits")
    monkeypatch.setenv("AERODOCS_UI_ENABLED", "false")
    monkeypatch.setenv("AERODOCS_UI_TITLE", "Evidence Desk")
    monkeypatch.setenv("AERODOCS_UI_DEFAULT_PAGE_SIZE", "7")
    monkeypatch.setenv("AERODOCS_UI_MAXIMUM_PAGE_SIZE", "30")
    monkeypatch.setenv("AERODOCS_MCP_SERVER_NAME", "AeroDocs Test MCP")
    monkeypatch.setenv("AERODOCS_MCP_SERVER_VERSION", "9.8.7")
    monkeypatch.setenv("AERODOCS_MCP_DEFAULT_PAGINATION_LIMIT", "4")
    monkeypatch.setenv("AERODOCS_MCP_MAXIMUM_PAGINATION_LIMIT", "12")
    monkeypatch.setenv("AERODOCS_MCP_MAXIMUM_TOOL_RESULT_CHARACTERS", "2048")

    settings = Settings()

    assert settings.app_name == "AeroDocs Test"
    assert settings.app_version == "9.9.9"
    assert settings.log_level == "DEBUG"
    assert settings.document_storage_root == Path("custom/runtime")
    assert settings.max_pdf_upload_bytes == 12_345
    assert settings.max_request_body_bytes == 13_000
    assert settings.max_pdf_pages == 12
    assert settings.max_extracted_characters_per_page == 345
    assert settings.max_extracted_characters_total == 678
    assert settings.allowed_pdf_media_type == "application/test-pdf"
    assert settings.extraction_format_version == "test-v2"
    assert settings.chunking_format_version == "chunk-v2"
    assert settings.processing_format_version == "process-v2"
    assert settings.chunk_max_characters == 800
    assert settings.chunk_overlap_characters == 80
    assert settings.embedding_model_name == "local/test-model"
    assert settings.embedding_model_revision == "a" * 40
    assert settings.embedding_batch_size == 7
    assert settings.embedding_torch_num_threads == 2
    assert settings.retrieval_default_top_k == 3
    assert settings.retrieval_maximum_top_k == 9
    assert settings.index_runtime_root == Path("custom/indexes")
    assert settings.generation_model_name == "local/generator"
    assert settings.answer_default_top_k == 2
    assert settings.answer_maximum_top_k == 6
    assert settings.generation_audit_root == Path("custom/audits")
    assert settings.ui_enabled is False
    assert settings.ui_title == "Evidence Desk"
    assert settings.ui_default_page_size == 7
    assert settings.ui_maximum_page_size == 30
    assert settings.mcp_server_name == "AeroDocs Test MCP"
    assert settings.mcp_server_version == "9.8.7"
    assert settings.mcp_default_pagination_limit == 4
    assert settings.mcp_maximum_pagination_limit == 12
    assert settings.mcp_maximum_tool_result_characters == 2048


def test_settings_are_immutable(default_settings: Settings) -> None:
    with pytest.raises(ValidationError):
        default_settings.app_name = "Changed"


def test_isolated_settings_import_ignores_dotenv_but_respects_environment(
    tmp_path: Path,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "AERODOCS_APP_NAME=Contaminated\nAERODOCS_APP_VERSION=9.9.9\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    for variable in (
        "AERODOCS_APP_NAME",
        "AERODOCS_APP_VERSION",
        "AERODOCS_LOG_LEVEL",
    ):
        environment.pop(variable, None)
    environment["AERODOCS_LOG_LEVEL"] = "DEBUG"
    environment["PYTHONPATH"] = str(Path(__file__).parents[1])

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from app.core.config import Settings; "
                "settings = Settings(_env_file=None); "
                "print(json.dumps(settings.model_dump(mode='json')))"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "app_name": "AeroDocs",
        "app_version": "0.1.0",
        "log_level": "DEBUG",
        "security_mode": "disabled",
        "oidc_issuer": None,
        "oidc_audience": None,
        "oidc_jwks_url": None,
        "oidc_viewer_group": None,
        "oidc_operator_group": None,
        "oidc_clock_skew_seconds": 60,
        "oidc_maximum_token_lifetime_seconds": 900,
        "document_storage_root": "data/runtime",
        "max_pdf_upload_bytes": 26214400,
        "max_request_body_bytes": 28311552,
        "max_pdf_pages": 1000,
        "max_extracted_characters_per_page": 500000,
        "max_extracted_characters_total": 10000000,
        "allowed_pdf_media_type": "application/pdf",
        "extraction_format_version": "1",
        "chunking_format_version": "1",
        "processing_format_version": "1",
        "chunk_max_characters": 2000,
        "chunk_overlap_characters": 200,
        "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_model_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "embedding_batch_size": 32,
        "embedding_device": "cpu",
        "embedding_torch_num_threads": 1,
        "retrieval_default_top_k": 5,
        "retrieval_maximum_top_k": 20,
        "minimum_similarity_score": 0.0,
        "index_format_version": "1",
        "embedding_normalization": True,
        "index_runtime_root": "data/runtime/indexes",
        "maximum_chunks_per_index_build": 50000,
        "maximum_total_embedding_characters_per_build": 50000000,
        "maximum_query_characters": 2000,
        "index_build_lock_timeout_seconds": 30.0,
        "generation_model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "generation_model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "generation_device": "cpu",
        "answer_default_top_k": 5,
        "answer_maximum_top_k": 10,
        "answer_minimum_similarity_score": 0.35,
        "maximum_answer_question_characters": 2000,
        "maximum_answer_context_chunks": 8,
        "maximum_answer_context_characters": 16000,
        "maximum_answer_evidence_characters": 3000,
        "maximum_generation_prompt_tokens": 4096,
        "maximum_generation_output_tokens": 256,
        "maximum_generation_claims": 8,
        "maximum_generation_claim_characters": 800,
        "maximum_generation_supports_per_claim": 8,
        "maximum_generation_quote_characters": 500,
        "maximum_generation_conflict_evidence_ids": 8,
        "minimum_claim_lexical_support": 0.6,
        "generation_timeout_seconds": 120.0,
        "generation_format_version": "1",
        "generation_audit_format_version": "1",
        "generation_audit_root": "data/runtime/generation/audit",
        "ui_enabled": True,
        "ui_title": "AeroDocs Evidence Review",
        "ui_default_page_size": 20,
        "ui_maximum_page_size": 100,
        "mcp_server_name": "AeroDocs",
        "mcp_server_version": "0.1.0",
        "mcp_transport": "stdio",
        "mcp_default_pagination_limit": 20,
        "mcp_maximum_pagination_limit": 100,
        "mcp_maximum_tool_result_characters": 131072,
    }


def test_invalid_chunk_configuration_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            document_storage_root=tmp_path,
            chunk_max_characters=100,
            chunk_overlap_characters=100,
        )  # type: ignore[call-arg]


def test_request_body_limit_must_exceed_pdf_limit(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            document_storage_root=tmp_path,
            max_pdf_upload_bytes=100,
            max_request_body_bytes=100,
        )  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "updates",
    [
        {"embedding_batch_size": 0},
        {"embedding_torch_num_threads": 0},
        {"retrieval_default_top_k": 0},
        {"retrieval_default_top_k": 5, "retrieval_maximum_top_k": 4},
        {"minimum_similarity_score": 2.0},
        {"maximum_chunks_per_index_build": 0},
        {"maximum_total_embedding_characters_per_build": 0},
        {"answer_default_top_k": 0},
        {"answer_default_top_k": 5, "answer_maximum_top_k": 4},
        {"answer_maximum_top_k": 21},
        {
            "embedding_normalization": False,
            "answer_minimum_similarity_score": -0.1,
        },
        {"maximum_generation_prompt_tokens": 0},
        {"minimum_claim_lexical_support": 1.1},
        {"generation_device": "mps"},
        {"embedding_model_revision": ""},
        {"embedding_model_revision": "main"},
        {"embedding_model_revision": "g" * 40},
        {"mcp_server_name": " "},
        {"mcp_server_version": ""},
        {"mcp_default_pagination_limit": 0},
        {
            "mcp_default_pagination_limit": 5,
            "mcp_maximum_pagination_limit": 4,
        },
        {"mcp_maximum_tool_result_characters": 0},
        {"ui_title": " "},
        {"ui_default_page_size": 0},
        {"ui_default_page_size": 10, "ui_maximum_page_size": 9},
        {"ui_maximum_page_size": 101},
    ],
)
def test_invalid_retrieval_settings_are_rejected(
    tmp_path: Path,
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            document_storage_root=tmp_path / "documents",
            index_runtime_root=tmp_path / "indexes",
            **updates,  # type: ignore[arg-type]
        )  # type: ignore[call-arg]


def test_mcp_minimum_safe_result_limit(tmp_path: Path) -> None:
    accepted = Settings(
        _env_file=None,  # type: ignore[call-arg]
        document_storage_root=tmp_path / "documents",
        index_runtime_root=tmp_path / "indexes",
        mcp_maximum_tool_result_characters=MINIMUM_SAFE_TOOL_RESULT_CHARACTERS,
    )
    assert (
        accepted.mcp_maximum_tool_result_characters
        == MINIMUM_SAFE_TOOL_RESULT_CHARACTERS
    )

    with pytest.raises(
        ValidationError,
        match="mcp_maximum_tool_result_characters must be at least",
    ):
        Settings(
            _env_file=None,  # type: ignore[call-arg]
            document_storage_root=tmp_path / "documents-below",
            index_runtime_root=tmp_path / "indexes-below",
            mcp_maximum_tool_result_characters=(
                MINIMUM_SAFE_TOOL_RESULT_CHARACTERS - 1
            ),
        )


@pytest.mark.parametrize(
    "revision",
    [
        "",
        " ",
        "main",
        "latest",
        "refs/heads/main",
        "v1.0",
        "A" * 40,
        "a" * 7,
        "a" * 39,
        "a" * 41,
        "g" * 40,
    ],
)
def test_generation_revision_requires_lowercase_commit_sha(
    tmp_path: Path,
    revision: str,
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            document_storage_root=tmp_path,
            generation_model_revision=revision,
        )  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "revision",
    [
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "a" * 40,
    ],
)
def test_generation_revision_accepts_exact_lowercase_commit_sha(
    tmp_path: Path,
    revision: str,
) -> None:
    settings = Settings(
        _env_file=None,
        document_storage_root=tmp_path,
        generation_model_revision=revision,
    )  # type: ignore[call-arg]
    assert settings.generation_model_revision == revision


@pytest.mark.parametrize(
    ("name", "ceiling"),
    [
        ("maximum_generation_claim_characters", MAX_DRAFT_CLAIM_CHARACTERS),
        ("maximum_generation_claims", MAX_DRAFT_CLAIMS),
        (
            "maximum_generation_supports_per_claim",
            MAX_DRAFT_SUPPORTS_PER_CLAIM,
        ),
        ("maximum_generation_quote_characters", MAX_DRAFT_QUOTE_CHARACTERS),
        (
            "maximum_generation_conflict_evidence_ids",
            MAX_DRAFT_CONFLICT_EVIDENCE_IDS,
        ),
    ],
)
def test_structured_generation_limit_configuration_boundaries(
    tmp_path: Path,
    name: str,
    ceiling: int,
) -> None:
    exact = Settings(
        _env_file=None,
        document_storage_root=tmp_path,
        **{name: ceiling},  # type: ignore[arg-type]
    )  # type: ignore[call-arg]
    lower = Settings(
        _env_file=None,
        document_storage_root=tmp_path,
        **{name: max(1, ceiling - 1)},  # type: ignore[arg-type]
    )  # type: ignore[call-arg]
    assert getattr(exact, name) == ceiling
    assert getattr(lower, name) == max(1, ceiling - 1)

    with pytest.raises(ValidationError, match=name):
        Settings(
            _env_file=None,
            document_storage_root=tmp_path,
            **{name: ceiling + 1},  # type: ignore[arg-type]
        )  # type: ignore[call-arg]
