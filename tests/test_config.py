"""Tests for application settings."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_default_settings(default_settings: Settings) -> None:
    assert default_settings.app_name == "AeroDocs"
    assert default_settings.app_version == "0.1.0"
    assert default_settings.log_level == "INFO"
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
