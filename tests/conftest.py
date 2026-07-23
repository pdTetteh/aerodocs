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
