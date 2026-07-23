"""Integration tests for document upload and read endpoints."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from app.core.config import Settings
from app.factory import create_app


def upload_pdf(
    client: TestClient,
    content: bytes,
    *,
    filename: str = "manual.pdf",
    media_type: str = "application/pdf",
) -> Response:
    """Upload one in-memory test document."""
    return cast(
        Response,
        client.post(
            "/documents",
            files={"file": (filename, content, media_type)},
        ),
    )


def test_valid_pdf_ingestion_and_read_endpoints(
    client: TestClient,
    make_pdf: Callable[..., bytes],
    tmp_path: Path,
) -> None:
    response = upload_pdf(
        client,
        make_pdf(
            ["1 Scope\nPublic aviation test content.", "Second page text."],
            metadata={"title": "Synthetic Public Manual"},
        ),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["duplicate"] is False
    metadata = body["document"]
    assert set(metadata) == {
        "document_id",
        "filename",
        "sha256",
        "title",
        "author",
        "subject",
        "page_count",
        "extracted_character_count",
        "ingested_at",
        "extraction_format_version",
        "chunking_format_version",
        "processing_format_version",
        "processing_profile_id",
        "chunk_max_characters",
        "chunk_overlap_characters",
        "encrypted",
        "file_size_bytes",
    }
    assert metadata["title"] == "Synthetic Public Manual"
    assert metadata["page_count"] == 2
    assert metadata["encrypted"] is False
    assert metadata["processing_profile_id"].startswith("prf_")
    assert str(tmp_path) not in response.text

    document_id = metadata["document_id"]
    metadata_response = client.get(f"/documents/{document_id}")
    pages_response = client.get(f"/documents/{document_id}/pages?offset=0&limit=1")
    chunks_response = client.get(f"/documents/{document_id}/chunks")

    assert metadata_response.status_code == 200
    assert metadata_response.json() == metadata
    assert pages_response.status_code == 200
    assert pages_response.json()["total"] == 2
    assert [page["page_number"] for page in pages_response.json()["pages"]] == [1]
    assert chunks_response.status_code == 200
    assert chunks_response.json()["chunks"]
    assert str(tmp_path) not in (
        metadata_response.text + pages_response.text + chunks_response.text
    )


def test_duplicate_upload_returns_existing_metadata(
    client: TestClient,
    make_pdf: Callable[..., bytes],
    default_settings: Settings,
) -> None:
    content = make_pdf(["Duplicate content"])

    first = upload_pdf(client, content, filename="first.pdf")
    second = upload_pdf(client, content, filename="second.pdf")

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["document"] == first.json()["document"]
    assert second.json()["document"]["filename"] == "first.pdf"
    document_directories = list(
        (default_settings.document_storage_root / "documents").iterdir()
    )
    assert len(document_directories) == 1


def test_duplicate_with_different_profile_returns_typed_conflict(
    client: TestClient,
    make_pdf: Callable[..., bytes],
    default_settings: Settings,
) -> None:
    content = make_pdf(["Profile conflict content", "Second page"])
    first = upload_pdf(client, content)
    changed_settings = default_settings.model_copy(
        update={"chunk_max_characters": 1_000, "max_pdf_pages": 1}
    )

    with TestClient(create_app(changed_settings)) as changed_client:
        conflict = upload_pdf(changed_client, content)

    assert first.status_code == 201
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "processing_configuration_conflict"
    stored = conflict.json()["details"]["stored_processing_configuration"]
    assert (
        stored["processing_profile_id"]
        == first.json()["document"]["processing_profile_id"]
    )
    assert stored["chunk_max_characters"] == 2_000


@pytest.mark.parametrize(
    "limit_name",
    [
        "max_pdf_pages",
        "max_extracted_characters_per_page",
        "max_extracted_characters_total",
    ],
)
def test_duplicate_enforces_current_expansion_limits(
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    default_settings: Settings,
    limit_name: str,
) -> None:
    root = tmp_path / limit_name
    permissive = default_settings.model_copy(update={"document_storage_root": root})
    content = make_pdf(["First page content", "Second page has more content"])

    with TestClient(create_app(permissive)) as initial_client:
        created = upload_pdf(initial_client, content)
        document_id = created.json()["document"]["document_id"]
        pages = initial_client.get(f"/documents/{document_id}/pages").json()["pages"]

    actual_limits = {
        "max_pdf_pages": len(pages),
        "max_extracted_characters_per_page": max(
            page["character_count"] for page in pages
        ),
        "max_extracted_characters_total": sum(
            page["character_count"] for page in pages
        ),
    }
    exact = permissive.model_copy(update={limit_name: actual_limits[limit_name]})
    with TestClient(create_app(exact)) as exact_client:
        duplicate = upload_pdf(exact_client, content)

    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True

    document_directory = root / "documents" / document_id
    stored_before = {
        path.name: path.read_bytes() for path in document_directory.iterdir()
    }
    lower = permissive.model_copy(update={limit_name: actual_limits[limit_name] - 1})
    with TestClient(create_app(lower)) as lower_client:
        rejected = upload_pdf(lower_client, content)

    assert rejected.status_code == 413
    assert rejected.json()["error"]["code"] == "pdf_expansion_limit"
    assert [path.name for path in (root / "documents").iterdir()] == [document_id]
    assert list((root / ".staging").iterdir()) == []
    assert {
        path.name: path.read_bytes() for path in document_directory.iterdir()
    } == stored_before


def test_corrupt_duplicate_precedes_current_expansion_limit(
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    default_settings: Settings,
) -> None:
    root = tmp_path / "integrity-precedence"
    permissive = default_settings.model_copy(update={"document_storage_root": root})
    content = make_pdf(["First page", "Second page"])
    with TestClient(create_app(permissive)) as initial_client:
        created = upload_pdf(initial_client, content)

    document_id = created.json()["document"]["document_id"]
    pages_path = root / "documents" / document_id / "pages.jsonl"
    pages_path.write_bytes(pages_path.read_bytes() + b"corrupt")
    lower = permissive.model_copy(update={"max_pdf_pages": 1})
    with TestClient(create_app(lower)) as lower_client:
        rejected = upload_pdf(lower_client, content)

    assert rejected.status_code == 500
    assert rejected.json()["error"]["code"] == "document_storage_error"
    assert list((root / ".staging").iterdir()) == []


def test_filename_sanitization_and_title_fallback(
    client: TestClient,
    make_pdf: Callable[..., bytes],
) -> None:
    response = upload_pdf(
        client,
        make_pdf(["Text"]),
        filename="FAA?:\u202e Manual.pdf",
    )

    assert response.status_code == 201
    assert response.json()["document"]["filename"] == "FAA__ Manual.pdf"
    assert response.json()["document"]["title"] == "FAA__ Manual"


def test_path_traversal_filename_is_rejected(
    client: TestClient,
    make_pdf: Callable[..., bytes],
) -> None:
    response = upload_pdf(client, make_pdf(["Text"]), filename="../manual.pdf")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_filename"


def test_unicode_normalized_path_separator_is_rejected(
    client: TestClient,
    make_pdf: Callable[..., bytes],
) -> None:
    response = upload_pdf(client, make_pdf(["Text"]), filename="..\uff0fmanual.pdf")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_filename"


def test_empty_filename_is_rejected(
    client: TestClient,
    make_pdf: Callable[..., bytes],
) -> None:
    response = upload_pdf(client, make_pdf(["Text"]), filename=" ")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_filename"


def test_incorrect_media_type_is_rejected(
    client: TestClient,
    make_pdf: Callable[..., bytes],
) -> None:
    response = upload_pdf(
        client,
        make_pdf(["Text"]),
        media_type="application/octet-stream",
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_empty_invalid_signature_and_malformed_pdf_are_rejected(
    client: TestClient,
) -> None:
    empty = upload_pdf(client, b"")
    wrong_signature = upload_pdf(client, b"not a PDF")
    malformed = upload_pdf(client, b"%PDF-not-structurally-valid")

    assert empty.status_code == 400
    assert empty.json()["error"]["code"] == "empty_pdf"
    assert wrong_signature.status_code == 400
    assert wrong_signature.json()["error"]["code"] == "invalid_pdf_signature"
    assert malformed.status_code == 400
    assert malformed.json()["error"]["code"] == "invalid_pdf"


def test_oversize_upload_is_rejected(
    default_settings: Settings,
    make_pdf: Callable[..., bytes],
) -> None:
    settings = default_settings.model_copy(update={"max_pdf_upload_bytes": 10})
    with TestClient(create_app(settings)) as client:
        response = upload_pdf(client, make_pdf(["Text"]))

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_encrypted_pdf_is_rejected(
    client: TestClient,
    make_pdf: Callable[..., bytes],
) -> None:
    response = upload_pdf(client, make_pdf(["Secret"], encrypted=True))

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "encrypted_pdf"


def test_unknown_document_returns_typed_404(client: TestClient) -> None:
    unknown_id = "doc_" + "f" * 64

    for suffix in ("", "/pages", "/chunks"):
        response = client.get(f"/documents/{unknown_id}{suffix}")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "document_not_found"


def test_invalid_document_identifier_does_not_leak_paths(client: TestClient) -> None:
    response = client.get("/documents/not-a-document-id")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "document_not_found",
            "message": "The requested document was not found.",
        }
    }
    assert "/" not in response.json()["error"]["message"]


def test_pagination_boundaries(
    client: TestClient,
    make_pdf: Callable[..., bytes],
) -> None:
    created = upload_pdf(client, make_pdf(["Page one", "Page two"]))
    document_id = created.json()["document"]["document_id"]

    assert client.get(f"/documents/{document_id}/pages?offset=-1").status_code == 422
    assert client.get(f"/documents/{document_id}/pages?limit=0").status_code == 422
    assert client.get(f"/documents/{document_id}/pages?limit=501").status_code == 422

    maximum = client.get(f"/documents/{document_id}/pages?limit=500")
    beyond = client.get(f"/documents/{document_id}/pages?offset=99&limit=1")
    assert maximum.status_code == 200
    assert maximum.json()["total"] == 2
    assert beyond.status_code == 200
    assert beyond.json()["total"] == 2
    assert beyond.json()["pages"] == []
