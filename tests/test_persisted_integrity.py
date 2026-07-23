"""Semantic integrity and bounded pagination tests for persisted JSONL."""

import hashlib
import json
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.factory import create_app
from app.ingestion.errors import StorageError
from app.ingestion.service import DocumentService


def create_document(
    tmp_path: Path,
    default_settings: Settings,
    content: bytes,
    *,
    chunk_max_characters: int = 2_000,
) -> tuple[DocumentService, Path, str]:
    """Ingest a document and return its service and final directory."""
    root = tmp_path / "runtime"
    settings = default_settings.model_copy(
        update={
            "document_storage_root": root,
            "chunk_max_characters": chunk_max_characters,
            "chunk_overlap_characters": min(20, chunk_max_characters - 1),
        }
    )
    service = DocumentService(settings)
    result = service.ingest(
        BytesIO(content),
        "manual.pdf",
        "application/pdf",
    )
    directory = root / "documents" / result.document.document_id
    return service, directory, result.document.document_id


def rewrite_jsonl(
    directory: Path,
    filename: str,
    records: list[dict[str, object]],
) -> None:
    """Rewrite JSONL and update its manifest hash so semantic checks execute."""
    path = directory / filename
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    refresh_descriptor(directory, filename)


def refresh_descriptor(directory: Path, filename: str) -> None:
    """Refresh a deliberately modified artifact descriptor."""
    path = directory / filename
    content = path.read_bytes()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["artifacts"][filename]["sha256"] = hashlib.sha256(content).hexdigest()
    manifest["artifacts"][filename]["byte_size"] = len(content)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "wrong_document",
        "character_count",
        "duplicate_number",
        "out_of_order",
        "missing_number",
    ],
)
def test_corrupt_page_sequences_are_rejected(
    tmp_path: Path,
    default_settings: Settings,
    make_pdf: Callable[..., bytes],
    corruption: str,
) -> None:
    service, directory, document_id = create_document(
        tmp_path,
        default_settings,
        make_pdf(["Page one", "Page two"]),
    )
    records = [
        json.loads(line)
        for line in (directory / "pages.jsonl").read_text("utf-8").splitlines()
    ]
    if corruption == "wrong_document":
        records[0]["document_id"] = "doc_" + "f" * 64
    elif corruption == "character_count":
        records[0]["character_count"] += 1
    elif corruption == "duplicate_number":
        records[1]["page_number"] = 1
    elif corruption == "out_of_order":
        records.reverse()
    else:
        records[1]["page_number"] = 3
    rewrite_jsonl(directory, "pages.jsonl", records)

    with pytest.raises(StorageError):
        service.pages(document_id, offset=0, limit=1)


@pytest.mark.parametrize(
    "corruption",
    [
        "wrong_document",
        "character_count",
        "order",
        "page_range",
    ],
)
def test_corrupt_chunk_records_are_rejected(
    tmp_path: Path,
    default_settings: Settings,
    make_pdf: Callable[..., bytes],
    corruption: str,
) -> None:
    service, directory, document_id = create_document(
        tmp_path,
        default_settings,
        make_pdf(["alpha beta gamma delta epsilon " * 20]),
        chunk_max_characters=80,
    )
    records = [
        json.loads(line)
        for line in (directory / "chunks.jsonl").read_text("utf-8").splitlines()
    ]
    if corruption == "wrong_document":
        records[0]["document_id"] = "doc_" + "f" * 64
    elif corruption == "character_count":
        records[0]["character_count"] += 1
    elif corruption == "order":
        records[0]["chunk_order"] = 2
    else:
        records[0]["page_end"] = 999
    rewrite_jsonl(directory, "chunks.jsonl", records)

    with pytest.raises(StorageError):
        service.chunks(document_id, offset=0, limit=1)


def test_malformed_jsonl_is_rejected(
    tmp_path: Path,
    default_settings: Settings,
    make_pdf: Callable[..., bytes],
) -> None:
    service, directory, document_id = create_document(
        tmp_path,
        default_settings,
        make_pdf(["Page one"]),
    )
    (directory / "pages.jsonl").write_text("{not-json}\n", encoding="utf-8")
    refresh_descriptor(directory, "pages.jsonl")

    with pytest.raises(StorageError):
        service.pages(document_id, offset=0, limit=1)


def test_metadata_count_disagreement_is_rejected(
    tmp_path: Path,
    default_settings: Settings,
    make_pdf: Callable[..., bytes],
) -> None:
    service, directory, document_id = create_document(
        tmp_path,
        default_settings,
        make_pdf(["Page one"]),
    )
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text("utf-8"))
    metadata["page_count"] = 2
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    refresh_descriptor(directory, "metadata.json")

    with pytest.raises(StorageError):
        service.metadata(document_id)


def test_large_page_collection_returns_only_requested_slice(
    tmp_path: Path,
    default_settings: Settings,
    make_pdf: Callable[..., bytes],
) -> None:
    settings = default_settings.model_copy(
        update={"document_storage_root": tmp_path / "runtime"}
    )
    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/documents",
            files={
                "file": (
                    "many-pages.pdf",
                    make_pdf([f"Page {number}" for number in range(60)]),
                    "application/pdf",
                )
            },
        )
        document_id = created.json()["document"]["document_id"]
        response = client.get(f"/documents/{document_id}/pages?offset=40&limit=2")

    assert response.status_code == 200
    assert response.json()["total"] == 60
    assert len(response.json()["pages"]) == 2
    assert [page["page_number"] for page in response.json()["pages"]] == [41, 42]
