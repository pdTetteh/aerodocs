"""Tests for containment and atomic ingestion cleanup."""

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Barrier

import pytest

from app.core.config import Settings
from app.ingestion.errors import (
    DocumentNotFoundError,
    InternalDocumentError,
    PdfExpansionLimitError,
    StorageError,
    UploadTooLargeError,
)
from app.ingestion.models import DocumentIngestResponse
from app.ingestion.service import DocumentService
from app.ingestion.storage import REQUIRED_ARTIFACTS, DocumentStore


def configured_service(
    default_settings: Settings,
    root: Path,
) -> DocumentService:
    """Create an isolated service with a concrete runtime root."""
    return DocumentService(
        default_settings.model_copy(update={"document_storage_root": root})
    )


def ingest(
    service: DocumentService,
    content: bytes,
) -> DocumentIngestResponse:
    """Ingest one in-memory PDF."""
    return service.ingest(BytesIO(content), "manual.pdf", "application/pdf")


def test_storage_rejects_identifier_path_traversal(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "runtime")

    with pytest.raises(DocumentNotFoundError):
        store.validate_document("../outside")


def test_storage_rejects_symlink_escape(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "runtime")
    outside = tmp_path / "outside"
    outside.mkdir()
    document_id = "doc_" + "a" * 64
    (store.documents_root / document_id).symlink_to(outside, target_is_directory=True)

    with pytest.raises(StorageError):
        store.validate_document(document_id)


@pytest.mark.parametrize("managed_name", ["documents", ".staging"])
def test_managed_directory_symlink_is_rejected(
    tmp_path: Path,
    managed_name: str,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (runtime / managed_name).symlink_to(outside, target_is_directory=True)

    with pytest.raises(StorageError):
        DocumentStore(runtime)

    assert list(outside.iterdir()) == []


def test_configured_root_symlink_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    configured_root = tmp_path / "runtime"
    configured_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(StorageError):
        DocumentStore(configured_root)

    assert list(outside.iterdir()) == []


def test_atomic_failure_cleans_staging_without_publishing(
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    monkeypatch: pytest.MonkeyPatch,
    default_settings: Settings,
) -> None:
    settings = default_settings.model_copy(
        update={"document_storage_root": tmp_path / "runtime"}
    )
    service = DocumentService(settings)

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(service.store, "write_artifacts", fail_write)

    with pytest.raises(InternalDocumentError):
        service.ingest(
            BytesIO(make_pdf(["Synthetic content"])),
            "manual.pdf",
            "application/pdf",
        )

    assert list((tmp_path / "runtime" / ".staging").iterdir()) == []
    assert list((tmp_path / "runtime" / "documents").iterdir()) == []


def test_pdf_byte_limit_is_exact_and_one_over_is_rejected(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "runtime")
    exact_staging = store.create_staging_directory()
    digest, size = store.write_upload(exact_staging, BytesIO(b"%PDF-"), 5)
    assert digest == hashlib.sha256(b"%PDF-").hexdigest()
    assert size == 5
    store.cleanup_staging(exact_staging)

    oversized_staging = store.create_staging_directory()
    with pytest.raises(UploadTooLargeError):
        store.write_upload(oversized_staging, BytesIO(b"%PDF-x"), 5)
    store.cleanup_staging(oversized_staging)

    assert list(store.staging_root.iterdir()) == []


def test_expansion_rejection_cleans_staging_before_chunking(
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    default_settings: Settings,
) -> None:
    root = tmp_path / "runtime"
    settings = default_settings.model_copy(
        update={
            "document_storage_root": root,
            "max_pdf_pages": 1,
        }
    )
    service = DocumentService(settings)

    with pytest.raises(PdfExpansionLimitError):
        ingest(service, make_pdf(["Page one", "Page two"]))

    assert list((root / ".staging").iterdir()) == []
    assert list((root / "documents").iterdir()) == []


@pytest.mark.parametrize(
    "checkpoint",
    [
        "source.pdf",
        "metadata.json",
        "pages.jsonl",
        "chunks.jsonl",
        "manifest.json",
    ],
)
def test_failure_after_each_artifact_cleans_staging(
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    default_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str,
) -> None:
    root = tmp_path / checkpoint.replace(".", "-")
    service = configured_service(default_settings, root)

    def fail_at_checkpoint(artifact_name: str) -> None:
        if artifact_name == checkpoint:
            raise RuntimeError("simulated checkpoint failure")

    monkeypatch.setattr(service.store, "checkpoint", fail_at_checkpoint)

    with pytest.raises(InternalDocumentError):
        ingest(service, make_pdf(["Synthetic content"]))

    assert list((root / ".staging").iterdir()) == []
    assert list((root / "documents").iterdir()) == []


def test_concurrent_identical_ingestion_publishes_once(
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    default_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runtime"
    first_service = configured_service(default_settings, root)
    second_service = configured_service(default_settings, root)
    barrier = Barrier(2)

    for service in (first_service, second_service):
        original_publish = service.store.publish

        def synchronized_publish(
            staging_directory: Path,
            document_id: str,
            *,
            publish: Callable[[Path, str], bool] = original_publish,
        ) -> bool:
            barrier.wait(timeout=5)
            return publish(staging_directory, document_id)

        monkeypatch.setattr(service.store, "publish", synchronized_publish)

    content = make_pdf(["Concurrent synthetic content"])
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(ingest, first_service, content),
            executor.submit(ingest, second_service, content),
        ]
        results = [future.result(timeout=10) for future in futures]

    assert sorted(result.duplicate for result in results) == [False, True]
    assert len(list((root / "documents").iterdir())) == 1
    assert list((root / ".staging").iterdir()) == []


def test_target_created_race_is_not_overwritten(
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    default_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runtime"
    service = configured_service(default_settings, root)
    original_publish = service.store.publish
    target: Path | None = None

    def create_incomplete_target(staging_directory: Path, document_id: str) -> bool:
        nonlocal target
        target = service.store.documents_root / document_id
        target.mkdir()
        return original_publish(staging_directory, document_id)

    monkeypatch.setattr(service.store, "publish", create_incomplete_target)

    with pytest.raises(StorageError):
        ingest(service, make_pdf(["Race content"]))

    assert target is not None
    assert list(target.iterdir()) == []
    assert list((root / ".staging").iterdir()) == []


@pytest.mark.parametrize("artifact_name", REQUIRED_ARTIFACTS)
@pytest.mark.parametrize("operation", ["delete", "corrupt"])
def test_duplicate_rejects_missing_or_corrupt_artifacts(
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    default_settings: Settings,
    artifact_name: str,
    operation: str,
) -> None:
    root = tmp_path / f"{operation}-{artifact_name}"
    service = configured_service(default_settings, root)
    content = make_pdf(["Integrity content"])
    first = ingest(service, content)
    artifact = root / "documents" / first.document.document_id / artifact_name
    if operation == "delete":
        artifact.unlink()
    else:
        artifact.write_bytes(artifact.read_bytes() + b"corrupt")

    with pytest.raises(StorageError):
        ingest(service, content)

    assert len(list((root / "documents").iterdir())) == 1
    assert list((root / ".staging").iterdir()) == []


def test_symlinked_artifact_is_rejected_without_touching_target(
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    default_settings: Settings,
) -> None:
    root = tmp_path / "runtime"
    service = configured_service(default_settings, root)
    content = make_pdf(["Integrity content"])
    first = ingest(service, content)
    source = root / "documents" / first.document.document_id / "source.pdf"
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"outside")
    source.unlink()
    source.symlink_to(outside)

    with pytest.raises(StorageError):
        ingest(service, content)

    assert outside.read_bytes() == b"outside"


def update_manifest_descriptor(document_directory: Path, filename: str) -> None:
    """Update one manifest descriptor after intentional test corruption."""
    manifest_path = document_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    content = (document_directory / filename).read_bytes()
    manifest["artifacts"][filename]["sha256"] = hashlib.sha256(content).hexdigest()
    manifest["artifacts"][filename]["byte_size"] = len(content)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "field",
    [
        "document_id",
        "source_sha256",
        "processing_profile_id",
        "page_count",
        "chunk_count",
    ],
)
def test_corrupt_manifest_identity_and_counts_are_rejected(
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    default_settings: Settings,
    field: str,
) -> None:
    root = tmp_path / field
    service = configured_service(default_settings, root)
    result = ingest(service, make_pdf(["Manifest content"]))
    manifest_path = root / "documents" / result.document.document_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest[field] = "corrupt" if isinstance(manifest[field], str) else 999
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StorageError):
        service.metadata(result.document.document_id)


def test_corrupt_manifest_artifact_hash_is_rejected(
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    default_settings: Settings,
) -> None:
    root = tmp_path / "manifest-hash"
    service = configured_service(default_settings, root)
    result = ingest(service, make_pdf(["Manifest content"]))
    manifest_path = root / "documents" / result.document.document_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["artifacts"]["pages.jsonl"]["sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StorageError):
        service.metadata(result.document.document_id)


@pytest.mark.parametrize("operation", ["delete", "malform"])
def test_missing_or_malformed_manifest_is_rejected(
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    default_settings: Settings,
    operation: str,
) -> None:
    root = tmp_path / f"manifest-{operation}"
    service = configured_service(default_settings, root)
    content = make_pdf(["Manifest content"])
    result = ingest(service, content)
    manifest_path = root / "documents" / result.document.document_id / "manifest.json"
    if operation == "delete":
        manifest_path.unlink()
    else:
        manifest_path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(StorageError):
        ingest(service, content)

    assert list((root / ".staging").iterdir()) == []
