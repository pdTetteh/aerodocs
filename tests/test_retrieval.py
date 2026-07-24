"""Offline integration tests for local indexing and evidence retrieval."""

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from app.core.config import Settings
from app.factory import create_app
from app.ingestion.service import DocumentService
from app.retrieval.errors import (
    EmbeddingDimensionError,
    EmbeddingModelUnavailableError,
    IndexIntegrityError,
    IndexResourceLimitError,
    NonFiniteEmbeddingError,
)
from app.retrieval.models import IndexManifest, ModelIdentity
from app.retrieval.service import RetrievalService


class FakeEmbeddingProvider:
    """Deterministic local provider that never accesses a model or network."""

    def __init__(self, revision: str = "test-1") -> None:
        self.identity_calls = 0
        self.document_calls = 0
        self.query_calls = 0
        self.revision = revision

    def model_identity(self) -> ModelIdentity:
        self.identity_calls += 1
        return ModelIdentity(
            name="fake-local",
            revision=self.revision,
            dimension=4,
            maximum_sequence_length=512,
        )

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.document_calls += 1
        return np.asarray([_vector(text) for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        self.query_calls += 1
        return np.asarray([_vector(text)], dtype=np.float32)


def _vector(text: str) -> list[float]:
    lowered = text.lower()
    return [
        float(lowered.count("corrosion")),
        float(lowered.count("inspect")),
        float(lowered.count("engine")),
        1.0,
    ]


def isolated_settings(default_settings: Settings, tmp_path: Path) -> Settings:
    return default_settings.model_copy(
        update={
            "document_storage_root": tmp_path / "documents",
            "index_runtime_root": tmp_path / "indexes",
            "embedding_model_name": "fake-local",
            "embedding_model_revision": "test-1",
            "embedding_batch_size": 2,
        }
    )


def upload(client: TestClient, content: bytes, filename: str) -> str:
    response = client.post(
        "/documents",
        files={"file": (filename, content, "application/pdf")},
    )
    assert response.status_code == 201
    return str(response.json()["document"]["document_id"])


def test_health_and_index_metadata_do_not_load_model(
    default_settings: Settings,
    tmp_path: Path,
) -> None:
    provider = FakeEmbeddingProvider()
    with TestClient(
        create_app(
            isolated_settings(default_settings, tmp_path), embedding_provider=provider
        )
    ) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/indexes/active").status_code == 404

    assert provider.identity_calls == 0
    assert provider.document_calls == 0


def test_build_unchanged_search_filter_and_provenance(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    provider = FakeEmbeddingProvider()
    settings = isolated_settings(default_settings, tmp_path)
    with TestClient(create_app(settings, embedding_provider=provider)) as client:
        corrosion_id = upload(
            client,
            make_pdf(["CORROSION INSPECTION\nInspect corrosion carefully."]),
            "corrosion.pdf",
        )
        engine_id = upload(
            client,
            make_pdf(["ENGINE CHECK\nInspect engine condition."]),
            "engine.pdf",
        )

        first = client.post("/indexes/rebuild")
        unchanged = client.post("/indexes/rebuild")
        metadata = client.get("/indexes/active")
        search = client.post(
            "/search",
            json={"query": "inspect corrosion", "top_k": 5},
        )
        filtered = client.post(
            "/search",
            json={
                "query": "inspect corrosion",
                "top_k": 5,
                "document_ids": [engine_id],
            },
        )

    assert first.status_code == 201
    assert first.json()["rebuilt"] is True
    assert unchanged.status_code == 200
    assert unchanged.json()["rebuilt"] is False
    assert metadata.status_code == 200
    assert metadata.json()["vector_count"] == 2
    assert search.status_code == 200
    assert search.json()["results"][0]["document_id"] == corrosion_id
    assert search.json()["results"][0]["filename"] == "corrosion.pdf"
    assert filtered.status_code == 200
    assert {item["document_id"] for item in filtered.json()["results"]} == {engine_id}
    assert provider.document_calls == 1


def test_empty_corpus_limits_and_index_not_ready(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    provider = FakeEmbeddingProvider()
    settings = isolated_settings(default_settings, tmp_path)
    with TestClient(create_app(settings, embedding_provider=provider)) as client:
        assert client.post("/indexes/rebuild").json()["error"]["code"] == "empty_corpus"
        not_ready = client.post("/search", json={"query": "inspection"})
        assert not_ready.status_code == 404

        upload(client, make_pdf(["one chunk"]), "one.pdf")

    limited = settings.model_copy(update={"maximum_chunks_per_index_build": 0})
    # model_copy is intentionally used for a service-policy boundary test.
    with TestClient(create_app(limited, embedding_provider=provider)) as client:
        response = client.post("/indexes/rebuild")
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "index_resource_limit"


@pytest.mark.parametrize(
    "setting_name",
    [
        "maximum_chunks_per_index_build",
        "maximum_total_embedding_characters_per_build",
    ],
)
def test_index_build_exact_limit_and_limit_plus_one(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    setting_name: str,
) -> None:
    settings = isolated_settings(default_settings, tmp_path)
    document = DocumentService(settings).ingest(
        BytesIO(make_pdf(["corrosion inspection"])),
        "manual.pdf",
        "application/pdf",
    )
    actual = (
        1
        if setting_name == "maximum_chunks_per_index_build"
        else document.document.extracted_character_count
    )
    exact = settings.model_copy(update={setting_name: actual})
    assert RetrievalService(exact, FakeEmbeddingProvider()).rebuild().rebuilt is True

    lower = settings.model_copy(update={setting_name: actual - 1})
    with pytest.raises(IndexResourceLimitError):
        RetrievalService(lower, FakeEmbeddingProvider()).rebuild()


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"query": " "}, "invalid_search_request"),
        ({"query": "x", "top_k": 0}, "invalid_search_request"),
        ({"query": "x", "top_k": 21}, "invalid_search_request"),
        ({"query": "x", "minimum_score": 2}, "invalid_search_request"),
        ({"query": "x", "document_ids": ["../bad"]}, "invalid_search_request"),
    ],
)
def test_search_validation(
    default_settings: Settings,
    tmp_path: Path,
    payload: dict[str, object],
    code: str,
) -> None:
    provider = FakeEmbeddingProvider()
    with TestClient(
        create_app(
            isolated_settings(default_settings, tmp_path), embedding_provider=provider
        )
    ) as client:
        response = client.post("/search", json=payload)
    # Request validation occurs before index lookup.
    assert response.json()["error"]["code"] == code


def test_unknown_document_filter_and_minimum_score(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    provider = FakeEmbeddingProvider()
    with TestClient(
        create_app(
            isolated_settings(default_settings, tmp_path), embedding_provider=provider
        )
    ) as client:
        upload(client, make_pdf(["corrosion"]), "manual.pdf")
        assert client.post("/indexes/rebuild").status_code == 201
        unknown = client.post(
            "/search",
            json={
                "query": "corrosion",
                "document_ids": ["doc_" + "f" * 64],
            },
        )
        empty = client.post(
            "/search",
            json={"query": "unrelated", "minimum_score": 1.0},
        )

    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "filtered_document_not_found"
    assert empty.status_code == 200
    assert empty.json()["results"] == []


def test_changed_corpus_rebuilds_and_corrupt_index_fails(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    provider = FakeEmbeddingProvider()
    settings = isolated_settings(default_settings, tmp_path)
    with TestClient(create_app(settings, embedding_provider=provider)) as client:
        upload(client, make_pdf(["first"]), "first.pdf")
        first = client.post("/indexes/rebuild")
        upload(client, make_pdf(["second"]), "second.pdf")
        stale = client.post("/search", json={"query": "first"})
        changed = client.post("/indexes/rebuild")
        version_id = changed.json()["index"]["index_profile_id"]

    assert first.status_code == 201
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "stale_index"
    assert changed.status_code == 201
    assert changed.json()["rebuilt"] is True
    assert version_id != first.json()["index"]["index_profile_id"]

    active_pointer = settings.index_runtime_root / "active.json"
    active_pointer.write_bytes(active_pointer.read_bytes() + b"corrupt")
    with TestClient(create_app(settings, embedding_provider=provider)) as client:
        corrupt = client.get("/indexes/active")
    assert corrupt.status_code == 500
    assert corrupt.json()["error"]["code"] == "index_integrity_error"


def test_dimension_and_non_finite_query_failures(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeEmbeddingProvider()
    settings = isolated_settings(default_settings, tmp_path)
    with TestClient(create_app(settings, embedding_provider=provider)) as client:
        upload(client, make_pdf(["corrosion"]), "manual.pdf")
        assert client.post("/indexes/rebuild").status_code == 201

        monkeypatch.setattr(
            provider,
            "embed_query",
            lambda _text: np.asarray([[1.0, 2.0]], dtype=np.float32),
        )
        wrong = client.post("/search", json={"query": "corrosion"})
        assert wrong.status_code == 500
        assert wrong.json()["error"]["code"] == EmbeddingDimensionError.code

        monkeypatch.setattr(
            provider,
            "embed_query",
            lambda _text: np.asarray([[np.nan, 0.0, 0.0, 1.0]], dtype=np.float32),
        )
        non_finite = client.post("/search", json={"query": "corrosion"})
        assert non_finite.status_code == 500
        assert non_finite.json()["error"]["code"] == NonFiniteEmbeddingError.code


def test_index_artifacts_are_deterministically_hashed(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    provider = FakeEmbeddingProvider()
    settings = isolated_settings(default_settings, tmp_path)
    with TestClient(create_app(settings, embedding_provider=provider)) as client:
        upload(client, make_pdf(["corrosion"]), "manual.pdf")
        built = client.post("/indexes/rebuild")

    pointer = (settings.index_runtime_root / "active.json").read_bytes()
    assert hashlib.sha256(pointer).hexdigest()
    assert built.json()["index"]["corpus_fingerprint"].startswith("cor_")


def test_existing_unlocked_lock_file_is_not_stale(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    provider = FakeEmbeddingProvider()
    settings = isolated_settings(default_settings, tmp_path)
    service = RetrievalService(settings, provider)
    DocumentService(settings).ingest(
        BytesIO(make_pdf(["corrosion"])),
        "manual.pdf",
        "application/pdf",
    )
    lock_path = settings.index_runtime_root / ".locks" / "rebuild.lock"
    lock_path.write_bytes(b"left by exited process")

    assert service.rebuild().rebuilt is True


def test_concurrent_identical_rebuild_publishes_once(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    settings = isolated_settings(default_settings, tmp_path)
    DocumentService(settings).ingest(
        BytesIO(make_pdf(["corrosion"])),
        "manual.pdf",
        "application/pdf",
    )
    first_service = RetrievalService(settings, FakeEmbeddingProvider())
    second_service = RetrievalService(settings, FakeEmbeddingProvider())

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [
            future.result(timeout=10)
            for future in (
                executor.submit(first_service.rebuild),
                executor.submit(second_service.rebuild),
            )
        ]

    assert sorted(result.rebuilt for result in results) == [False, True]
    assert len(list((settings.index_runtime_root / "versions").iterdir())) == 1
    assert list((settings.index_runtime_root / ".staging").iterdir()) == []


def test_previous_active_index_survives_failed_changed_build(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = isolated_settings(default_settings, tmp_path)
    documents = DocumentService(settings)
    documents.ingest(
        BytesIO(make_pdf(["first corrosion"])),
        "first.pdf",
        "application/pdf",
    )
    provider = FakeEmbeddingProvider()
    service = RetrievalService(settings, provider)
    first = service.rebuild()
    pointer_before = (settings.index_runtime_root / "active.json").read_bytes()
    documents.ingest(
        BytesIO(make_pdf(["second engine"])),
        "second.pdf",
        "application/pdf",
    )

    def fail_embedding(_texts: Sequence[str]) -> NDArray[np.float32]:
        raise EmbeddingModelUnavailableError

    monkeypatch.setattr(provider, "embed_documents", fail_embedding)
    with pytest.raises(EmbeddingModelUnavailableError):
        service.rebuild()

    assert (settings.index_runtime_root / "active.json").read_bytes() == pointer_before
    assert service.active_metadata().index_profile_id == first.index.index_profile_id
    assert list((settings.index_runtime_root / ".staging").iterdir()) == []


@pytest.mark.parametrize("artifact", ["index.faiss", "records.jsonl", "metadata.json"])
def test_corrupt_active_artifact_is_rejected(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    artifact: str,
) -> None:
    settings = isolated_settings(default_settings, tmp_path)
    documents = DocumentService(settings)
    documents.ingest(
        BytesIO(make_pdf(["corrosion"])),
        "manual.pdf",
        "application/pdf",
    )
    service = RetrievalService(settings, FakeEmbeddingProvider())
    service.rebuild()
    pointer = json.loads(
        (settings.index_runtime_root / "active.json").read_text("utf-8")
    )
    path = settings.index_runtime_root / "versions" / pointer["version_id"] / artifact
    path.write_bytes(path.read_bytes() + b"corrupt")

    with pytest.raises(IndexIntegrityError):
        service.active_metadata()


def test_symlinked_index_root_is_rejected(
    default_settings: Settings,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "indexes"
    root.symlink_to(outside, target_is_directory=True)
    settings = default_settings.model_copy(
        update={
            "document_storage_root": tmp_path / "documents",
            "index_runtime_root": root,
        }
    )

    with pytest.raises(IndexIntegrityError):
        RetrievalService(settings, FakeEmbeddingProvider())

    assert list(outside.iterdir()) == []


def _active_version_directory(settings: Settings) -> Path:
    pointer = json.loads(
        (settings.index_runtime_root / "active.json").read_text("utf-8")
    )
    version_id = str(pointer["version_id"])
    return settings.index_runtime_root / "versions" / version_id


def _directory_snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    }


@pytest.mark.parametrize(
    "component",
    [
        "active.json",
        "manifest.json",
        "metadata.json",
        "records.jsonl",
        "index.faiss",
    ],
)
def test_rebuild_recovers_from_each_corrupt_active_component(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    component: str,
) -> None:
    settings = isolated_settings(default_settings, tmp_path)
    provider = FakeEmbeddingProvider()
    with TestClient(create_app(settings, embedding_provider=provider)) as client:
        upload(client, make_pdf(["corrosion inspection"]), "manual.pdf")
        assert client.post("/indexes/rebuild").status_code == 201
        corrupt_directory = _active_version_directory(settings)
        corrupt_path = (
            settings.index_runtime_root / "active.json"
            if component == "active.json"
            else corrupt_directory / component
        )
        corrupt_path.write_bytes(corrupt_path.read_bytes() + b"corrupt")
        corrupt_snapshot = _directory_snapshot(corrupt_directory)

        metadata_before = client.get("/indexes/active")
        search_before = client.post("/search", json={"query": "corrosion"})
        recovered = client.post("/indexes/rebuild")
        search_after = client.post("/search", json={"query": "corrosion"})

    assert metadata_before.status_code == 500
    assert metadata_before.json()["error"]["code"] == "index_integrity_error"
    assert search_before.status_code == 500
    assert search_before.json()["error"]["code"] == "index_integrity_error"
    expected_status = 200 if component == "active.json" else 201
    assert recovered.status_code == expected_status
    assert recovered.json()["rebuilt"] is (component != "active.json")
    if component == "active.json":
        assert _active_version_directory(settings) == corrupt_directory
    else:
        assert _active_version_directory(settings) != corrupt_directory
    assert _directory_snapshot(corrupt_directory) == corrupt_snapshot
    assert search_after.status_code == 200
    assert search_after.json()["results"][0]["document_id"].startswith("doc_")


def test_existing_profile_is_reactivated_without_embedding_or_new_metadata(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    settings = isolated_settings(default_settings, tmp_path)
    DocumentService(settings).ingest(
        BytesIO(make_pdf(["corrosion inspection"])),
        "manual.pdf",
        "application/pdf",
    )
    first_a = RetrievalService(settings, FakeEmbeddingProvider("profile-a")).rebuild()
    a_directory = _active_version_directory(settings)
    a_snapshot = _directory_snapshot(a_directory)
    RetrievalService(settings, FakeEmbeddingProvider("profile-b")).rebuild()
    reuse_provider = FakeEmbeddingProvider("profile-a")

    with TestClient(create_app(settings, embedding_provider=reuse_provider)) as client:
        response = client.post("/indexes/rebuild")
        active = client.get("/indexes/active")

    assert response.status_code == 200
    assert response.json()["rebuilt"] is False
    assert response.json()["index"] == active.json()
    original_built_at = first_a.index.built_at.isoformat().replace("+00:00", "Z")
    assert response.json()["index"]["built_at"] == original_built_at
    assert reuse_provider.document_calls == 0
    assert _active_version_directory(settings) == a_directory
    assert _directory_snapshot(a_directory) == a_snapshot
    assert len(list((settings.index_runtime_root / "versions").iterdir())) == 2


def test_corrupt_matching_version_is_not_reused_or_overwritten(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    settings = isolated_settings(default_settings, tmp_path)
    DocumentService(settings).ingest(
        BytesIO(make_pdf(["corrosion inspection"])),
        "manual.pdf",
        "application/pdf",
    )
    RetrievalService(settings, FakeEmbeddingProvider("profile-a")).rebuild()
    corrupt_a = _active_version_directory(settings)
    RetrievalService(settings, FakeEmbeddingProvider("profile-b")).rebuild()
    records = corrupt_a / "records.jsonl"
    records.write_bytes(records.read_bytes() + b"corrupt")
    corrupt_snapshot = _directory_snapshot(corrupt_a)

    with TestClient(
        create_app(settings, embedding_provider=FakeEmbeddingProvider("profile-a"))
    ) as client:
        rebuilt = client.post("/indexes/rebuild")

    assert rebuilt.status_code == 201
    assert rebuilt.json()["rebuilt"] is True
    assert _active_version_directory(settings) != corrupt_a
    assert _directory_snapshot(corrupt_a) == corrupt_snapshot
    assert len(list((settings.index_runtime_root / "versions").iterdir())) == 3


@pytest.mark.parametrize(
    "failure_stage",
    [
        "records",
        "index",
        "metadata",
        "manifest",
        "staged_validation",
        "pointer_activation",
    ],
)
def test_failed_corrupt_index_recovery_preserves_every_existing_byte(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    settings = isolated_settings(default_settings, tmp_path)
    DocumentService(settings).ingest(
        BytesIO(make_pdf(["corrosion inspection"])),
        "manual.pdf",
        "application/pdf",
    )
    service = RetrievalService(settings, FakeEmbeddingProvider())
    service.rebuild()
    corrupt_directory = _active_version_directory(settings)
    records = corrupt_directory / "records.jsonl"
    records.write_bytes(records.read_bytes() + b"corrupt")
    pointer_before = (settings.index_runtime_root / "active.json").read_bytes()
    corrupt_snapshot = _directory_snapshot(corrupt_directory)

    method_by_stage = {
        "records": "finish_records",
        "index": "write_index",
        "metadata": "write_metadata",
        "manifest": "write_manifest",
    }
    if failure_stage in method_by_stage:
        method_name = method_by_stage[failure_stage]
        original = getattr(service.indexes, method_name)

        def fail_after_artifact(*args: object, **kwargs: object) -> object:
            original(*args, **kwargs)
            raise IndexIntegrityError

        monkeypatch.setattr(service.indexes, method_name, fail_after_artifact)
    elif failure_stage == "staged_validation":

        def fail_after_validation(staging: Path, manifest: IndexManifest) -> object:
            service.indexes.validate_directory(
                staging,
                expected_version=manifest.version_id,
            )
            raise IndexIntegrityError

        monkeypatch.setattr(service.indexes, "publish", fail_after_validation)
    else:

        def fail_pointer(_source: Path, _destination: Path) -> None:
            raise OSError("simulated pointer activation failure")

        monkeypatch.setattr(os, "replace", fail_pointer)

    with pytest.raises(IndexIntegrityError):
        service.rebuild()

    assert (settings.index_runtime_root / "active.json").read_bytes() == pointer_before
    assert _directory_snapshot(corrupt_directory) == corrupt_snapshot
    assert list((settings.index_runtime_root / ".staging").iterdir()) == []
    assert list((settings.index_runtime_root / "versions").iterdir()) == [
        corrupt_directory
    ]
