"""Read-only UI route, contract, security, and laziness tests."""

from collections.abc import Callable, Sequence
from io import BytesIO
from pathlib import Path
from typing import cast

import faiss
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from starlette.routing import BaseRoute

from app.core.config import Settings
from app.factory import create_app
from app.ingestion.service import DocumentService
from app.ingestion.storage import ValidatedDocument
from app.retrieval.models import ModelIdentity
from app.retrieval.service import RetrievalService
from app.ui.config import ABSOLUTE_UI_RESPONSE_BYTES, build_public_ui_config
from app.ui.errors import UIConfigurationError


class UiFakeEmbeddingProvider:
    """Deterministic provider used without model or network access."""

    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(
            name="ui-test",
            revision="ui-test",
            dimension=4,
            maximum_sequence_length=512,
        )

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray(
            [[float("inspect" in text.lower()), 0.0, 0.0, 1.0] for text in texts],
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self.embed_documents([text])


def registered_route_paths(application: FastAPI) -> set[str]:
    """Return concrete paths from FastAPI's nested included routers."""
    paths: set[str] = set()
    pending = list(application.routes)
    while pending:
        route = pending.pop()
        path = getattr(route, "path", None)
        if isinstance(path, str):
            paths.add(path)
        children = getattr(route, "routes", None)
        if children is not None:
            pending.extend(cast(Sequence[BaseRoute], children))
        original_router = getattr(route, "original_router", None)
        if original_router is not None:
            pending.extend(cast(Sequence[BaseRoute], original_router.routes))
    return paths


def test_ui_page_catalogue_and_root_contract(client: TestClient) -> None:
    assert client.get("/").status_code == 404
    for path, heading in (
        ("/ui/", "System status"),
        ("/ui/documents", "Documents"),
        ("/ui/search", "Evidence search"),
        ("/ui/ask", "Ask AeroDocs"),
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert heading in response.text
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        policy = response.headers["content-security-policy"]
        assert "default-src 'none'" in policy
        assert "connect-src 'self'" in policy
        assert "frame-ancestors 'none'" in policy

    not_found = client.get("/ui/not/a/page")
    assert not_found.status_code == 404
    assert "Page not found" in not_found.text


def test_ui_assets_are_local_and_read_only(client: TestClient) -> None:
    script = client.get("/ui/assets/app.js")
    style = client.get("/ui/assets/app.css")

    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")
    assert style.status_code == 200
    assert style.headers["content-type"].startswith("text/css")
    assert script.headers["x-content-type-options"] == "nosniff"
    assert "localStorage" not in script.text
    assert "sessionStorage" not in script.text
    assert "serviceWorker" not in script.text
    assert ".innerHTML" not in script.text
    assert '"/indexes/rebuild"' not in script.text
    assert 'method: "POST"' in script.text

    documents = client.get("/ui/documents").text
    assert "upload" not in documents.lower()
    assert "rebuild" not in documents.lower()
    assert "/indexes/rebuild" not in documents


def test_ui_can_be_disabled_without_changing_health(
    default_settings: Settings,
) -> None:
    expected = {"name": "AeroDocs", "version": "0.1.0", "status": "healthy"}
    application = create_app(default_settings.model_copy(update={"ui_enabled": False}))
    route_paths = registered_route_paths(application)
    with TestClient(application) as client:
        assert client.get("/ui/").status_code == 404
        assert client.get("/ui/documents").status_code == 404
        assert client.get("/ui/search").status_code == 404
        assert client.get("/ui/ask").status_code == 404
        assert client.get("/ui/assets/app.js").status_code == 404
        assert client.get("/ui/api/index-status").status_code == 404
        assert "/ui/api/index-status" not in route_paths
        assert "/ui/api/index-status" not in client.get("/openapi.json").json()["paths"]
        active = client.get("/indexes/active")
        assert active.status_code == 404
        assert active.json()["error"]["code"] == "index_not_ready"
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == expected
        assert "content-security-policy" not in health.headers


def test_document_list_http_contract_is_bounded_and_deterministic(
    client: TestClient,
    make_pdf: Callable[..., bytes],
) -> None:
    for filename in ("second.pdf", "first.pdf"):
        response = client.post(
            "/documents",
            files={
                "file": (
                    filename,
                    make_pdf(
                        [filename],
                        metadata={"title": filename.removesuffix(".pdf")},
                    ),
                    "application/pdf",
                )
            },
        )
        assert response.status_code == 201

    listed = client.get("/documents?offset=0&limit=1")
    assert listed.status_code == 200
    assert set(listed.json()) == {"offset", "limit", "total", "documents"}
    assert listed.json()["offset"] == 0
    assert listed.json()["limit"] == 1
    assert listed.json()["total"] == 2
    assert len(listed.json()["documents"]) == 1
    assert set(listed.json()["documents"][0]) == {
        "document_id",
        "filename",
        "title",
        "page_count",
        "chunk_count",
        "processing_profile",
        "ingested_at",
    }
    assert client.get("/documents?limit=101").status_code == 422


def test_ui_pages_and_document_listing_do_not_load_models(
    default_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("model loading is forbidden for metadata pages")

    monkeypatch.setattr(
        "app.retrieval.embeddings.SentenceTransformerEmbeddingProvider._load",
        unexpected,
    )
    monkeypatch.setattr(
        "app.generation.provider.TransformersLlmProvider._load",
        unexpected,
    )
    with TestClient(create_app(default_settings)) as client:
        assert client.get("/ui/").status_code == 200
        assert client.get("/ui/documents").status_code == 200
        assert client.get("/documents").status_code == 200
        assert client.get("/health").status_code == 200


def test_active_index_metadata_does_not_deserialize_faiss(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = default_settings.model_copy(
        update={
            "document_storage_root": tmp_path / "documents",
            "index_runtime_root": tmp_path / "indexes",
            "generation_audit_root": tmp_path / "audits",
        }
    )
    documents = DocumentService(settings)
    documents.ingest(
        BytesIO(make_pdf(["inspection evidence"])),
        "manual.pdf",
        "application/pdf",
    )
    provider = UiFakeEmbeddingProvider()
    expected = RetrievalService(settings, provider).rebuild().index

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("FAISS deserialization is forbidden for metadata")

    monkeypatch.setattr(faiss, "read_index", unexpected)
    with TestClient(create_app(settings, embedding_provider=provider)) as client:
        response = client.get("/indexes/active")
        compact = client.get("/ui/api/index-status")

    assert response.status_code == 200
    assert response.json() == expected.model_dump(mode="json")
    assert compact.status_code == 200


def test_ui_headers_do_not_change_json_api_headers(client: TestClient) -> None:
    health = client.get("/health")
    documents = client.get("/documents")

    assert "content-security-policy" not in health.headers
    assert "content-security-policy" not in documents.headers


def test_public_ui_configuration_uses_authoritative_large_limits(
    default_settings: Settings,
) -> None:
    settings = default_settings.model_copy(
        update={
            "maximum_query_characters": 6_000,
            "chunk_max_characters": 20_000,
            "chunk_overlap_characters": 200,
        }
    )
    config = build_public_ui_config(settings)

    assert config.query_characters == 6_000
    assert config.chunk_characters == 20_000
    assert config.response_bytes.search > 1024 * 1024
    assert config.response_bytes.search <= ABSOLUTE_UI_RESPONSE_BYTES

    with TestClient(create_app(settings)) as client:
        html = client.get("/ui/search").text
    assert 'data-query-characters="6000"' in html
    assert 'data-chunk-characters="20000"' in html
    assert str(config.response_bytes.search) in html
    for forbidden in (
        str(settings.document_storage_root),
        str(settings.index_runtime_root),
        settings.embedding_model_revision,
        settings.generation_model_revision,
    ):
        assert forbidden not in html


def test_large_core_index_limit_does_not_prevent_application_startup(
    default_settings: Settings,
) -> None:
    settings = default_settings.model_copy(
        update={"maximum_chunks_per_index_build": 100_000}
    )
    config = build_public_ui_config(settings)

    assert config.response_bytes.index == 64 * 1024
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ui/").status_code == 200

    workflow_limited = settings.model_copy(
        update={"maximum_query_characters": 100_000_000}
    )
    limited_config = build_public_ui_config(workflow_limited)
    assert limited_config.search_available is False
    assert limited_config.response_bytes.search == ABSOLUTE_UI_RESPONSE_BYTES
    with TestClient(create_app(workflow_limited)) as client:
        assert client.get("/health").status_code == 200
        assert 'data-search-available="false"' in client.get("/ui/search").text


def test_typed_ui_configuration_failure_isolated_from_core_application(
    default_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_settings: Settings) -> object:
        raise UIConfigurationError

    monkeypatch.setattr("app.factory.create_ui_router", unavailable)
    application = create_app(default_settings)
    route_paths = registered_route_paths(application)
    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/ui/")
        compact = client.get("/ui/api/index-status")

    assert response.status_code == 503
    assert compact.status_code == 503
    assert "/ui/api/index-status" not in route_paths
    assert "Core AeroDocs services remain available" in response.text
    assert "configuration" not in response.text.lower()


def test_document_listing_validates_only_requested_page(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = default_settings.model_copy(
        update={"document_storage_root": tmp_path / "documents"}
    )
    service = DocumentService(settings)
    for number in range(3):
        service.ingest(
            BytesIO(make_pdf([f"document {number}"])),
            f"manual-{number}.pdf",
            "application/pdf",
        )
    document_ids = service.store.list_document_ids()
    original = service.store.validate_document
    validated_ids: list[str] = []

    def tracked(
        document_id: str,
        *,
        page_offset: int = 0,
        page_limit: int = 0,
        chunk_offset: int = 0,
        chunk_limit: int = 0,
    ) -> ValidatedDocument:
        validated_ids.append(document_id)
        return original(
            document_id,
            page_offset=page_offset,
            page_limit=page_limit,
            chunk_offset=chunk_offset,
            chunk_limit=chunk_limit,
        )

    monkeypatch.setattr(service.store, "validate_document", tracked)
    result = service.list_documents(offset=1, limit=1)

    assert [item.document_id for item in result.documents] == [document_ids[1]]
    assert validated_ids == [document_ids[1]]


def test_compact_index_status_is_bounded_and_has_no_ui_html_headers(
    default_settings: Settings,
    tmp_path: Path,
    make_pdf: Callable[..., bytes],
) -> None:
    settings = default_settings.model_copy(
        update={
            "document_storage_root": tmp_path / "documents",
            "index_runtime_root": tmp_path / "indexes",
            "generation_audit_root": tmp_path / "audits",
        }
    )
    documents = DocumentService(settings)
    documents.ingest(
        BytesIO(make_pdf(["inspection evidence"])),
        "manual.pdf",
        "application/pdf",
    )
    provider = UiFakeEmbeddingProvider()
    RetrievalService(settings, provider).rebuild()

    application = create_app(settings, embedding_provider=provider)
    route_paths = registered_route_paths(application)
    with TestClient(application) as client:
        compact = client.get("/ui/api/index-status")
        full = client.get("/indexes/active")
        openapi_paths = client.get("/openapi.json").json()["paths"]

    assert compact.status_code == 200
    assert "/ui/api/index-status" in route_paths
    assert "/ui/api/index-status" not in openapi_paths
    assert "content-security-policy" not in compact.headers
    assert set(compact.json()) == {
        "status",
        "index_profile_id",
        "corpus_fingerprint",
        "index_format_version",
        "embedding_model_name",
        "embedding_model_revision",
        "vector_count",
        "indexed_chunk_count",
        "indexed_character_count",
        "document_count",
        "zero_chunk_document_count",
        "similarity_metric",
        "index_type",
        "built_at",
    }
    assert "sources" not in compact.json()
    assert "zero_chunk_document_ids" not in compact.json()
    assert full.status_code == 200
    assert "sources" in full.json()
