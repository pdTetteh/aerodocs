"""Production-path reproducibility verification for synthetic sources."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, ConfigDict

from app.core.config import Settings
from app.evaluation.benchmark import VerifiedSource
from app.evaluation.providers import OfflineEmbeddingProvider, OfflineLlmProvider
from app.evaluation.security import AuxiliaryFileError
from app.services.runtime import create_services


class ProductionIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pdf_sha256: str
    document_id: str
    page_count: int
    processing_profile_id: str
    chunk_ids: list[str]
    chunk_orders: list[int]
    stable_manifest_fingerprint: str
    vector_profile_id: str
    index_profile_id: str
    stable_corpus_fingerprint: str
    actual_corpus_fingerprint: str


def verify_synthetic_reproducibility(
    source: VerifiedSource,
    base_settings: Settings,
) -> ProductionIdentity:
    """Ingest and index the same generated source in two isolated service graphs."""
    if source.content is None:
        raise AuxiliaryFileError("synthetic_content_unavailable")
    first = _production_identity(source, base_settings)
    second = _production_identity(source, base_settings)
    stable_fields = (
        "pdf_sha256",
        "document_id",
        "page_count",
        "processing_profile_id",
        "chunk_ids",
        "chunk_orders",
        "stable_manifest_fingerprint",
        "vector_profile_id",
        "index_profile_id",
        "stable_corpus_fingerprint",
        "actual_corpus_fingerprint",
    )
    if any(getattr(first, field) != getattr(second, field) for field in stable_fields):
        raise AuxiliaryFileError("synthetic_reproducibility_mismatch")
    return first


def _production_identity(
    source: VerifiedSource,
    base_settings: Settings,
) -> ProductionIdentity:
    with TemporaryDirectory(prefix="aerodocs-evaluation-") as temporary:
        if source.content is None:
            raise AuxiliaryFileError("synthetic_content_unavailable")
        content = source.content
        root = Path(temporary)
        settings = base_settings.model_copy(
            update={
                "document_storage_root": root / "documents",
                "index_runtime_root": root / "indexes",
                "generation_audit_root": root / "audits",
                "embedding_model_name": "aerodocs-evaluation-fake",
                "embedding_model_revision": "offline-v1",
            }
        )
        services = create_services(
            settings,
            embedding_provider=OfflineEmbeddingProvider(),
            llm_provider=OfflineLlmProvider(),
        )
        result = services.documents.ingest(
            BytesIO(content),
            f"{source.source_id}.pdf",
            settings.allowed_pdf_media_type,
            ingested_at=datetime(2000, 1, 1, tzinfo=UTC),
        )
        validated = services.documents.store.validate_document(
            result.document.document_id,
            page_limit=settings.max_pdf_pages,
            chunk_limit=settings.maximum_chunks_per_index_build,
        )
        built = services.retrieval.rebuild().index
        stable_manifest = hashlib.sha256(
            json.dumps(
                {
                    "document_id": validated.manifest.document_id,
                    "source_sha256": validated.manifest.source_sha256,
                    "processing_profile_id": (validated.manifest.processing_profile_id),
                    "format_versions": [
                        validated.manifest.processing_format_version,
                        validated.manifest.extraction_format_version,
                        validated.manifest.chunking_format_version,
                    ],
                    "pages": validated.manifest.page_count,
                    "chunks": [
                        chunk.model_dump(mode="json") for chunk in validated.chunks
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        stable_corpus = hashlib.sha256(
            json.dumps(
                {
                    "document_id": validated.manifest.document_id,
                    "manifest": stable_manifest,
                    "chunk_ids": [chunk.chunk_id for chunk in validated.chunks],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return ProductionIdentity(
            pdf_sha256=hashlib.sha256(content).hexdigest(),
            document_id=result.document.document_id,
            page_count=validated.manifest.page_count,
            processing_profile_id=validated.manifest.processing_profile_id,
            chunk_ids=[chunk.chunk_id for chunk in validated.chunks],
            chunk_orders=[chunk.chunk_order for chunk in validated.chunks],
            stable_manifest_fingerprint=stable_manifest,
            vector_profile_id=built.vector_profile_id,
            index_profile_id=built.index_profile_id,
            stable_corpus_fingerprint=stable_corpus,
            actual_corpus_fingerprint=built.corpus_fingerprint,
        )
