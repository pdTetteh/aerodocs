"""Synchronous local index building and evidence retrieval."""

import logging
import math
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from typing import Literal, TextIO

import faiss

from app.core.config import Settings
from app.ingestion.errors import StorageError
from app.ingestion.models import ChunkRecord
from app.ingestion.storage import DOCUMENT_ID_PATTERN, DocumentStore
from app.retrieval.embeddings import (
    EmbeddingProvider,
    normalize_matrix,
    validate_matrix,
)
from app.retrieval.errors import (
    CorruptSourceDocumentError,
    EmptyCorpusError,
    FilteredDocumentNotFoundError,
    IncompatibleIndexProfileError,
    IndexIntegrityError,
    IndexNotReadyError,
    IndexResourceLimitError,
    InvalidSearchRequestError,
    StaleIndexError,
)
from app.retrieval.models import (
    IndexBuildResponse,
    IndexMetadata,
    IndexRecord,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SourceSummary,
)
from app.retrieval.profiles import (
    corpus_fingerprint,
    index_profile_id,
    vector_profile_id,
)
from app.retrieval.storage import IndexStore

LOGGER = logging.getLogger("aerodocs.retrieval")


class RetrievalService:
    """Build and query one application-local global retrieval index."""

    def __init__(
        self,
        settings: Settings,
        provider: EmbeddingProvider,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.documents = DocumentStore(settings.document_storage_root)
        self.indexes = IndexStore(
            settings.index_runtime_root,
            settings.index_build_lock_timeout_seconds,
        )

    def rebuild(self) -> IndexBuildResponse:
        """Build or reuse the deterministic active corpus index."""
        with self.indexes.build_lock():
            sources = self._discover_sources()
            total_chunks = sum(source.chunk_count for source in sources)
            total_characters = sum(source.total_characters for source in sources)
            if total_chunks == 0:
                raise EmptyCorpusError
            self._validate_build_limits(total_chunks, total_characters)

            model = self.provider.model_identity()
            index_type: Literal["IndexFlatIP", "IndexFlatL2"] = (
                "IndexFlatIP"
                if self.settings.embedding_normalization
                else "IndexFlatL2"
            )
            metric: Literal["cosine", "inverse_squared_l2"] = (
                "cosine"
                if self.settings.embedding_normalization
                else "inverse_squared_l2"
            )
            vector_id = vector_profile_id(
                index_format_version=self.settings.index_format_version,
                model=model,
                normalization=self.settings.embedding_normalization,
                index_type=index_type,
                metric=metric,
            )
            corpus_id = corpus_fingerprint(sources)
            profile_id = index_profile_id(vector_id, corpus_id)

            try:
                active_manifest, _active_index = self.indexes.active()
            except (IndexNotReadyError, IndexIntegrityError):
                pass
            else:
                if active_manifest.index_profile_id == profile_id:
                    return IndexBuildResponse(
                        rebuilt=False,
                        index=IndexMetadata.model_validate(
                            active_manifest.model_dump()
                        ),
                    )

            compatible = self.indexes.find_compatible(profile_id, corpus_id)
            if compatible is not None:
                publication = self.indexes.reactivate(compatible)
                return IndexBuildResponse(
                    rebuilt=publication.created,
                    index=IndexMetadata.model_validate(
                        publication.manifest.model_dump()
                    ),
                )

            staging = self.indexes.create_staging()
            try:
                index: faiss.Index = (
                    faiss.IndexFlatIP(model.dimension)
                    if self.settings.embedding_normalization
                    else faiss.IndexFlatL2(model.dimension)
                )
                writer = self.indexes.records_writer(staging)
                row = 0
                try:
                    batch_texts: list[str] = []
                    batch_records: list[IndexRecord] = []
                    for source, chunk in self._iter_source_chunks(sources):
                        record = _record(row, source, chunk)
                        batch_texts.append(chunk.text)
                        batch_records.append(record)
                        row += 1
                        if len(batch_texts) == self.settings.embedding_batch_size:
                            self._add_batch(index, writer, batch_texts, batch_records)
                            batch_texts = []
                            batch_records = []
                    if batch_texts:
                        self._add_batch(index, writer, batch_texts, batch_records)
                finally:
                    self.indexes.finish_records(writer)

                built_at = datetime.now(UTC)
                metadata = IndexMetadata(
                    index_profile_id=profile_id,
                    vector_profile_id=vector_id,
                    corpus_fingerprint=corpus_id,
                    index_format_version=self.settings.index_format_version,
                    model=model,
                    index_type=index_type,
                    similarity_metric=metric,
                    embedding_normalization=self.settings.embedding_normalization,
                    vector_count=total_chunks,
                    indexed_chunk_count=total_chunks,
                    indexed_character_count=total_characters,
                    source_document_count=len(sources),
                    zero_chunk_document_ids=[
                        source.document_id
                        for source in sources
                        if source.chunk_count == 0
                    ],
                    sources=sources,
                    built_at=built_at,
                )
                self.indexes.write_index(staging, index)
                self.indexes.write_metadata(staging, metadata)
                version_id = self.indexes.create_version_id(profile_id)
                manifest = self.indexes.write_manifest(
                    staging,
                    metadata,
                    version_id,
                )
                publication = self.indexes.publish(staging, manifest)
                return IndexBuildResponse(
                    rebuilt=publication.created,
                    index=IndexMetadata.model_validate(
                        publication.manifest.model_dump()
                    ),
                )
            finally:
                self.indexes.cleanup_staging(staging)

    def active_metadata(self) -> IndexMetadata:
        """Return fully validated public active-index metadata."""
        manifest, _index = self.indexes.active()
        return IndexMetadata.model_validate(manifest.model_dump())

    def search(self, request: SearchRequest) -> SearchResponse:
        """Return deterministic ranked evidence passages without generation."""
        query, top_k, minimum_score, filters = self._validate_search(request)
        manifest, index = self.indexes.active()
        sources = self._discover_sources()
        if corpus_fingerprint(sources) != manifest.corpus_fingerprint:
            raise StaleIndexError
        model = self.provider.model_identity()
        expected_vector_id = vector_profile_id(
            index_format_version=self.settings.index_format_version,
            model=model,
            normalization=self.settings.embedding_normalization,
            index_type=manifest.index_type,
            metric=manifest.similarity_metric,
        )
        if expected_vector_id != manifest.vector_profile_id:
            raise IncompatibleIndexProfileError
        indexed_ids = {source.document_id for source in manifest.sources}
        if filters is not None and not filters <= indexed_ids:
            raise FilteredDocumentNotFoundError

        query_vector = self.provider.embed_query(query)
        query_vector = validate_matrix(
            query_vector,
            rows=1,
            dimension=manifest.model.dimension,
        )
        if manifest.embedding_normalization:
            query_vector = normalize_matrix(query_vector)
        distances, labels = index.search(query_vector, manifest.vector_count)
        scores_by_row = {
            int(row): _score(float(distance), manifest.similarity_metric)
            for row, distance in zip(labels[0], distances[0], strict=True)
            if int(row) >= 0
        }
        candidates: list[tuple[float, IndexRecord]] = []
        directory = self.indexes.active_directory(manifest)
        for record in self.indexes.iter_records(directory):
            if filters is not None and record.document_id not in filters:
                continue
            score = scores_by_row[record.row]
            if score >= minimum_score:
                candidates.append((score, record))
                candidates.sort(key=lambda item: (-item[0], item[1].chunk_id))
                del candidates[top_k:]
        results = [
            SearchResult(
                rank=rank,
                score=score,
                **record.model_dump(exclude={"row", "processing_profile_id"}),
            )
            for rank, (score, record) in enumerate(candidates, start=1)
        ]
        return SearchResponse(
            query=query,
            index_profile_id=manifest.index_profile_id,
            corpus_fingerprint=manifest.corpus_fingerprint,
            similarity_metric=manifest.similarity_metric,
            requested_top_k=top_k,
            returned_count=len(results),
            results=results,
        )

    def _discover_sources(self) -> list[SourceSummary]:
        try:
            document_ids = self.documents.list_document_ids()
            sources: list[SourceSummary] = []
            for document_id in document_ids:
                validated = self.documents.validate_document(document_id)
                total_characters = sum(
                    chunk.character_count
                    for chunk in self.documents.iter_document_chunks(document_id)
                )
                sources.append(
                    SourceSummary(
                        document_id=document_id,
                        processing_profile_id=(
                            validated.metadata.processing_profile_id
                        ),
                        manifest_sha256=self.documents.manifest_sha256(document_id),
                        chunk_count=validated.manifest.chunk_count,
                        total_characters=total_characters,
                        filename=validated.metadata.filename,
                        title=validated.metadata.title,
                    )
                )
            return sources
        except StorageError as error:
            raise CorruptSourceDocumentError from error

    def _iter_source_chunks(
        self,
        sources: Sequence[SourceSummary],
    ) -> Iterator[tuple[SourceSummary, ChunkRecord]]:
        try:
            for source in sources:
                for chunk in self.documents.iter_document_chunks(source.document_id):
                    yield source, chunk
        except StorageError as error:
            raise CorruptSourceDocumentError from error

    def _add_batch(
        self,
        index: faiss.Index,
        writer: TextIO,
        texts: list[str],
        records: list[IndexRecord],
    ) -> None:
        matrix = self.provider.embed_documents(texts)
        matrix = validate_matrix(
            matrix,
            rows=len(texts),
            dimension=index.d,
        )
        if self.settings.embedding_normalization:
            matrix = normalize_matrix(matrix)
        index.add(matrix)
        for record in records:
            self.indexes.write_record(writer, record)

    def _validate_build_limits(self, chunks: int, characters: int) -> None:
        if (
            chunks > self.settings.maximum_chunks_per_index_build
            or characters > self.settings.maximum_total_embedding_characters_per_build
        ):
            raise IndexResourceLimitError

    def _validate_search(
        self,
        request: SearchRequest,
    ) -> tuple[str, int, float, set[str] | None]:
        query = request.query.strip()
        top_k = (
            self.settings.retrieval_default_top_k
            if request.top_k is None
            else request.top_k
        )
        minimum = (
            self.settings.minimum_similarity_score
            if request.minimum_score is None
            else request.minimum_score
        )
        minimum_bound = -1.0 if self.settings.embedding_normalization else 0.0
        if (
            not query
            or len(query) > self.settings.maximum_query_characters
            or not 1 <= top_k <= self.settings.retrieval_maximum_top_k
            or not minimum_bound <= minimum <= 1.0
            or not math.isfinite(minimum)
        ):
            raise InvalidSearchRequestError
        filters = None
        if request.document_ids is not None:
            filters = set(request.document_ids)
            if (
                not filters
                or len(filters) != len(request.document_ids)
                or any(not DOCUMENT_ID_PATTERN.fullmatch(value) for value in filters)
            ):
                raise InvalidSearchRequestError
        return query, top_k, minimum, filters


def _record(row: int, source: SourceSummary, chunk: ChunkRecord) -> IndexRecord:
    return IndexRecord(
        row=row,
        document_id=source.document_id,
        processing_profile_id=source.processing_profile_id,
        filename=source.filename,
        title=source.title,
        chunk_id=chunk.chunk_id,
        chunk_order=chunk.chunk_order,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_heading=chunk.section_heading,
        text=chunk.text,
        character_count=chunk.character_count,
    )


def _score(value: float, metric: str) -> float:
    if not math.isfinite(value):
        raise IndexIntegrityError
    return value if metric == "cosine" else 1.0 / (1.0 + value)
