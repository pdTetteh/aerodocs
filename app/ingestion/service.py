"""Document ingestion orchestration and read services."""

import logging
import re
import unicodedata
from datetime import UTC, datetime
from typing import BinaryIO

from app.core.config import Settings
from app.ingestion.chunking import create_chunks
from app.ingestion.errors import (
    DocumentError,
    DocumentNotFoundError,
    InternalDocumentError,
    InvalidFilenameError,
    PdfExpansionLimitError,
    ProcessingConfigurationConflictError,
    StorageError,
    UnsupportedMediaTypeError,
)
from app.ingestion.models import (
    ChunkListResponse,
    DocumentIngestResponse,
    DocumentListResponse,
    DocumentManifest,
    DocumentMetadata,
    DocumentSummary,
    PageListResponse,
    ProcessingProfile,
)
from app.ingestion.pdf import extract_pdf
from app.ingestion.profile import create_processing_profile
from app.ingestion.storage import DocumentStore, ValidatedDocument

LOGGER = logging.getLogger("aerodocs.ingestion")
RESERVED_FILENAME_CHARACTERS = re.compile(r'[<>:"|?*]')


class DocumentService:
    """Ingest and read traceable local PDF representations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.processing_profile = create_processing_profile(settings)
        self.store = DocumentStore(settings.document_storage_root)

    def ingest(
        self,
        stream: BinaryIO,
        filename: str | None,
        media_type: str | None,
    ) -> DocumentIngestResponse:
        """Validate, extract, chunk, and atomically persist one PDF."""
        sanitized_filename = sanitize_filename(filename)
        if media_type != self.settings.allowed_pdf_media_type:
            raise UnsupportedMediaTypeError

        staging_directory = self.store.create_staging_directory()
        try:
            digest, file_size = self.store.write_upload(
                staging_directory,
                stream,
                self.settings.max_pdf_upload_bytes,
            )
            self.store.checkpoint("source.pdf")
            document_id = f"doc_{digest}"
            try:
                existing = self.store.validate_document(document_id)
            except DocumentNotFoundError:
                pass
            else:
                self._validate_duplicate(existing, digest)
                return DocumentIngestResponse(
                    document=existing.metadata,
                    duplicate=True,
                )

            extraction = extract_pdf(
                self.store.source_path(staging_directory),
                document_id,
                maximum_pages=self.settings.max_pdf_pages,
                maximum_characters_per_page=(
                    self.settings.max_extracted_characters_per_page
                ),
                maximum_characters_total=(self.settings.max_extracted_characters_total),
            )
            title = extraction.title or filename_stem(sanitized_filename)
            chunks = create_chunks(
                extraction.pages,
                processing_profile_id=(self.processing_profile.processing_profile_id),
                maximum_characters=self.settings.chunk_max_characters,
                overlap_characters=self.settings.chunk_overlap_characters,
            )
            metadata = DocumentMetadata(
                document_id=document_id,
                filename=sanitized_filename,
                sha256=digest,
                title=title,
                author=extraction.author,
                subject=extraction.subject,
                page_count=len(extraction.pages),
                extracted_character_count=sum(
                    page.character_count for page in extraction.pages
                ),
                ingested_at=datetime.now(UTC),
                extraction_format_version=(
                    self.processing_profile.extraction_format_version
                ),
                chunking_format_version=(
                    self.processing_profile.chunking_format_version
                ),
                processing_format_version=(
                    self.processing_profile.processing_format_version
                ),
                processing_profile_id=self.processing_profile.processing_profile_id,
                chunk_max_characters=(self.processing_profile.chunk_max_characters),
                chunk_overlap_characters=(
                    self.processing_profile.chunk_overlap_characters
                ),
                encrypted=False,
                file_size_bytes=file_size,
            )
            self.store.write_artifacts(
                staging_directory,
                metadata,
                extraction.pages,
                chunks,
                self.processing_profile,
            )
            if not self.store.publish(staging_directory, document_id):
                existing = self.store.validate_document(document_id)
                self._validate_duplicate(existing, digest)
                return DocumentIngestResponse(
                    document=existing.metadata,
                    duplicate=True,
                )
            return DocumentIngestResponse(document=metadata, duplicate=False)
        except DocumentError:
            raise
        except Exception as error:
            LOGGER.exception("Unexpected document ingestion failure")
            raise InternalDocumentError from error
        finally:
            try:
                self.store.cleanup_staging(staging_directory)
            except Exception:
                LOGGER.exception("Failed to clean an ingestion staging directory")

    def metadata(self, document_id: str) -> DocumentMetadata:
        """Return metadata without exposing its storage location."""
        try:
            return self.store.validate_document(document_id).metadata
        except DocumentError:
            raise
        except Exception as error:
            LOGGER.exception(
                "Unexpected document metadata read failure",
                extra={"document_id": document_id},
            )
            raise InternalDocumentError from error

    def list_documents(self, *, offset: int, limit: int) -> DocumentListResponse:
        """Return a bounded deterministic list of validated document summaries."""
        try:
            document_ids = self.store.list_document_ids()
            summaries: list[DocumentSummary] = []
            for index, document_id in enumerate(document_ids):
                validated = self.store.validate_document(document_id)
                if offset <= index < offset + limit:
                    metadata = validated.metadata
                    summaries.append(
                        DocumentSummary(
                            document_id=document_id,
                            filename=metadata.filename,
                            title=metadata.title,
                            page_count=validated.manifest.page_count,
                            chunk_count=validated.manifest.chunk_count,
                            processing_profile=ProcessingProfile(
                                processing_profile_id=(metadata.processing_profile_id),
                                processing_format_version=(
                                    metadata.processing_format_version
                                ),
                                extraction_format_version=(
                                    metadata.extraction_format_version
                                ),
                                chunking_format_version=(
                                    metadata.chunking_format_version
                                ),
                                chunk_max_characters=(metadata.chunk_max_characters),
                                chunk_overlap_characters=(
                                    metadata.chunk_overlap_characters
                                ),
                            ),
                            ingested_at=metadata.ingested_at,
                        )
                    )
            return DocumentListResponse(
                offset=offset,
                limit=limit,
                total=len(document_ids),
                documents=summaries,
            )
        except DocumentError:
            raise
        except Exception as error:
            LOGGER.exception("Unexpected document list failure")
            raise InternalDocumentError from error

    def pages(
        self,
        document_id: str,
        *,
        offset: int,
        limit: int,
    ) -> PageListResponse:
        """Return a deterministic page slice."""
        try:
            document = self.store.validate_document(
                document_id,
                page_offset=offset,
                page_limit=limit,
            )
            return PageListResponse(
                document_id=document_id,
                total=document.manifest.page_count,
                offset=offset,
                limit=limit,
                pages=document.pages,
            )
        except DocumentError:
            raise
        except Exception as error:
            LOGGER.exception(
                "Unexpected document page read failure",
                extra={"document_id": document_id},
            )
            raise InternalDocumentError from error

    def chunks(
        self,
        document_id: str,
        *,
        offset: int,
        limit: int,
    ) -> ChunkListResponse:
        """Return a deterministic chunk slice."""
        try:
            document = self.store.validate_document(
                document_id,
                chunk_offset=offset,
                chunk_limit=limit,
            )
            return ChunkListResponse(
                document_id=document_id,
                total=document.manifest.chunk_count,
                offset=offset,
                limit=limit,
                chunks=document.chunks,
            )
        except DocumentError:
            raise
        except Exception as error:
            LOGGER.exception(
                "Unexpected document chunk read failure",
                extra={"document_id": document_id},
            )
            raise InternalDocumentError from error

    def _validate_duplicate(
        self,
        existing: ValidatedDocument,
        digest: str,
    ) -> None:
        """Validate integrity-derived duplicate data against current policy."""
        if existing.metadata.sha256 != digest:
            raise StorageError
        self._ensure_compatible(existing.manifest)
        if (
            existing.manifest.page_count > self.settings.max_pdf_pages
            or existing.maximum_page_characters
            > self.settings.max_extracted_characters_per_page
            or existing.total_page_characters
            > self.settings.max_extracted_characters_total
        ):
            raise PdfExpansionLimitError

    def _ensure_compatible(self, manifest: DocumentManifest) -> None:
        if (
            manifest.processing_profile_id
            == self.processing_profile.processing_profile_id
        ):
            return
        raise ProcessingConfigurationConflictError(
            {
                "processing_profile_id": manifest.processing_profile_id,
                "processing_format_version": manifest.processing_format_version,
                "extraction_format_version": manifest.extraction_format_version,
                "chunking_format_version": manifest.chunking_format_version,
                "chunk_max_characters": manifest.chunk_max_characters,
                "chunk_overlap_characters": manifest.chunk_overlap_characters,
            }
        )


def sanitize_filename(filename: str | None) -> str:
    """Reject traversal and normalize a bounded filesystem-safe filename."""
    if filename is None or not filename.strip():
        raise InvalidFilenameError
    if "\x00" in filename or "/" in filename or "\\" in filename:
        raise InvalidFilenameError
    if filename.strip() in {".", ".."}:
        raise InvalidFilenameError

    sanitized = unicodedata.normalize("NFKC", filename)
    if "/" in sanitized or "\\" in sanitized or sanitized.strip() in {".", ".."}:
        raise InvalidFilenameError
    sanitized = "".join(
        " "
        if character.isspace()
        else ""
        if unicodedata.category(character).startswith("C")
        else character
        for character in sanitized
    )
    sanitized = RESERVED_FILENAME_CHARACTERS.sub("_", sanitized)
    sanitized = " ".join(sanitized.split()).strip(" .")
    sanitized = sanitized[:200].rstrip(" .")
    if not sanitized or not filename_stem(sanitized):
        raise InvalidFilenameError
    return sanitized


def filename_stem(filename: str) -> str:
    """Return a safe human-readable title fallback from a sanitized filename."""
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    stem = stem.strip(" .")
    if not stem:
        raise InvalidFilenameError
    return stem
