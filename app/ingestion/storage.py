"""Contained, atomic, manifest-validated local document storage."""

import errno
import hashlib
import json
import os
import re
import shutil
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO, cast
from uuid import uuid4

from pydantic import ValidationError

from app.ingestion.errors import (
    DocumentNotFoundError,
    EmptyUploadError,
    InvalidPdfSignatureError,
    StorageError,
    UploadTooLargeError,
)
from app.ingestion.models import (
    ArtifactDescriptor,
    ChunkRecord,
    DocumentManifest,
    DocumentMetadata,
    PageRecord,
    ProcessingProfile,
)
from app.ingestion.profile import calculate_processing_profile_id

DOCUMENT_ID_PATTERN = re.compile(r"^doc_[0-9a-f]{64}$")
READ_BLOCK_SIZE = 1024 * 1024
REQUIRED_ARTIFACTS = (
    "source.pdf",
    "metadata.json",
    "pages.jsonl",
    "chunks.jsonl",
)
PUBLICATION_LOCK_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class ValidatedDocument:
    """Validated metadata, manifest, and bounded requested records."""

    metadata: DocumentMetadata
    manifest: DocumentManifest
    pages: list[PageRecord]
    chunks: list[ChunkRecord]
    maximum_page_characters: int
    total_page_characters: int


class DocumentStore:
    """Persist and validate complete results beneath one configured root."""

    def __init__(self, root: Path) -> None:
        configured_root = root.expanduser().absolute()
        if configured_root.is_symlink():
            raise StorageError
        configured_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if configured_root.is_symlink() or not configured_root.is_dir():
            raise StorageError

        self.configured_root = configured_root
        self.root = configured_root.resolve(strict=True)
        self.documents_root = self.root / "documents"
        self.staging_root = self.root / ".staging"
        self._ensure_managed_directory(self.documents_root)
        self._ensure_managed_directory(self.staging_root)

    def create_staging_directory(self) -> Path:
        """Create a unique private staging directory."""
        self._validate_managed_roots()
        path = self.staging_root / f"upload-{uuid4().hex}"
        self._assert_inside_root(path)
        try:
            path.mkdir(mode=0o700)
        except OSError as error:
            raise StorageError from error
        self._require_directory(path)
        return path

    def write_upload(
        self,
        staging_directory: Path,
        stream: BinaryIO,
        maximum_bytes: int,
    ) -> tuple[str, int]:
        """Stream an upload to staging while hashing and enforcing limits."""
        self._require_staging_directory(staging_directory)
        source_path = staging_directory / "source.pdf"
        digest = hashlib.sha256()
        size = 0
        signature = b""

        try:
            with self._open_exclusive_binary(source_path) as destination:
                while block := stream.read(READ_BLOCK_SIZE):
                    size += len(block)
                    if size > maximum_bytes:
                        raise UploadTooLargeError
                    if len(signature) < 5:
                        signature = (signature + block)[:5]
                    digest.update(block)
                    destination.write(block)
                destination.flush()
                os.fsync(destination.fileno())
        except DocumentNotFoundError as error:
            raise StorageError from error

        if size == 0:
            raise EmptyUploadError
        if signature != b"%PDF-":
            raise InvalidPdfSignatureError
        return digest.hexdigest(), size

    def write_artifacts(
        self,
        staging_directory: Path,
        metadata: DocumentMetadata,
        pages: list[PageRecord],
        chunks: list[ChunkRecord],
        profile: ProcessingProfile,
    ) -> DocumentManifest:
        """Write artifacts and their integrity manifest, with manifest last."""
        self._require_staging_directory(staging_directory)
        self._atomic_write_text(
            staging_directory / "metadata.json",
            _model_json(metadata),
        )
        self.checkpoint("metadata.json")
        self._atomic_write_jsonl(staging_directory / "pages.jsonl", pages)
        self.checkpoint("pages.jsonl")
        self._atomic_write_jsonl(staging_directory / "chunks.jsonl", chunks)
        self.checkpoint("chunks.jsonl")

        descriptors = {
            filename: self._describe_regular_file(staging_directory / filename)
            for filename in REQUIRED_ARTIFACTS
        }
        manifest = DocumentManifest(
            manifest_format_version="1",
            document_id=metadata.document_id,
            source_sha256=metadata.sha256,
            processing_profile_id=profile.processing_profile_id,
            processing_format_version=profile.processing_format_version,
            extraction_format_version=profile.extraction_format_version,
            chunking_format_version=profile.chunking_format_version,
            chunk_max_characters=profile.chunk_max_characters,
            chunk_overlap_characters=profile.chunk_overlap_characters,
            page_count=len(pages),
            chunk_count=len(chunks),
            artifacts=descriptors,
        )
        self._atomic_write_text(
            staging_directory / "manifest.json",
            _model_json(manifest),
        )
        self.checkpoint("manifest.json")
        return manifest

    def checkpoint(self, _artifact_name: str) -> None:
        """Provide a deterministic failure-injection point for storage tests."""

    def publish(self, staging_directory: Path, document_id: str) -> bool:
        """Publish without replacing an existing application-owned target."""
        self._require_staging_directory(staging_directory)
        destination = self._document_directory(document_id, require_exists=False)
        with self._publication_lock(document_id):
            self._validate_managed_roots()
            destination_status = _lstat(destination)
            if destination_status is not None:
                self._require_directory(destination)
                return False
            try:
                os.rename(staging_directory, destination)
            except OSError as error:
                if error.errno in {errno.EEXIST, errno.ENOTEMPTY}:
                    return False
                raise StorageError from error
            self._require_directory(destination)
        return True

    def validate_document(
        self,
        document_id: str,
        *,
        page_offset: int = 0,
        page_limit: int = 0,
        chunk_offset: int = 0,
        chunk_limit: int = 0,
    ) -> ValidatedDocument:
        """Validate all persisted integrity and retain only requested windows."""
        directory = self._document_directory(document_id, require_exists=True)
        manifest = cast(
            DocumentManifest,
            self._read_model(directory / "manifest.json", DocumentManifest),
        )
        self._validate_manifest_identity(manifest, document_id)
        self._validate_artifact_descriptors(directory, manifest)

        metadata = cast(
            DocumentMetadata,
            self._read_model(directory / "metadata.json", DocumentMetadata),
        )
        self._validate_metadata(metadata, manifest)
        pages, maximum_page_characters, total_page_characters = self._stream_pages(
            directory / "pages.jsonl",
            manifest,
            offset=page_offset,
            limit=page_limit,
        )
        chunks = self._stream_chunks(
            directory / "chunks.jsonl",
            manifest,
            offset=chunk_offset,
            limit=chunk_limit,
        )
        if metadata.extracted_character_count != total_page_characters:
            raise StorageError
        return ValidatedDocument(
            metadata=metadata,
            manifest=manifest,
            pages=pages,
            chunks=chunks,
            maximum_page_characters=maximum_page_characters,
            total_page_characters=total_page_characters,
        )

    def cleanup_staging(self, staging_directory: Path) -> None:
        """Remove only a non-symlinked staging directory owned by this store."""
        self._validate_managed_roots()
        self._assert_inside_root(staging_directory)
        status = _lstat(staging_directory)
        if status is None:
            return
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise StorageError
        shutil.rmtree(staging_directory)

    def source_path(self, staging_directory: Path) -> Path:
        """Return a verified regular staged source path."""
        self._require_staging_directory(staging_directory)
        path = staging_directory / "source.pdf"
        self._require_regular_file(path)
        return path

    def _stream_pages(
        self,
        path: Path,
        manifest: DocumentManifest,
        *,
        offset: int,
        limit: int,
    ) -> tuple[list[PageRecord], int, int]:
        selected: list[PageRecord] = []
        expected_number = 1
        maximum_characters = 0
        total_characters = 0
        for line in self._iter_jsonl(path):
            try:
                page = PageRecord.model_validate_json(line)
            except ValidationError as error:
                raise StorageError from error
            if (
                page.document_id != manifest.document_id
                or page.page_number != expected_number
                or page.character_count != len(page.text)
            ):
                raise StorageError
            maximum_characters = max(maximum_characters, page.character_count)
            total_characters += page.character_count
            index = expected_number - 1
            if limit > 0 and offset <= index < offset + limit:
                selected.append(page)
            expected_number += 1
        if expected_number - 1 != manifest.page_count:
            raise StorageError
        return selected, maximum_characters, total_characters

    def _stream_chunks(
        self,
        path: Path,
        manifest: DocumentManifest,
        *,
        offset: int,
        limit: int,
    ) -> list[ChunkRecord]:
        selected: list[ChunkRecord] = []
        expected_order = 1
        for line in self._iter_jsonl(path):
            try:
                chunk = ChunkRecord.model_validate_json(line)
            except ValidationError as error:
                raise StorageError from error
            if (
                chunk.document_id != manifest.document_id
                or chunk.chunk_order != expected_order
                or chunk.character_count != len(chunk.text)
                or chunk.page_start > chunk.page_end
                or chunk.page_end > manifest.page_count
            ):
                raise StorageError
            index = expected_order - 1
            if limit > 0 and offset <= index < offset + limit:
                selected.append(chunk)
            expected_order += 1
        if expected_order - 1 != manifest.chunk_count:
            raise StorageError
        return selected

    def _validate_manifest_identity(
        self,
        manifest: DocumentManifest,
        document_id: str,
    ) -> None:
        if (
            manifest.document_id != document_id
            or document_id != f"doc_{manifest.source_sha256}"
            or set(manifest.artifacts) != set(REQUIRED_ARTIFACTS)
            or manifest.processing_profile_id
            != calculate_processing_profile_id(
                processing_format_version=manifest.processing_format_version,
                extraction_format_version=manifest.extraction_format_version,
                chunking_format_version=manifest.chunking_format_version,
                chunk_max_characters=manifest.chunk_max_characters,
                chunk_overlap_characters=manifest.chunk_overlap_characters,
            )
        ):
            raise StorageError

    def _validate_artifact_descriptors(
        self,
        directory: Path,
        manifest: DocumentManifest,
    ) -> None:
        for filename in REQUIRED_ARTIFACTS:
            descriptor = manifest.artifacts[filename]
            if descriptor.filename != filename:
                raise StorageError
            actual = self._describe_regular_file(directory / filename)
            if actual != descriptor:
                raise StorageError

    @staticmethod
    def _validate_metadata(
        metadata: DocumentMetadata,
        manifest: DocumentManifest,
    ) -> None:
        if (
            metadata.document_id != manifest.document_id
            or metadata.sha256 != manifest.source_sha256
            or metadata.page_count != manifest.page_count
            or metadata.file_size_bytes != manifest.artifacts["source.pdf"].byte_size
            or metadata.processing_profile_id != manifest.processing_profile_id
            or metadata.processing_format_version != manifest.processing_format_version
            or metadata.extraction_format_version != manifest.extraction_format_version
            or metadata.chunking_format_version != manifest.chunking_format_version
            or metadata.chunk_max_characters != manifest.chunk_max_characters
            or metadata.chunk_overlap_characters != manifest.chunk_overlap_characters
        ):
            raise StorageError

    def _document_directory(
        self,
        document_id: str,
        *,
        require_exists: bool,
    ) -> Path:
        self._validate_managed_roots()
        if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
            raise DocumentNotFoundError
        path = self.documents_root / document_id
        self._assert_inside_root(path)
        status = _lstat(path)
        if status is None:
            if require_exists:
                raise DocumentNotFoundError
            return path
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise StorageError
        return path

    def _atomic_write_jsonl(
        self,
        path: Path,
        records: list[PageRecord] | list[ChunkRecord],
    ) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with self._open_exclusive_text(temporary) as output:
                for record in records:
                    output.write(
                        json.dumps(
                            record.model_dump(mode="json"),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            self._replace_staged_file(temporary, path)
        finally:
            self._unlink_regular_file_if_present(temporary)

    def _atomic_write_text(self, path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with self._open_exclusive_text(temporary) as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            self._replace_staged_file(temporary, path)
        finally:
            self._unlink_regular_file_if_present(temporary)

    def _replace_staged_file(self, temporary: Path, destination: Path) -> None:
        self._require_regular_file(temporary)
        self._assert_inside_root(destination)
        if _lstat(destination) is not None:
            raise StorageError
        try:
            os.replace(temporary, destination)
        except OSError as error:
            raise StorageError from error
        self._require_regular_file(destination)

    def _read_model(
        self,
        path: Path,
        model: type[DocumentManifest] | type[DocumentMetadata],
    ) -> DocumentManifest | DocumentMetadata:
        try:
            with self._open_regular_text(path) as source:
                return model.model_validate_json(source.read())
        except (OSError, UnicodeError, ValidationError) as error:
            raise StorageError from error

    def _iter_jsonl(self, path: Path) -> Iterator[str]:
        try:
            with self._open_regular_text(path) as source:
                for line in source:
                    if not line.strip():
                        raise StorageError
                    yield line
        except (OSError, UnicodeError) as error:
            raise StorageError from error

    def _describe_regular_file(self, path: Path) -> ArtifactDescriptor:
        self._require_regular_file(path)
        digest = hashlib.sha256()
        byte_size = 0
        try:
            with self._open_regular_binary(path) as source:
                while block := source.read(READ_BLOCK_SIZE):
                    digest.update(block)
                    byte_size += len(block)
        except OSError as error:
            raise StorageError from error
        return ArtifactDescriptor(
            filename=path.name,
            sha256=digest.hexdigest(),
            byte_size=byte_size,
        )

    def _validate_managed_roots(self) -> None:
        try:
            invalid_root = (
                self.configured_root.is_symlink()
                or self.configured_root.resolve(strict=True) != self.root
            )
        except OSError as error:
            raise StorageError from error
        if invalid_root:
            raise StorageError
        self._require_directory(self.documents_root)
        self._require_directory(self.staging_root)

    def _ensure_managed_directory(self, path: Path) -> None:
        self._assert_inside_root(path)
        status = _lstat(path)
        if status is not None:
            if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
                raise StorageError
            return
        try:
            path.mkdir(mode=0o700)
        except OSError as error:
            raise StorageError from error
        self._require_directory(path)

    def _require_staging_directory(self, path: Path) -> None:
        self._validate_managed_roots()
        self._assert_inside_root(path)
        try:
            path.relative_to(self.staging_root)
        except ValueError as error:
            raise StorageError from error
        self._require_directory(path)

    def _require_directory(self, path: Path) -> None:
        self._assert_inside_root(path)
        status = _lstat(path)
        if (
            status is None
            or stat.S_ISLNK(status.st_mode)
            or not stat.S_ISDIR(status.st_mode)
        ):
            raise StorageError

    def _require_regular_file(self, path: Path) -> None:
        self._assert_inside_root(path)
        status = _lstat(path)
        if (
            status is None
            or stat.S_ISLNK(status.st_mode)
            or not stat.S_ISREG(status.st_mode)
        ):
            raise StorageError

    def _assert_inside_root(self, path: Path) -> None:
        try:
            path.resolve(strict=False).relative_to(self.root)
        except (OSError, ValueError) as error:
            raise StorageError from error

    def _open_exclusive_binary(self, path: Path) -> BinaryIO:
        self._assert_inside_root(path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _no_follow_flag()
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as error:
            raise StorageError from error
        return os.fdopen(descriptor, "wb")

    def _open_exclusive_text(self, path: Path) -> TextIO:
        self._assert_inside_root(path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _no_follow_flag()
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as error:
            raise StorageError from error
        return os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")

    def _open_regular_binary(self, path: Path) -> BinaryIO:
        self._require_regular_file(path)
        try:
            descriptor = os.open(path, os.O_RDONLY | _no_follow_flag())
        except OSError as error:
            raise StorageError from error
        return os.fdopen(descriptor, "rb")

    def _open_regular_text(self, path: Path) -> TextIO:
        self._require_regular_file(path)
        try:
            descriptor = os.open(path, os.O_RDONLY | _no_follow_flag())
        except OSError as error:
            raise StorageError from error
        return os.fdopen(descriptor, "r", encoding="utf-8", newline="")

    def _unlink_regular_file_if_present(self, path: Path) -> None:
        status = _lstat(path)
        if status is None:
            return
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise StorageError
        path.unlink()

    @contextmanager
    def _publication_lock(self, document_id: str) -> Iterator[None]:
        lock_path = self.documents_root / f".publish-{document_id}.lock"
        self._assert_inside_root(lock_path)
        deadline = time.monotonic() + PUBLICATION_LOCK_TIMEOUT_SECONDS
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | _no_follow_flag(),
                    0o600,
                )
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise StorageError from None
                time.sleep(0.01)
            except OSError as error:
                raise StorageError from error
        try:
            yield
        finally:
            os.close(descriptor)
            status = _lstat(lock_path)
            if status is None or stat.S_ISLNK(status.st_mode):
                raise StorageError
            lock_path.unlink()


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise StorageError from error


def _no_follow_flag() -> int:
    return int(getattr(os, "O_NOFOLLOW", 0))


def _model_json(model: DocumentMetadata | DocumentManifest) -> str:
    return (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
