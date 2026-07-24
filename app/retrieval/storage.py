"""Contained versioned FAISS storage with atomic active-pointer publication."""

import hashlib
import json
import os
import re
import shutil
import stat
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, TextIO, cast
from uuid import uuid4

import faiss
from pydantic import BaseModel, ValidationError

from app.retrieval.errors import (
    IndexBuildInProgressError,
    IndexIntegrityError,
    IndexNotReadyError,
)
from app.retrieval.models import (
    ActivePointer,
    IndexArtifact,
    IndexManifest,
    IndexMetadata,
    IndexPublication,
    IndexRecord,
)

ARTIFACTS = ("index.faiss", "records.jsonl", "metadata.json")
BLOCK_SIZE = 1024 * 1024
VERSION_ID_PATTERN = re.compile(r"idx_[0-9a-f]{64}(?:_[0-9a-f]{32})?")


class IndexStore:
    """Write and validate immutable index versions beneath one root."""

    def __init__(self, root: Path, lock_timeout: float) -> None:
        configured = root.expanduser().absolute()
        if configured.is_symlink():
            raise IndexIntegrityError
        configured.mkdir(parents=True, exist_ok=True, mode=0o700)
        if configured.is_symlink() or not configured.is_dir():
            raise IndexIntegrityError
        self.root = configured.resolve(strict=True)
        self.staging_root = self.root / ".staging"
        self.versions_root = self.root / "versions"
        self.locks_root = self.root / ".locks"
        self.active_path = self.root / "active.json"
        self.lock_timeout = lock_timeout
        for path in (self.staging_root, self.versions_root, self.locks_root):
            self._ensure_directory(path)

    @contextmanager
    def build_lock(self) -> Iterator[None]:
        """Acquire an OS-owned lock that is released when the process exits."""
        lock_path = self.locks_root / "rebuild.lock"
        self._assert_inside(lock_path)
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | _no_follow_flag(),
            0o600,
        )
        deadline = time.monotonic() + self.lock_timeout
        acquired = False
        try:
            while not acquired:
                try:
                    _lock_descriptor(descriptor)
                    acquired = True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise IndexBuildInProgressError from None
                    time.sleep(0.02)
            yield
        finally:
            if acquired:
                _unlock_descriptor(descriptor)
            os.close(descriptor)

    def create_staging(self) -> Path:
        self._validate_roots()
        path = self.staging_root / f"build-{uuid4().hex}"
        path.mkdir(mode=0o700)
        self._require_directory(path)
        return path

    def records_writer(self, staging: Path) -> TextIO:
        self._require_staging(staging)
        return self._open_exclusive_text(staging / "records.jsonl")

    @staticmethod
    def write_record(output: TextIO, record: IndexRecord) -> None:
        output.write(
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )

    def finish_records(self, output: TextIO) -> None:
        output.flush()
        os.fsync(output.fileno())
        output.close()

    def write_index(self, staging: Path, index: faiss.Index) -> None:
        self._require_staging(staging)
        path = staging / "index.faiss"
        if path.exists():
            raise IndexIntegrityError
        try:
            faiss.write_index(index, str(path))
        except Exception as error:
            raise IndexIntegrityError from error
        self._require_regular(path)

    def write_metadata(self, staging: Path, metadata: IndexMetadata) -> None:
        self._write_json(staging / "metadata.json", metadata)

    def write_manifest(
        self,
        staging: Path,
        metadata: IndexMetadata,
        version_id: str,
    ) -> IndexManifest:
        descriptors = {name: self._describe(staging / name) for name in ARTIFACTS}
        manifest = IndexManifest(
            **metadata.model_dump(),
            manifest_format_version="1",
            version_id=version_id,
            artifacts=descriptors,
        )
        self._write_json(staging / "manifest.json", manifest)
        return manifest

    @staticmethod
    def create_version_id(index_profile_id: str) -> str:
        """Create a private immutable physical ID for one artifact set."""
        profile_digest = hashlib.sha256(index_profile_id.encode()).hexdigest()
        return f"idx_{profile_digest}_{uuid4().hex}"

    def find_compatible(
        self,
        index_profile_id: str,
        corpus_fingerprint: str,
    ) -> IndexManifest | None:
        """Choose the lexicographically first fully valid compatible version."""
        self._validate_roots()
        matches: list[IndexManifest] = []
        directories = sorted(
            self.versions_root.iterdir(),
            key=lambda path: path.name,
        )
        for directory in directories:
            if not VERSION_ID_PATTERN.fullmatch(directory.name):
                continue
            try:
                manifest, _index = self.validate_directory(
                    directory,
                    expected_version=directory.name,
                )
            except IndexIntegrityError:
                continue
            if (
                manifest.index_profile_id == index_profile_id
                and manifest.corpus_fingerprint == corpus_fingerprint
            ):
                matches.append(manifest)
        return matches[0] if matches else None

    def reactivate(self, manifest: IndexManifest) -> IndexPublication:
        """Atomically reactivate one independently validated stored version."""
        directory = self.versions_root / manifest.version_id
        stored, _index = self.validate_directory(
            directory,
            expected_version=manifest.version_id,
        )
        if (
            stored.index_profile_id != manifest.index_profile_id
            or stored.corpus_fingerprint != manifest.corpus_fingerprint
        ):
            raise IndexIntegrityError
        self._activate(stored)
        return IndexPublication(created=False, manifest=stored)

    def publish(
        self,
        staging: Path,
        manifest: IndexManifest,
    ) -> IndexPublication:
        """Publish a new immutable version and atomically activate it."""
        staged_manifest, _index = self.validate_directory(
            staging,
            expected_version=manifest.version_id,
        )
        if staged_manifest != manifest:
            raise IndexIntegrityError
        destination = self.versions_root / manifest.version_id
        self._assert_inside(destination)
        if _lstat(destination) is not None:
            raise IndexIntegrityError
        try:
            os.rename(staging, destination)
        except OSError as error:
            raise IndexIntegrityError from error
        try:
            stored, _index = self.validate_directory(
                destination,
                expected_version=manifest.version_id,
            )
            self._activate(stored)
        except Exception:
            self._remove_new_version(destination)
            raise
        return IndexPublication(created=True, manifest=stored)

    def _activate(self, manifest: IndexManifest) -> None:
        directory = self.versions_root / manifest.version_id
        stored, _index = self.validate_directory(
            directory,
            expected_version=manifest.version_id,
        )
        if stored != manifest:
            raise IndexIntegrityError
        manifest_hash = self._describe(directory / "manifest.json").sha256
        pointer = ActivePointer(
            version_id=manifest.version_id,
            manifest_sha256=manifest_hash,
        )
        temporary = self.root / f".active-{uuid4().hex}.tmp"
        self._write_json(temporary, pointer)
        try:
            os.replace(temporary, self.active_path)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            raise IndexIntegrityError from error

    def _remove_new_version(self, directory: Path) -> None:
        self._require_directory(directory)
        shutil.rmtree(directory)

    def active(self) -> tuple[IndexManifest, faiss.Index]:
        if _lstat(self.active_path) is None:
            raise IndexNotReadyError
        pointer = cast(ActivePointer, self._read_model(self.active_path, ActivePointer))
        directory = self.versions_root / pointer.version_id
        if (
            self._describe(directory / "manifest.json").sha256
            != pointer.manifest_sha256
        ):
            raise IndexIntegrityError
        manifest, index = self.validate_directory(
            directory,
            expected_version=pointer.version_id,
        )
        return manifest, index

    def validate_directory(
        self,
        directory: Path,
        *,
        expected_version: str,
    ) -> tuple[IndexManifest, faiss.Index]:
        self._require_directory(directory)
        manifest = cast(
            IndexManifest,
            self._read_model(directory / "manifest.json", IndexManifest),
        )
        if manifest.version_id != expected_version or set(manifest.artifacts) != set(
            ARTIFACTS
        ):
            raise IndexIntegrityError
        for name in ARTIFACTS:
            if manifest.artifacts[name] != self._describe(directory / name):
                raise IndexIntegrityError
        metadata = cast(
            IndexMetadata,
            self._read_model(directory / "metadata.json", IndexMetadata),
        )
        if metadata != IndexMetadata.model_validate(manifest.model_dump()):
            raise IndexIntegrityError
        if (
            manifest.vector_count != manifest.indexed_chunk_count
            or manifest.source_document_count != len(manifest.sources)
            or len({source.document_id for source in manifest.sources})
            != len(manifest.sources)
        ):
            raise IndexIntegrityError
        sources = {source.document_id: source for source in manifest.sources}
        source_records: Counter[str] = Counter()
        source_characters: Counter[str] = Counter()
        chunk_ids: set[str] = set()
        records = 0
        for record in self.iter_records(directory):
            source = sources.get(record.document_id)
            if (
                record.row != records
                or record.character_count != len(record.text)
                or record.chunk_id in chunk_ids
                or source is None
                or record.processing_profile_id != source.processing_profile_id
                or record.filename != source.filename
                or record.title != source.title
                or record.page_end < record.page_start
            ):
                raise IndexIntegrityError
            chunk_ids.add(record.chunk_id)
            source_records[record.document_id] += 1
            source_characters[record.document_id] += record.character_count
            records += 1
        if (
            records != manifest.vector_count
            or sum(source_characters.values()) != manifest.indexed_character_count
            or any(
                source_records[source.document_id] != source.chunk_count
                or source_characters[source.document_id] != source.total_characters
                for source in manifest.sources
            )
            or sorted(
                source.document_id
                for source in manifest.sources
                if source.chunk_count == 0
            )
            != sorted(manifest.zero_chunk_document_ids)
        ):
            raise IndexIntegrityError
        try:
            index = faiss.read_index(str(directory / "index.faiss"))
        except Exception as error:
            raise IndexIntegrityError from error
        if (
            index.ntotal != manifest.vector_count
            or index.d != manifest.model.dimension
            or not index.is_trained
        ):
            raise IndexIntegrityError
        if index.__class__.__name__ != manifest.index_type:
            raise IndexIntegrityError
        expected_metric = (
            faiss.METRIC_INNER_PRODUCT
            if manifest.index_type == "IndexFlatIP"
            else faiss.METRIC_L2
        )
        if index.metric_type != expected_metric:
            raise IndexIntegrityError
        return manifest, index

    def iter_records(self, directory: Path) -> Iterator[IndexRecord]:
        path = directory / "records.jsonl"
        try:
            with self._open_regular_text(path) as source:
                for line in source:
                    if not line.strip():
                        raise IndexIntegrityError
                    try:
                        yield IndexRecord.model_validate_json(line)
                    except ValidationError as error:
                        raise IndexIntegrityError from error
        except (OSError, UnicodeError) as error:
            raise IndexIntegrityError from error

    def active_directory(self, manifest: IndexManifest) -> Path:
        path = self.versions_root / manifest.version_id
        self._require_directory(path)
        return path

    def cleanup_staging(self, staging: Path) -> None:
        self._assert_inside(staging)
        status = _lstat(staging)
        if status is None:
            return
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise IndexIntegrityError
        shutil.rmtree(staging)

    def _write_json(self, path: Path, model: BaseModel) -> None:
        output = self._open_exclusive_text(path)
        try:
            value = model.model_dump(mode="json")
            output.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        finally:
            output.close()

    def _read_model(self, path: Path, model: type[BaseModel]) -> BaseModel:
        try:
            with self._open_regular_text(path) as source:
                return model.model_validate_json(source.read())
        except (OSError, UnicodeError, ValidationError) as error:
            raise IndexIntegrityError from error

    def _describe(self, path: Path) -> IndexArtifact:
        digest = hashlib.sha256()
        size = 0
        with self._open_regular_binary(path) as source:
            while block := source.read(BLOCK_SIZE):
                digest.update(block)
                size += len(block)
        return IndexArtifact(
            filename=path.name, sha256=digest.hexdigest(), byte_size=size
        )

    def _validate_roots(self) -> None:
        for path in (self.root, self.staging_root, self.versions_root, self.locks_root):
            self._require_directory(path)

    def _ensure_directory(self, path: Path) -> None:
        self._assert_inside(path)
        if _lstat(path) is None:
            path.mkdir(mode=0o700)
        self._require_directory(path)

    def _require_staging(self, path: Path) -> None:
        self._validate_roots()
        try:
            path.relative_to(self.staging_root)
        except ValueError as error:
            raise IndexIntegrityError from error
        self._require_directory(path)

    def _require_directory(self, path: Path) -> None:
        self._assert_inside(path)
        status = _lstat(path)
        if (
            status is None
            or stat.S_ISLNK(status.st_mode)
            or not stat.S_ISDIR(status.st_mode)
        ):
            raise IndexIntegrityError

    def _require_regular(self, path: Path) -> None:
        self._assert_inside(path)
        status = _lstat(path)
        if (
            status is None
            or stat.S_ISLNK(status.st_mode)
            or not stat.S_ISREG(status.st_mode)
        ):
            raise IndexIntegrityError

    def _assert_inside(self, path: Path) -> None:
        try:
            path.resolve(strict=False).relative_to(self.root)
        except (OSError, ValueError) as error:
            raise IndexIntegrityError from error

    def _open_exclusive_text(self, path: Path) -> TextIO:
        self._assert_inside(path)
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _no_follow_flag(),
                0o600,
            )
        except OSError as error:
            raise IndexIntegrityError from error
        return os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")

    def _open_regular_text(self, path: Path) -> TextIO:
        descriptor = self._open_regular_descriptor(path)
        return os.fdopen(descriptor, "r", encoding="utf-8", newline="")

    def _open_regular_binary(self, path: Path) -> BinaryIO:
        descriptor = self._open_regular_descriptor(path)
        return os.fdopen(descriptor, "rb")

    def _open_regular_descriptor(self, path: Path) -> int:
        self._require_regular(path)
        try:
            return os.open(path, os.O_RDONLY | _no_follow_flag())
        except OSError as error:
            raise IndexIntegrityError from error


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise IndexIntegrityError from error


def _no_follow_flag() -> int:
    return int(getattr(os, "O_NOFOLLOW", 0))


def _lock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)
