"""Descriptor-relative, duplicate-aware evaluation file access."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, BinaryIO

from pydantic import BaseModel, ValidationError

DEFAULT_MAXIMUM_BYTES = 4 * 1024 * 1024
MAXIMUM_JSONL_LINE_BYTES = 64 * 1024
MAXIMUM_EVALUATION_DATASET_BYTES = 32 * 1024 * 1024


class AuxiliaryFileError(ValueError):
    """An evaluation file failed containment or integrity validation."""


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys at every depth."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuxiliaryFileError("duplicate_json_key")
        result[key] = value
    return result


def decode_json(data: bytes) -> Any:
    """Decode UTF-8 JSON with recursive duplicate-key rejection."""
    try:
        return json.loads(data, object_pairs_hook=reject_duplicate_pairs)
    except AuxiliaryFileError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AuxiliaryFileError("invalid_json") from error


def safe_identifier(value: str, *, prefix: str = "") -> str:
    """Validate a bounded lowercase identifier accepted by CLI selectors."""
    import re

    pattern = rf"^{re.escape(prefix)}[a-z0-9][a-z0-9_-]{{1,63}}$"
    if not re.fullmatch(pattern, value):
        raise AuxiliaryFileError("unsafe_identifier")
    return value


def safe_relative_path(name: str) -> tuple[str, ...]:
    """Validate portable POSIX path components without platform separators."""
    if (
        not name
        or name.startswith("/")
        or name.endswith("/")
        or "\\" in name
        or "\x00" in name
        or "//" in name
        or any(separator in name for separator in ("\u2215", "\u2044", "\uff0f"))
    ):
        raise AuxiliaryFileError("unsafe_relative_path")
    parts = tuple(name.split("/"))
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise AuxiliaryFileError("unsafe_relative_path")
    return parts


class ByteBudget:
    """Reserve opened file sizes against one aggregate input ceiling."""

    def __init__(self, maximum_bytes: int = MAXIMUM_EVALUATION_DATASET_BYTES) -> None:
        if maximum_bytes <= 0:
            raise ValueError("maximum_bytes must be positive")
        self.maximum_bytes = maximum_bytes
        self.used_bytes = 0

    def reserve(self, size: int) -> None:
        """Reject before allocation when another opened file exceeds the budget."""
        if size < 0 or self.used_bytes + size > self.maximum_bytes:
            raise AuxiliaryFileError("dataset_too_large")
        self.used_bytes += size


class SecureDirectory:
    """Open files relative to one held root descriptor without path re-resolution."""

    def __init__(self, root: Path) -> None:
        if (
            not hasattr(os, "O_NOFOLLOW")
            or not hasattr(os, "O_DIRECTORY")
            or os.open not in os.supports_dir_fd
        ):
            raise AuxiliaryFileError("secure_descriptor_access_unavailable")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            self._descriptor = os.open(root, flags)
        except OSError as error:
            raise AuxiliaryFileError("root_unavailable") from error
        status = os.fstat(self._descriptor)
        if not stat.S_ISDIR(status.st_mode):
            os.close(self._descriptor)
            raise AuxiliaryFileError("unsafe_root")

    def close(self) -> None:
        descriptor = getattr(self, "_descriptor", -1)
        if descriptor >= 0:
            os.close(descriptor)
            self._descriptor = -1

    def __enter__(self) -> SecureDirectory:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def open_directory(self, name: str) -> SecureDirectory:
        """Open one child directory and transfer its verified descriptor."""
        parts = safe_relative_path(name)
        descriptor = self._walk_directories(parts)
        child = object.__new__(SecureDirectory)
        child._descriptor = descriptor
        return child

    def read(
        self,
        name: str,
        *,
        maximum_bytes: int = DEFAULT_MAXIMUM_BYTES,
        budget: ByteBudget | None = None,
    ) -> bytes:
        """Read a contained regular file from the already-open root."""
        if maximum_bytes <= 0:
            raise ValueError("maximum_bytes must be positive")
        parts = safe_relative_path(name)
        parent = self._walk_directories(parts[:-1])
        try:
            flags = os.O_RDONLY | os.O_NOFOLLOW
            descriptor = os.open(parts[-1], flags, dir_fd=parent)
            try:
                status = os.fstat(descriptor)
                if not stat.S_ISREG(status.st_mode):
                    raise AuxiliaryFileError("not_regular_file")
                if status.st_size > maximum_bytes:
                    raise AuxiliaryFileError("file_too_large")
                if budget is not None:
                    budget.reserve(status.st_size)
                chunks: list[bytes] = []
                remaining = maximum_bytes + 1
                read_size = 0
                reserved = status.st_size
                while remaining:
                    chunk = os.read(descriptor, min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    read_size += len(chunk)
                    if budget is not None and read_size > reserved:
                        budget.reserve(read_size - reserved)
                        reserved = read_size
                    chunks.append(chunk)
                    remaining -= len(chunk)
                data = b"".join(chunks)
                if len(data) > maximum_bytes:
                    raise AuxiliaryFileError("file_too_large")
                return data
            finally:
                os.close(descriptor)
        except AuxiliaryFileError:
            raise
        except OSError as error:
            raise AuxiliaryFileError("file_read_failed") from error
        finally:
            if parent != self._descriptor:
                os.close(parent)

    def sha256(
        self,
        name: str,
        *,
        maximum_bytes: int,
        budget: ByteBudget | None = None,
    ) -> tuple[str, int]:
        """Stream and hash a contained regular file without retaining its bytes."""
        import hashlib

        parts = safe_relative_path(name)
        parent = self._walk_directories(parts[:-1])
        descriptor = -1
        try:
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode):
                raise AuxiliaryFileError("not_regular_file")
            if status.st_size > maximum_bytes:
                raise AuxiliaryFileError("file_too_large")
            if budget is not None:
                budget.reserve(status.st_size)
            digest = hashlib.sha256()
            total = 0
            reserved = status.st_size
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    raise AuxiliaryFileError("file_too_large")
                if budget is not None and total > reserved:
                    budget.reserve(total - reserved)
                    reserved = total
                digest.update(chunk)
            return digest.hexdigest(), total
        except AuxiliaryFileError:
            raise
        except OSError as error:
            raise AuxiliaryFileError("file_read_failed") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if parent != self._descriptor:
                os.close(parent)

    def open_binary(self, name: str) -> BinaryIO:
        """Return an opened, verified regular file owned by the caller."""
        parts = safe_relative_path(name)
        parent = self._walk_directories(parts[:-1])
        descriptor = -1
        try:
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise AuxiliaryFileError("not_regular_file")
            stream = os.fdopen(descriptor, "rb")
            descriptor = -1
            return stream
        except AuxiliaryFileError:
            raise
        except OSError as error:
            raise AuxiliaryFileError("file_read_failed") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if parent != self._descriptor:
                os.close(parent)

    def _walk_directories(self, parts: tuple[str, ...]) -> int:
        current = os.dup(self._descriptor)
        if not parts:
            return current
        try:
            for part in parts:
                following = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=current,
                )
                os.close(current)
                current = following
                if not stat.S_ISDIR(os.fstat(current).st_mode):
                    raise AuxiliaryFileError("unsafe_parent")
            return current
        except AuxiliaryFileError:
            os.close(current)
            raise
        except OSError as error:
            os.close(current)
            raise AuxiliaryFileError("unsafe_parent") from error


def read_bytes(
    root: Path,
    name: str,
    *,
    maximum_bytes: int = DEFAULT_MAXIMUM_BYTES,
) -> bytes:
    """Compatibility wrapper around descriptor-relative access."""
    with SecureDirectory(root) as directory:
        return directory.read(name, maximum_bytes=maximum_bytes)


def read_json_model(
    root: Path,
    name: str,
    model: type[BaseModel],
    *,
    maximum_bytes: int = DEFAULT_MAXIMUM_BYTES,
) -> BaseModel:
    """Parse a strict model after duplicate-aware secure decoding."""
    try:
        return model.model_validate(
            decode_json(read_bytes(root, name, maximum_bytes=maximum_bytes))
        )
    except AuxiliaryFileError:
        raise
    except ValidationError as error:
        raise AuxiliaryFileError("invalid_json") from error
