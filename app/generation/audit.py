"""Atomic, path-contained generation audit persistence."""

import json
import os
import stat
from pathlib import Path
from uuid import uuid4

from app.generation.errors import AuditPersistenceError
from app.generation.models import GenerationAudit


class GenerationAuditStore:
    """Persist sanitized audit records in one configured runtime directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, record: GenerationAudit) -> None:
        """Atomically write one immutable audit record without following links."""
        temporary: Path | None = None
        try:
            self._prepare_root()
            destination = self._contained(f"{record.audit_id}.json")
            temporary = self._contained(f".{record.audit_id}.{uuid4().hex}.tmp")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(temporary, flags, 0o600)
            try:
                payload = (
                    json.dumps(
                        record.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    if written <= 0:
                        raise AuditPersistenceError
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if destination.exists() or destination.is_symlink():
                raise AuditPersistenceError
            os.replace(temporary, destination)
        except AuditPersistenceError:
            raise
        except (OSError, TypeError, ValueError) as error:
            raise AuditPersistenceError from error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _prepare_root(self) -> None:
        if self.root.is_symlink():
            raise AuditPersistenceError
        self.root.mkdir(parents=True, exist_ok=True)
        info = self.root.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise AuditPersistenceError
        self._contained("")

    def _contained(self, name: str) -> Path:
        resolved_root = self.root.resolve(strict=True)
        candidate = self.root / name
        resolved = candidate.parent.resolve(strict=True) / candidate.name
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise AuditPersistenceError
        return candidate
