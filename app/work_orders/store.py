"""Atomic, path-contained work-order persistence with integrity checks."""

import os
import re
import stat
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from app.work_orders.errors import (
    WorkOrderIntegrityError,
    WorkOrderNotFoundError,
    WorkOrderPersistenceError,
    WorkOrderResourceLimitError,
)
from app.work_orders.models import WorkOrderDraft, compute_content_sha256

WORK_ORDER_ID_PATTERN = re.compile(r"wo_[0-9a-f]{32}")
MAX_WORK_ORDER_FILE_BYTES = 1_048_576
MAX_STORED_WORK_ORDERS = 10_000


def new_work_order_id() -> str:
    return f"wo_{uuid4().hex}"


class WorkOrderStore:
    """Persist one JSON file per draft inside a single runtime directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def create(self, draft: WorkOrderDraft) -> None:
        """Write a new draft; refuse to overwrite an existing one."""
        self._prepare_root()
        if len(self._ids()) >= MAX_STORED_WORK_ORDERS:
            raise WorkOrderResourceLimitError
        self._write(draft, replace=False)

    def replace(self, draft: WorkOrderDraft) -> None:
        """Atomically replace an existing draft (used for the review step)."""
        self._prepare_root()
        self._write(draft, replace=True)

    def read(self, work_order_id: str) -> WorkOrderDraft:
        if not WORK_ORDER_ID_PATTERN.fullmatch(work_order_id):
            raise WorkOrderNotFoundError
        try:
            self._prepare_root()
            path = self._contained(f"{work_order_id}.json")
            info = path.lstat()
        except FileNotFoundError as error:
            raise WorkOrderNotFoundError from error
        except OSError as error:
            raise WorkOrderPersistenceError from error
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_WORK_ORDER_FILE_BYTES:
            raise WorkOrderIntegrityError
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as handle:
                payload = handle.read(MAX_WORK_ORDER_FILE_BYTES + 1)
        except FileNotFoundError as error:
            raise WorkOrderNotFoundError from error
        except OSError as error:
            raise WorkOrderPersistenceError from error
        if len(payload) > MAX_WORK_ORDER_FILE_BYTES:
            raise WorkOrderIntegrityError
        try:
            draft = WorkOrderDraft.model_validate_json(payload)
        except ValidationError as error:
            raise WorkOrderIntegrityError from error
        if (
            draft.work_order_id != work_order_id
            or draft.content_sha256 != compute_content_sha256(draft)
            or (draft.status == "draft") != (draft.review is None)
        ):
            raise WorkOrderIntegrityError
        return draft

    def list_all(self) -> list[WorkOrderDraft]:
        """Return every stored draft, newest first."""
        try:
            self._prepare_root()
        except OSError as error:
            raise WorkOrderPersistenceError from error
        drafts = [self.read(work_order_id) for work_order_id in self._ids()]
        drafts.sort(
            key=lambda item: (item.created_at, item.work_order_id), reverse=True
        )
        return drafts

    def _ids(self) -> list[str]:
        try:
            names = sorted(entry.name for entry in os.scandir(self.root))
        except OSError as error:
            raise WorkOrderPersistenceError from error
        return [
            name[: -len(".json")]
            for name in names
            if name.endswith(".json")
            and WORK_ORDER_ID_PATTERN.fullmatch(name[: -len(".json")])
        ]

    def _write(self, draft: WorkOrderDraft, *, replace: bool) -> None:
        temporary: Path | None = None
        try:
            destination = self._contained(f"{draft.work_order_id}.json")
            exists = destination.exists() or destination.is_symlink()
            if exists == (not replace):
                raise WorkOrderPersistenceError
            temporary = self._contained(f".{draft.work_order_id}.{uuid4().hex}.tmp")
            payload = (draft.model_dump_json() + "\n").encode("utf-8")
            if len(payload) > MAX_WORK_ORDER_FILE_BYTES:
                raise WorkOrderResourceLimitError
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(temporary, flags, 0o600)
            try:
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    if written <= 0:
                        raise WorkOrderPersistenceError
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, destination)
        except (WorkOrderPersistenceError, WorkOrderResourceLimitError):
            raise
        except (OSError, TypeError, ValueError) as error:
            raise WorkOrderPersistenceError from error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def _prepare_root(self) -> None:
        try:
            if self.root.is_symlink():
                raise WorkOrderPersistenceError
            self.root.mkdir(parents=True, exist_ok=True)
            info = self.root.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise WorkOrderPersistenceError
            self._contained("")
        except OSError as error:
            raise WorkOrderPersistenceError from error

    def _contained(self, name: str) -> Path:
        resolved_root = self.root.resolve(strict=True)
        candidate = self.root / name
        resolved = candidate.parent.resolve(strict=True) / candidate.name
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise WorkOrderPersistenceError
        return candidate
