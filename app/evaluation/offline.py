"""Offline isolation and production-pinned real-provider policy."""

from __future__ import annotations

import asyncio
import socket
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from app.core.config import (
    PINNED_EMBEDDING_MODEL_NAME,
    PINNED_EMBEDDING_MODEL_REVISION,
    PINNED_GENERATION_MODEL_NAME,
    PINNED_GENERATION_MODEL_REVISION,
    Settings,
)


class NetworkDeniedError(RuntimeError):
    """Raised when an offline evaluation attempts network access."""


def deny_network(*_args: object, **_kwargs: object) -> None:
    raise NetworkDeniedError("network_access_denied")


@contextmanager
def network_denial_guard() -> Iterator[None]:
    """Deny socket, HTTP, hub, transformer, and sentence-transformer resolution."""
    targets: list[Any] = [
        patch.object(socket, "create_connection", deny_network),
        patch.object(socket.socket, "connect", deny_network),
        patch.object(socket.socket, "connect_ex", deny_network),
        patch.object(asyncio.BaseEventLoop, "create_connection", deny_network),
        patch("urllib.request.urlopen", deny_network),
        patch("httpx.Client.request", deny_network),
        patch("httpx.AsyncClient.request", deny_network),
    ]
    optional = (
        ("huggingface_hub.file_download", "hf_hub_download"),
        ("huggingface_hub", "snapshot_download"),
        ("transformers.utils.hub", "cached_file"),
        ("sentence_transformers", "SentenceTransformer"),
    )
    targets.extend(
        patch.object(module, attribute, deny_network)
        for module_name, attribute in optional
        if (module := sys.modules.get(module_name)) is not None
        and hasattr(module, attribute)
    )
    started: list[Any] = []
    try:
        for item in targets:
            item.start()
            started.append(item)
        yield
    finally:
        for item in reversed(started):
            item.stop()


def real_mode_settings(
    settings: Settings,
    *,
    allow_model_downloads: bool,
) -> Settings:
    """Validate and return the exact production model policy for real evaluation."""
    if (
        settings.embedding_model_name != PINNED_EMBEDDING_MODEL_NAME
        or settings.embedding_model_revision != PINNED_EMBEDDING_MODEL_REVISION
        or settings.generation_model_name != PINNED_GENERATION_MODEL_NAME
        or settings.generation_model_revision != PINNED_GENERATION_MODEL_REVISION
    ):
        raise ValueError("evaluation_real_model_identity_mismatch")
    return settings
