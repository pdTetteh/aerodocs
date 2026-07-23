"""Tests for request-body enforcement before multipart parsing."""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient
from starlette.types import Message, Scope

from app.core.config import Settings
from app.core.request_limits import RequestBodyLimitMiddleware
from app.factory import create_app
from app.ingestion.service import DocumentService


async def run_limited_request(
    *,
    maximum_bytes: int,
    bodies: list[bytes],
    content_length: int | None = None,
) -> tuple[bool, int, int]:
    """Run a minimal ASGI request and return call, receive, and status state."""
    called = False
    receive_calls = 0
    status_code = 0
    messages = [
        {
            "type": "http.request",
            "body": body,
            "more_body": index < len(bodies) - 1,
        }
        for index, body in enumerate(bodies)
    ]

    async def downstream(
        _scope: Scope,
        receive: Callable[[], Awaitable[Message]],
        send: Callable[[Message], Awaitable[None]],
    ) -> None:
        nonlocal called
        called = True
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive() -> Message:
        nonlocal receive_calls
        receive_calls += 1
        return cast(Message, messages.pop(0))

    async def send(message: Message) -> None:
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]

    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    scope = cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/documents",
            "raw_path": b"/documents",
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "client": ("test", 1),
            "server": ("test", 80),
        },
    )
    middleware = RequestBodyLimitMiddleware(downstream, maximum_bytes)
    await middleware(scope, receive, send)
    return called, receive_calls, status_code


@pytest.mark.asyncio
async def test_request_body_exact_boundary_is_permitted() -> None:
    called, receive_calls, status = await run_limited_request(
        maximum_bytes=5,
        bodies=[b"abc", b"de"],
    )

    assert called is True
    assert receive_calls == 2
    assert status == 204


@pytest.mark.asyncio
async def test_request_body_one_byte_over_is_rejected_while_streaming() -> None:
    _called, receive_calls, status = await run_limited_request(
        maximum_bytes=5,
        bodies=[b"abc", b"def"],
    )

    assert receive_calls == 2
    assert status == 413


@pytest.mark.asyncio
async def test_content_length_rejects_before_reading_or_calling_app() -> None:
    called, receive_calls, status = await run_limited_request(
        maximum_bytes=5,
        bodies=[b"ignored"],
        content_length=6,
    )

    assert called is False
    assert receive_calls == 0
    assert status == 413


@pytest.mark.asyncio
async def test_limit_after_response_start_never_sends_second_response() -> None:
    sent: list[Message] = []

    async def downstream(
        _scope: Scope,
        receive: Callable[[], Awaitable[Message]],
        send: Callable[[Message], Awaitable[None]],
    ) -> None:
        await send({"type": "http.response.start", "status": 202, "headers": []})
        await receive()

    async def receive() -> Message:
        return {
            "type": "http.request",
            "body": b"limit-plus-one",
            "more_body": False,
        }

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = RequestBodyLimitMiddleware(downstream, maximum_bytes=5)
    await middleware(_http_scope(), receive, send)

    response_starts = [
        message for message in sent if message["type"] == "http.response.start"
    ]
    assert len(response_starts) == 1
    assert response_starts[0]["status"] == 202


@pytest.mark.asyncio
async def test_disconnect_passes_through_without_false_413() -> None:
    received: Message | None = None
    sent: list[Message] = []

    async def downstream(
        _scope: Scope,
        receive: Callable[[], Awaitable[Message]],
        _send: Callable[[Message], Awaitable[None]],
    ) -> None:
        nonlocal received
        received = await receive()

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = RequestBodyLimitMiddleware(downstream, maximum_bytes=5)
    await middleware(_http_scope(), receive, send)

    assert received == {"type": "http.disconnect"}
    assert sent == []


@pytest.mark.asyncio
async def test_non_http_scope_passes_through_unchanged() -> None:
    observed_scope: Scope | None = None
    websocket_scope = cast(
        Scope,
        {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "scheme": "ws",
            "path": "/socket",
            "raw_path": b"/socket",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("test", 1),
            "server": ("test", 80),
            "subprotocols": [],
        },
    )

    async def downstream(
        scope: Scope,
        _receive: Callable[[], Awaitable[Message]],
        _send: Callable[[Message], Awaitable[None]],
    ) -> None:
        nonlocal observed_scope
        observed_scope = scope

    async def receive() -> Message:
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(_message: Message) -> None:
        return None

    middleware = RequestBodyLimitMiddleware(downstream, maximum_bytes=5)
    await middleware(websocket_scope, receive, send)

    assert observed_scope is websocket_scope


def test_middleware_rejection_never_calls_document_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        document_storage_root=tmp_path / "runtime",
        max_pdf_upload_bytes=100,
        max_request_body_bytes=200,
    )  # type: ignore[call-arg]
    ingest_called = False

    def unexpected_ingest(*_args: object, **_kwargs: object) -> None:
        nonlocal ingest_called
        ingest_called = True

    monkeypatch.setattr(DocumentService, "ingest", unexpected_ingest)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/documents",
            files={"file": ("large.pdf", b"x" * 500, "application/pdf")},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_body_too_large"
    assert ingest_called is False
    assert list((tmp_path / "runtime" / ".staging").iterdir()) == []
    assert list((tmp_path / "runtime" / "documents").iterdir()) == []


def _http_scope() -> Scope:
    return cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/documents",
            "raw_path": b"/documents",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("test", 1),
            "server": ("test", 80),
        },
    )
