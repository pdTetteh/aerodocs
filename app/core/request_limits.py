"""Pure ASGI request-body limiting before multipart parsing."""

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyTooLarge(OSError):
    """Signal an oversized request while allowing multipart spool cleanup."""


class RequestBodyLimitMiddleware:
    """Reject oversized HTTP request bodies before framework parsing."""

    def __init__(self, app: ASGIApp, maximum_bytes: int) -> None:
        self.app = app
        self.maximum_bytes = maximum_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = _content_length(scope)
        if content_length is not None and content_length > self.maximum_bytes:
            await _send_too_large(send)
            return

        received_bytes = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.maximum_bytes:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if not response_started:
                await _send_too_large(send)


def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name.lower() == b"content-length":
            try:
                parsed = int(value)
            except ValueError:
                return None
            return parsed if parsed >= 0 else None
    return None


async def _send_too_large(send: Send) -> None:
    payload = json.dumps(
        {
            "error": {
                "code": "request_body_too_large",
                "message": "The request body exceeds the configured size limit.",
            }
        },
        separators=(",", ":"),
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})
