"""Credential-blind, bounded mock AeroDocs upstream for Stage 8B-0."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

from app.security.bearer import direct_bearer_token
from app.security.models import InvalidCredentialsError

MAX_BODY_BYTES: Final = 64 * 1024 * 1024
MAX_PATH_BYTES: Final = 2048
MAX_WORKERS: Final = 16
READ_TIMEOUT_SECONDS: Final = 5
LONG_RESPONSE_SECONDS: Final = 45
GATEWAY_HEADER: Final = "x-aerodocs-gateway-access-token"
OBSERVED_HEADERS: Final = frozenset(
    {
        "authorization",
        "cookie",
        GATEWAY_HEADER,
        "x-auth-request-access-token",
        "x-forwarded-access-token",
        "x-forwarded-basic-auth",
        "x-forwarded-email",
        "x-forwarded-groups",
        "x-forwarded-preferred-username",
        "x-forwarded-user",
        "x_aerodocs_gateway_access_token",
        "x_auth_request_access_token",
        "x_forwarded_access_token",
        "x_forwarded_email",
        "x_forwarded_groups",
        "x_forwarded_preferred_username",
        "x_forwarded_user",
    }
)


class ObservationStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._invocations = 0
        self._last: dict[str, object] = {}
        self._expected_gateway_token: str | None = None

    def reset(self, expected_gateway_token: str | None = None) -> None:
        with self._lock:
            self._invocations = 0
            self._last = {}
            self._expected_gateway_token = expected_gateway_token

    def record(self, observation: dict[str, object]) -> None:
        with self._lock:
            self._invocations += 1
            self._last = observation

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {"invocations": self._invocations, "last": dict(self._last)}

    def token_matches(self, values: list[str]) -> bool:
        with self._lock:
            expected = self._expected_gateway_token
        return expected is not None and values == [expected]


STORE = ObservationStore()


def parser_verdict(raw_headers: list[tuple[bytes, bytes]]) -> str:
    try:
        direct_bearer_token({"type": "http", "headers": raw_headers})
    except InvalidCredentialsError:
        return "REJECTED"
    return "ACCEPTED"


def build_metadata(
    body_length: int,
    raw_headers: list[tuple[bytes, bytes]],
    expected_gateway_match: bool = False,
) -> dict[str, object]:
    authorization = [
        value for name, value in raw_headers if name.lower() == b"authorization"
    ]
    gateway = [
        value for name, value in raw_headers if name.lower() == GATEWAY_HEADER.encode()
    ]
    cookie = [value for name, value in raw_headers if name.lower() == b"cookie"]
    hostile = sorted(
        {
            name.decode("ascii", "replace").lower()
            for name, _ in raw_headers
            if name.decode("ascii", "replace").lower() in OBSERVED_HEADERS
            and name.lower() not in {b"authorization", GATEWAY_HEADER.encode()}
        }
    )
    return {
        "authorization_field_count": len(authorization),
        "authorization_field_lengths": [len(value) for value in authorization],
        "body_bytes": body_length,
        "browser_cookie_field_count": len(cookie),
        "gateway_header_count": len(gateway),
        "gateway_token_bytes": sum(len(value) for value in gateway),
        "gateway_token_matches_expected": expected_gateway_match,
        "hostile_header_names": hostile,
        "phase8a_parser_verdict": parser_verdict(raw_headers),
    }


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = MAX_WORKERS

    def __init__(
        self, server_address: tuple[str, int], handler: type[BaseHTTPRequestHandler]
    ):
        self._slots = threading.BoundedSemaphore(MAX_WORKERS)
        super().__init__(server_address, handler)

    def process_request(
        self, request: socket.socket, client_address: tuple[str, int]
    ) -> None:
        if not self._slots.acquire(blocking=False):
            request.close()
            return
        super().process_request(request, client_address)

    def process_request_thread(
        self, request: socket.socket, client_address: tuple[str, int]
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class MockHandler(BaseHTTPRequestHandler):
    server_version = "AeroDocsSpikeMock/2"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(READ_TIMEOUT_SECONDS)

    def _send(self, status: HTTPStatus, body: bytes, *, head: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _reject(self, status: HTTPStatus) -> None:
        self._send(status, b'{"result":"REJECTED"}', head=self.command == "HEAD")

    def _body_length(self) -> int | None:
        transfer = self.headers.get_all("Transfer-Encoding") or []
        lengths = self.headers.get_all("Content-Length") or []
        if transfer or len(set(lengths)) > 1 or len(lengths) > 1:
            return None
        if not lengths:
            return 0
        try:
            length = int(lengths[0], 10)
        except ValueError:
            return None
        return length if 0 <= length <= MAX_BODY_BYTES else None

    def _consume(self, length: int) -> bool:
        remaining = length
        while remaining:
            chunk = self.rfile.read(min(remaining, 64 * 1024))
            if not chunk:
                return False
            remaining -= len(chunk)
        return True

    def _raw_headers(self) -> list[tuple[bytes, bytes]]:
        return [
            (name.encode("ascii", "replace"), value.encode("latin-1", "replace"))
            for name, value in self.headers.raw_items()
        ]

    def _handle(self) -> None:
        if len(self.path.encode("utf-8", "replace")) > MAX_PATH_BYTES:
            self._reject(HTTPStatus.REQUEST_URI_TOO_LONG)
            return
        if self.path == "/__control/reset":
            if self.client_address[0] not in {"127.0.0.1", "::1"}:
                self._reject(HTTPStatus.FORBIDDEN)
                return
            expected = self.headers.get("X-Spike-Expected-Gateway-Token")
            STORE.reset(expected)
            self._send(HTTPStatus.NO_CONTENT, b"", head=self.command == "HEAD")
            return
        if self.path == "/__control/snapshot":
            if self.client_address[0] not in {"127.0.0.1", "::1"}:
                self._reject(HTTPStatus.FORBIDDEN)
                return
            body = json.dumps(
                STORE.snapshot(), sort_keys=True, separators=(",", ":")
            ).encode()
            self._send(HTTPStatus.OK, body, head=self.command == "HEAD")
            return
        route_path = urllib.parse.urlsplit(self.path).path
        allowed = {
            "/__mock/health",
            "/documents",
            "/indexes/active",
            "/indexes/rebuild",
            "/search",
            "/answers",
            "/ui/api/index-status",
            "/ui/",
            "/ui/documents",
            "/ui/search",
            "/ui/ask",
        }
        if route_path not in allowed and not route_path.startswith("/documents/"):
            self._reject(HTTPStatus.NOT_FOUND)
            return
        length = self._body_length()
        if length is None:
            self._reject(HTTPStatus.BAD_REQUEST)
            return
        if not self._consume(length):
            self.close_connection = True
            return
        if self.headers.get("X-Spike-Long-Response") == "45":
            deadline = time.monotonic() + LONG_RESPONSE_SECONDS
            while (remaining := deadline - time.monotonic()) > 0:
                time.sleep(min(remaining, 0.25))
        raw_headers = self._raw_headers()
        gateway = [
            value.decode("latin-1")
            for name, value in raw_headers
            if name.lower() == GATEWAY_HEADER.encode()
        ]
        observation = build_metadata(length, raw_headers, STORE.token_matches(gateway))
        observation["method"] = self.command
        observation["method_is_head"] = self.command == "HEAD"
        observation["route_id"] = {
            "/__mock/health": "health",
            "/documents": "documents",
            "/indexes/active": "index_active",
            "/indexes/rebuild": "index_rebuild",
            "/search": "search",
            "/answers": "answers",
            "/ui/api/index-status": "ui_index_status",
            "/ui/": "ui_home",
            "/ui/documents": "ui_documents",
            "/ui/search": "ui_search",
            "/ui/ask": "ui_ask",
        }.get(route_path, "document_resource")
        STORE.record(observation)
        body = b'{"mock":"AERODOCS_RESPONSE_SENTINEL"}'
        self._send(HTTPStatus.OK, body, head=self.command == "HEAD")

    do_DELETE = _handle
    do_GET = _handle
    do_HEAD = _handle
    do_PATCH = _handle
    do_POST = _handle
    do_PUT = _handle

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18000)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise SystemExit("port must be between 1024 and 65535")
    BoundedThreadingHTTPServer(("127.0.0.1", args.port), MockHandler).serve_forever()


if __name__ == "__main__":
    main()
