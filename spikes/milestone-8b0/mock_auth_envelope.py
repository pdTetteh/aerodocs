"""Loopback-only synthetic auth envelope source with bounded counters."""

from __future__ import annotations

import argparse
import json
import socket
import socketserver
import threading
from typing import Final

MAX_REQUEST_BYTES: Final = 32 * 1024
STATUS_LINE: Final = b"HTTP/1.1 202 Accepted\r\n"
TAIL: Final = b"Content-Length: 0\r\nConnection: close\r\n\r\n"
TOKEN: Final = b"aaa.bbb.ccc"
SET_COOKIE_OVERHEAD: Final = len(b"Set-Cookie: \r\n")
_lock = threading.Lock()
_invocations = 0


def _cookie(name: bytes, value_bytes: int) -> bytes:
    return (
        name + b"=" + b"x" * value_bytes + b"; Path=/; Secure; HttpOnly; SameSite=Lax"
    )


def serialize(fields: list[tuple[bytes, bytes]]) -> bytes:
    response = bytearray(STATUS_LINE)
    for name, value in fields:
        response.extend(name + b": " + value + b"\r\n")
    response.extend(TAIL)
    return bytes(response)


def _set_cookie_wire(target: int, name: bytes = b"part") -> bytes:
    attributes = b"; Path=/; Secure; HttpOnly; SameSite=Lax"
    fixed = SET_COOKIE_OVERHEAD + len(name) + 1 + len(attributes)
    if target < fixed:
        raise ValueError("target_too_small")
    return _cookie(name, target - fixed)


def fields_for(case: bytes) -> list[tuple[bytes, bytes]]:
    base = [(b"X-Auth-Request-Access-Token", TOKEN), (b"Cache-Control", b"no-store")]
    if case == b"baseline":
        return base
    if case == b"missing-token":
        return [(b"Cache-Control", b"no-store")]
    if case == b"empty-token":
        return [(b"X-Auth-Request-Access-Token", b""), (b"Cache-Control", b"no-store")]
    if case == b"identical-token-fields":
        return [(b"X-Auth-Request-Access-Token", TOKEN)] * 2
    if case == b"different-token-fields":
        return [
            (b"X-Auth-Request-Access-Token", TOKEN),
            (b"X-Auth-Request-Access-Token", b"ddd.eee.fff"),
        ]
    if case == b"comma-token":
        return [(b"X-Auth-Request-Access-Token", b"aaa.bbb.ccc,ddd.eee.fff")]
    if case == b"leading-space-token":
        return [(b"X-Auth-Request-Access-Token", b" aaa.bbb.ccc")]
    if case == b"trailing-space-token":
        return [(b"X-Auth-Request-Access-Token", b"aaa.bbb.ccc ")]
    if case == b"control-token":
        return [(b"X-Auth-Request-Access-Token", b"aaa.bbb\t.ccc")]
    if case == b"non-ascii-token":
        return [(b"X-Auth-Request-Access-Token", b"aaa.b\xc3\xa9b.ccc")]
    if case == b"one-segment-token":
        return [(b"X-Auth-Request-Access-Token", b"aaabbbccc")]
    if case == b"two-segment-token":
        return [(b"X-Auth-Request-Access-Token", b"aaa.bbb")]
    if case == b"empty-segment-token":
        return [(b"X-Auth-Request-Access-Token", b"aaa..ccc")]
    if case in {b"gateway-token-boundary", b"gateway-token-over"}:
        size = 3072 + int(case.endswith(b"over"))
        return [
            (b"X-Auth-Request-Access-Token", b"a." + b"x" * (size - 4) + b".c"),
            (b"Cache-Control", b"no-store"),
        ]
    if case in {b"set-cookie-count-boundary", b"set-cookie-count-over"}:
        count = 4 + (case.endswith(b"over"))
        return base + [
            (b"Set-Cookie", _cookie(b"part" + str(i).encode(), 8)) for i in range(count)
        ]
    if case in {b"set-cookie-field-boundary", b"set-cookie-field-over"}:
        return base + [
            (
                b"Set-Cookie",
                _set_cookie_wire(4608 + (case.endswith(b"over")), b"unrelated"),
            )
        ]
    if case in {b"set-cookie-total-boundary", b"set-cookie-total-over"}:
        total = 18432 + (case.endswith(b"over"))
        sizes = [4608, 4608, 4608, total - 13824]
        return base + [
            (b"Set-Cookie", _set_cookie_wire(size, b"unrelated" + str(i).encode()))
            for i, size in enumerate(sizes)
        ]
    if case in {b"active-parts-boundary", b"active-parts-over"}:
        count = 3 + (case.endswith(b"over"))
        return base + [
            (b"Set-Cookie", _cookie(b"__Host-aerodocs_spike_" + str(i).encode(), 8))
            for i in range(count)
        ]
    if case in {b"active-values-boundary", b"active-values-over"}:
        lengths = [3414, 3413, 3413 + (case.endswith(b"over"))]
        return base + [
            (b"Set-Cookie", _cookie(b"__Host-aerodocs_spike_" + str(i).encode(), size))
            for i, size in enumerate(lengths)
        ]
    if case in {b"name-value-boundary", b"name-value-over"}:
        name = b"__Host-aerodocs_spike"
        return base + [
            (b"Set-Cookie", _cookie(name, 4096 - len(name) + (case.endswith(b"over"))))
        ]
    if case in {b"complete-header-boundary", b"complete-header-over"}:
        target = 24576 + (case.endswith(b"over"))
        empty = base + [(b"X-Spike-Padding", b"")]
        return base + [(b"X-Spike-Padding", b"x" * (target - len(serialize(empty))))]
    raise ValueError("unknown_case")


class Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        global _invocations
        self.connection.settimeout(5)
        request = bytearray()
        try:
            while True:
                line = self.rfile.readline(MAX_REQUEST_BYTES + 1 - len(request))
                if not line or len(request) + len(line) > MAX_REQUEST_BYTES:
                    return
                request.extend(line)
                if line == b"\r\n":
                    break
        except TimeoutError:
            return
        case = b"baseline"
        request_line = bytes(request).split(b"\r\n", 1)[0].split(b" ")
        if len(request_line) != 3:
            return
        path = request_line[1]
        if path == b"/__spike/snapshot":
            with _lock:
                body = json.dumps({"invocations": _invocations}).encode()
            self.wfile.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + body
            )
            return
        if path == b"/__spike/reset":
            with _lock:
                _invocations = 0
            self.wfile.write(
                b"HTTP/1.1 204 No Content\r\n"
                b"Content-Length: 0\r\nConnection: close\r\n\r\n"
            )
            return
        for line in bytes(request).split(b"\r\n"):
            if line.lower().startswith(b"x-spike-case:"):
                case = line.split(b":", 1)[1].strip()
        with _lock:
            _invocations += 1
        try:
            self.wfile.write(serialize(fields_for(case)))
        except ValueError:
            self.wfile.write(
                b"HTTP/1.1 400 Bad Request\r\n"
                b"Content-Length: 0\r\nConnection: close\r\n\r\n"
            )


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, server_address: tuple[str, int], handler: type[Handler]):
        self._slots = threading.BoundedSemaphore(16)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise SystemExit(2)
    with Server(("127.0.0.1", args.port), Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
