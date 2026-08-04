#!/usr/bin/env python3
"""Fail-closed HTTP/1.1 and HTTP/2 Authorization-preservation probe."""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

HOST: Final = "aerodocs-spike.invalid"
PATH: Final = "/search"
MAX_RESPONSE_BYTES: Final = 64 * 1024


@dataclass(frozen=True)
class Vector:
    test_id: str
    fields: tuple[bytes, ...]
    valid_control: bool = False


VECTORS: Final = (
    Vector("valid_control", (b"Bearer aaa.bbb.ccc",), True),
    Vector("different_fields", (b"Bearer aaa.bbb.ccc", b"Bearer ddd.eee.fff")),
    Vector("identical_fields", (b"Bearer aaa.bbb.ccc", b"Bearer aaa.bbb.ccc")),
    Vector("comma_combined", (b"Bearer aaa.bbb.ccc, Bearer ddd.eee.fff",)),
    Vector("repeated_scheme", (b"Bearer Bearer aaa.bbb.ccc",)),
    Vector("repeated_space", (b"Bearer  aaa.bbb.ccc",)),
    Vector("tab_separator", (b"Bearer\taaa.bbb.ccc",)),
    Vector("empty_field", (b"",)),
)


class ProbeFailure(RuntimeError):
    pass


def _approved_url(value: str, *, public: bool) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(value)
    allowed = {"aerodocs-spike.invalid"} if public else {"127.0.0.1", "::1"}
    expected_scheme = "https" if public else "http"
    if (
        parsed.scheme != expected_scheme
        or parsed.hostname not in allowed
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.username
    ):
        raise ProbeFailure("invalid_endpoint")
    return parsed


def _control(base_url: str, path: str) -> dict[str, object]:
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=3) as response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
    if len(body) > MAX_RESPONSE_BYTES:
        raise ProbeFailure("inconclusive")
    return json.loads(body) if body else {}


def _reset(mock_url: str, auth_url: str) -> None:
    for base, path in ((mock_url, "/__control/reset"), (auth_url, "/__spike/reset")):
        request = urllib.request.Request(base.rstrip("/") + path, method="POST")
        with urllib.request.urlopen(request, timeout=3):
            pass


def _http1(context: ssl.SSLContext, port: int, vector: Vector) -> int:
    request = (
        b"\r\n".join(
            [
                b"GET " + PATH.encode() + b" HTTP/1.1",
                b"Host: " + HOST.encode(),
                b"Connection: close",
            ]
            + [b"Authorization: " + value for value in vector.fields]
        )
        + b"\r\n\r\n"
    )
    with socket.create_connection((HOST, port), timeout=5) as raw:
        with context.wrap_socket(raw, server_hostname=HOST) as tls:
            tls.sendall(request)
            response = tls.recv(MAX_RESPONSE_BYTES + 1)
    if len(response) > MAX_RESPONSE_BYTES or b"\r\n" not in response:
        raise ProbeFailure("inconclusive")
    return int(response.split(b"\r\n", 1)[0].split(b" ", 2)[1])


def parse_http2_trace(trace: bytes) -> int:
    count = 0
    in_send_header = False
    for raw_line in trace.splitlines():
        if raw_line.startswith(b"=> Send header"):
            in_send_header = True
            continue
        if raw_line.startswith((b"<= ", b"== ", b"=> Send data")):
            in_send_header = False
        if not in_send_header or b":" not in raw_line:
            continue
        offset, content = raw_line.split(b":", 1)
        if (
            len(offset) == 4
            and all(byte in b"0123456789abcdefABCDEF" for byte in offset)
            and content.lstrip().lower().startswith(b"authorization:")
        ):
            count += 1
    return count


def _http2(
    cafile: Path, port: int, vector: Vector, private_runtime: Path
) -> tuple[int, int]:
    with tempfile.TemporaryDirectory(prefix="http2-trace-", dir=private_runtime) as tmp:
        work = Path(tmp)
        trace_file = work / "trace"
        body_file = work / "body"
        status_file = work / "status"
        stderr_file = work / "stderr"
        for path in (trace_file, body_file, status_file, stderr_file):
            path.touch(mode=0o600)
        args = [
            "curl",
            "--silent",
            "--show-error",
            "--http2",
            "--cacert",
            str(cafile),
            "--trace-ascii",
            str(trace_file),
            "--output",
            str(body_file),
            "--write-out",
            "%{http_code}",
        ]
        for value in vector.fields:
            args.extend(["--header", "Authorization: " + value.decode("ascii")])
        args.append(f"https://{HOST}:{port}{PATH}")
        with (
            status_file.open("wb") as status_out,
            stderr_file.open("wb") as diagnostics,
        ):
            completed = subprocess.run(
                args, stdout=status_out, stderr=diagnostics, check=False, timeout=10
            )
        trace = trace_file.read_bytes()
        sent = parse_http2_trace(trace)
        status_text = status_file.read_text(encoding="ascii").strip()
        if (
            completed.returncode
            or sent != len(vector.fields)
            or not status_text.isdigit()
        ):
            raise ProbeFailure("http2_unavailable")
        return int(status_text), sent


def _evaluate(
    protocol: str,
    vector: Vector,
    status: int,
    before_mock: int,
    after_mock: dict[str, object],
    before_auth: int,
    after_auth: dict[str, object],
) -> dict[str, object]:
    mock_delta = int(after_mock["invocations"]) - before_mock
    auth_delta = int(after_auth["invocations"]) - before_auth
    last = after_mock.get("last", {}) if mock_delta else {}
    if not isinstance(last, dict):
        raise ProbeFailure("inconclusive")
    parser = last.get("phase8a_parser_verdict", "NOT_OBSERVED")
    downstream_count = int(last.get("authorization_field_count", 0))
    downstream_lengths = last.get("authorization_field_lengths", [])
    rejected_before_both = mock_delta == 0 and auth_delta == 0
    direct = mock_delta == 1 and auth_delta == 0
    safe = rejected_before_both or (
        direct
        and (
            (vector.valid_control and parser == "ACCEPTED")
            or (not vector.valid_control and parser == "REJECTED")
        )
    )
    if not safe:
        raise ProbeFailure("authorization_ambiguous")
    return {
        "protocol": protocol,
        "sent_field_count": len(vector.fields),
        "sent_field_lengths": [len(value) for value in vector.fields],
        "caddy_status": status,
        "oauth_invocation_delta": auth_delta,
        "mock_invocation_delta": mock_delta,
        "downstream_field_count": downstream_count,
        "downstream_field_lengths": downstream_lengths,
        "direct_branch": direct,
        "browser_branch": auth_delta > 0,
        "phase8a_parser_accepted": parser == "ACCEPTED",
        "safe": safe,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ca-file", type=Path, required=True)
    parser.add_argument("--caddy-url", required=True)
    parser.add_argument("--mock-control-url", required=True)
    parser.add_argument("--auth-control-url", required=True)
    parser.add_argument("--protocol", choices=("both",), required=True)
    parser.add_argument("--phase8a-repository", type=Path, required=True)
    parser.add_argument("--private-runtime", type=Path, required=True)
    args = parser.parse_args()
    public = _approved_url(args.caddy_url, public=True)
    _approved_url(args.mock_control_url, public=False)
    _approved_url(args.auth_control_url, public=False)
    expected_repository = Path(__file__).resolve().parents[3]
    if (
        args.phase8a_repository.resolve() != expected_repository
        or not (expected_repository / "app/security/bearer.py").is_file()
    ):
        raise SystemExit(2)
    port = public.port or 443
    context = ssl.create_default_context(cafile=str(args.ca_file))
    private_runtime = args.private_runtime.resolve()
    if not private_runtime.is_dir() or private_runtime.is_symlink():
        raise SystemExit(2)
    results: dict[str, object] = {}
    try:
        for protocol in ("http1", "http2"):
            for vector in VECTORS:
                _reset(args.mock_control_url, args.auth_control_url)
                before_mock = int(
                    _control(args.mock_control_url, "/__control/snapshot")[
                        "invocations"
                    ]
                )
                before_auth = int(
                    _control(args.auth_control_url, "/__spike/snapshot")["invocations"]
                )
                if protocol == "http1":
                    status = _http1(context, port, vector)
                    transmitted = len(vector.fields)
                else:
                    status, transmitted = _http2(
                        args.ca_file, port, vector, private_runtime
                    )
                if transmitted != len(vector.fields):
                    raise ProbeFailure("transmitted_field_count")
                results[f"{protocol}_{vector.test_id}"] = _evaluate(
                    protocol,
                    vector,
                    status,
                    before_mock,
                    _control(args.mock_control_url, "/__control/snapshot"),
                    before_auth,
                    _control(args.auth_control_url, "/__spike/snapshot"),
                )
                results[f"{protocol}_{vector.test_id}"][
                    "observed_transmitted_field_count"
                ] = transmitted
    except (
        KeyError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        ProbeFailure,
    ):
        print(
            json.dumps(
                {
                    "schema_version": 2,
                    "test_id": "duplicate_authorization",
                    "observed_at": datetime.now(UTC)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "result": "FAIL",
                    "measurements": {"checked": True},
                    "failure_code": "AUTHORIZATION",
                    "gate_results": {
                        "duplicate_authorization_http1": {
                            "result": "FAIL",
                            "measurements": {"checked": True},
                        }
                    },
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None
    measurements = {
        "case_count": len(results),
        "http1_case_count": sum(key.startswith("http1_") for key in results),
        "http2_case_count": sum(key.startswith("http2_") for key in results),
        "all_safe": all(
            isinstance(item, dict) and item.get("safe") is True
            for item in results.values()
        ),
    }
    print(
        json.dumps(
            {
                "schema_version": 2,
                "test_id": "duplicate_authorization",
                "observed_at": datetime.now(UTC)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "result": "PASS",
                "measurements": measurements,
                "failure_code": "NONE",
                "gate_results": {
                    gate: {"result": "PASS", "measurements": {"case_count": 8}}
                    for gate in (
                        "duplicate_authorization_http1",
                        "duplicate_authorization_http2",
                    )
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
