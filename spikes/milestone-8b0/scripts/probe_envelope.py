#!/usr/bin/env python3
"""Exercise stock-Caddy auth-envelope behavior at each approved limit."""

from __future__ import annotations

import argparse
import http.client
import json
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

HOST: Final = "aerodocs-spike.invalid"
MOCK_CONTROL = "http://127.0.0.1:18000"
AUTH_CONTROL = "http://127.0.0.1:14180"


@dataclass(frozen=True)
class Case:
    name: str
    should_pass: bool
    expected_set_cookie: int | None = None


CASES: Final = (
    Case("baseline", True, 0),
    Case("missing-token", False),
    Case("empty-token", False),
    Case("identical-token-fields", False),
    Case("different-token-fields", False),
    Case("comma-token", False),
    Case("leading-space-token", False),
    Case("trailing-space-token", False),
    Case("control-token", False),
    Case("non-ascii-token", False),
    Case("one-segment-token", False),
    Case("two-segment-token", False),
    Case("empty-segment-token", False),
    Case("set-cookie-count-boundary", True, 4),
    Case("set-cookie-count-over", False),
    Case("set-cookie-field-boundary", True, 1),
    Case("set-cookie-field-over", False),
    Case("active-parts-boundary", True, 3),
    Case("active-parts-over", False),
    Case("active-values-boundary", True, 3),
    Case("active-values-over", False),
    Case("name-value-boundary", True, 1),
    Case("name-value-over", False),
    Case("gateway-token-boundary", True, 0),
    Case("gateway-token-over", False),
    Case("complete-header-boundary", True),
    Case("complete-header-over", False),
)


class ProbeFailure(RuntimeError):
    pass


ENVELOPE_GATES: Final = (
    "repeated_set_cookie",
    "gateway_token_limit",
    "active_cookie_values_limit",
    "active_cookie_parts_limit",
    "cookie_name_value_limit",
    "set_cookie_field_limit",
    "incoming_cookie_limit",
    "set_cookie_total_limit",
    "set_cookie_count_limit",
    "auth_response_header_limit",
)


def evidence(result: str, measurements: dict[str, bool | int]) -> dict[str, object]:
    state = "PASS" if result == "PASS" else "FAIL"
    return {
        "schema_version": 2,
        "test_id": "auth_envelope",
        "observed_at": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "result": result,
        "measurements": measurements,
        "failure_code": "NONE" if result == "PASS" else "COOKIE_ENVELOPE",
        "gate_results": {
            "gateway_token_limit": {
                "result": state,
                "measurements": {"checked": True},
            }
        },
    }


def _json_control(
    base: str, path: str, expected: str | None = None
) -> dict[str, object]:
    request = urllib.request.Request(
        base + path, method="POST" if "reset" in path else "GET"
    )
    if expected is not None:
        request.add_header("X-Spike-Expected-Gateway-Token", expected)
    with urllib.request.urlopen(request, timeout=3) as response:
        body = response.read(65537)
    if len(body) > 65536:
        raise ProbeFailure("control_response_too_large")
    return json.loads(body) if body else {}


def run_case(context: ssl.SSLContext, port: int, case: Case) -> dict[str, int | bool]:
    expected = "aaa.bbb.ccc"
    if case.name == "gateway-token-boundary":
        expected = "a." + "x" * (3072 - 4) + ".c"
    _json_control(MOCK_CONTROL, "/__control/reset", expected)
    _json_control(AUTH_CONTROL, "/__spike/reset")
    connection = http.client.HTTPSConnection(
        HOST, port=port, context=context, timeout=8
    )
    connection.request("GET", "/search", headers={"X-Spike-Case": case.name})
    response = connection.getresponse()
    body = response.read(64 * 1024 + 1)
    response_headers = response.getheaders()
    connection.close()
    if len(body) > 64 * 1024:
        raise ProbeFailure("response_too_large")
    cookies = [
        value for name, value in response_headers if name.lower() == "set-cookie"
    ]
    leaked_token = any(
        name.lower()
        in {"x-auth-request-access-token", "x-aerodocs-gateway-access-token"}
        for name, _ in response_headers
    )
    mock_snapshot = _json_control(MOCK_CONTROL, "/__control/snapshot")
    auth_snapshot = _json_control(AUTH_CONTROL, "/__spike/snapshot")
    invoked = mock_snapshot.get("invocations") == 1
    auth_invocations = int(auth_snapshot.get("invocations", 0))
    if leaked_token:
        raise ProbeFailure("access_token_leaked")
    # Fail-safe Outcome B: stock Caddy cannot prove auth-response field
    # cardinality, so no browser-authenticated request may reach AeroDocs.
    if response.status != 503 or invoked or cookies or auth_invocations != 1:
        raise ProbeFailure("fail_safe_browser_branch_bypassed")
    return {
        "aerodocs_invoked": False,
        "response_set_cookie_fields": 0,
        "access_token_header_absent": not leaked_token,
        "oauth_invocations": auth_invocations,
    }


def run_incoming_cookie_case(
    context: ssl.SSLContext, port: int, size: int
) -> tuple[int, int]:
    import socket

    cookie_field = complete_cookie_field(size)
    request = (
        b"GET /search HTTP/1.1\r\nHost: "
        + HOST.encode()
        + b"\r\nConnection: close\r\nX-Spike-Case: baseline\r\n"
        + cookie_field
        + b"\r\n"
    )
    with socket.create_connection((HOST, port), timeout=8) as raw:
        with context.wrap_socket(raw, server_hostname=HOST) as tls:
            tls.sendall(request)
            response = tls.recv(64 * 1024 + 1)
    if len(response) > 64 * 1024 or b"\r\n" not in response:
        raise ProbeFailure("cookie_response_invalid")
    status = int(response.split(b"\r\n", 1)[0].split(b" ", 2)[1])
    return status, len(cookie_field)


def complete_cookie_field(target_size: int) -> bytes:
    prefix = b"Cookie: unrelated="
    suffix = b"\r\n"
    if target_size < len(prefix) + len(suffix):
        raise ProbeFailure("cookie_target_too_small")
    field = prefix + b"x" * (target_size - len(prefix) - len(suffix)) + suffix
    validate_cookie_wire([field], target_size)
    return field


def validate_cookie_wire(fields: list[bytes], target_size: int) -> None:
    if len(fields) != 1:
        raise ProbeFailure("multiple_cookie_fields")
    field = fields[0]
    if len(field) != target_size:
        raise ProbeFailure("cookie_rewrite_detected")
    if not field.startswith(b"Cookie: ") or not field.endswith(b"\r\n"):
        raise ProbeFailure("cookie_generation_invalid")
    value = field[len(b"Cookie: ") : -2]
    if (
        b"\r" in value
        or b"\n" in value
        or any(byte < 0x20 or byte == 0x7F for byte in value)
    ):
        raise ProbeFailure("cookie_control")


def main() -> None:
    global MOCK_CONTROL, AUTH_CONTROL
    parser = argparse.ArgumentParser()
    parser.add_argument("--ca-file", type=Path, required=True)
    parser.add_argument("--endpoints-file", type=Path, required=True)
    args = parser.parse_args()
    endpoints = json.loads(args.endpoints_file.read_text(encoding="utf-8"))
    public = urllib.parse.urlparse(str(endpoints["caddy_url"]))
    MOCK_CONTROL = str(endpoints["mock_control_url"]).rstrip("/")
    AUTH_CONTROL = str(endpoints["oauth_control_url"]).rstrip("/")
    if (
        public.scheme != "https"
        or public.hostname != HOST
        or not MOCK_CONTROL.startswith("http://127.0.0.1:")
        or not AUTH_CONTROL.startswith("http://127.0.0.1:")
    ):
        raise SystemExit(2)
    port = public.port or 443
    context = ssl.create_default_context(cafile=str(args.ca_file))
    measurements: dict[str, bool | int] = {}
    try:
        for case in CASES:
            result = run_case(context, port, case)
            measurements[case.name + "_passed"] = True
            measurements[case.name + "_set_cookie_fields"] = int(
                result["response_set_cookie_fields"]
            )
        # A valid token is also blocked, so the reviewed stock-Caddy
        # architecture cannot pass the gateway-token-envelope capability.
        if not all(measurements[name + "_passed"] for name in {c.name for c in CASES}):
            raise ProbeFailure("case_incomplete")
        print(json.dumps(evidence("FAIL", measurements), sort_keys=True))
        return
    except (OSError, ProbeFailure):
        print(json.dumps(evidence("FAIL", {"case_count": 0}), sort_keys=True))
        raise SystemExit(1) from None
    raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
