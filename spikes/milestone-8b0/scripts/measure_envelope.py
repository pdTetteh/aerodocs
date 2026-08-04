#!/usr/bin/env python3
"""Strict raw-wire authentication envelope measurement."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Final

LIMITS: Final = {
    "gateway_token_bytes": 3072,
    "active_cookie_value_bytes": 10240,
    "active_cookie_parts": 3,
    "cookie_name_value_bytes": 4096,
    "set_cookie_field_bytes": 4608,
    "incoming_cookie_bytes": 16384,
    "set_cookie_total_bytes": 18432,
    "set_cookie_fields": 4,
    "auth_response_header_bytes": 24576,
}
STATUS_LINE = re.compile(rb"HTTP/1\.[01] [1-5][0-9]{2} [\x20-\x7e]{1,64}\r\n")
FIELD_NAME = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}")


class EnvelopeError(ValueError):
    """A deterministic, non-disclosing envelope rejection."""


def _session_cookie_name(name: bytes) -> bool:
    return name == b"__Host-aerodocs_spike" or name.startswith(
        b"__Host-aerodocs_spike_"
    )


def _cookie_pair(value: bytes) -> tuple[bytes, bytes]:
    pair = value.split(b";", 1)[0]
    if b"=" not in pair:
        raise EnvelopeError("malformed_cookie")
    name, cookie_value = pair.split(b"=", 1)
    if not name or not FIELD_NAME.fullmatch(name):
        raise EnvelopeError("malformed_cookie")
    if any(byte < 0x21 or byte == 0x7F for byte in cookie_value):
        raise EnvelopeError("malformed_cookie")
    return name, cookie_value


def parse_response_header_block(raw: bytes) -> list[tuple[bytes, bytes, int]]:
    """Parse one complete HTTP/1.x response header block as wire octets."""

    if len(raw) > LIMITS["auth_response_header_bytes"]:
        raise EnvelopeError("auth_response_header_bytes")
    if not raw.endswith(b"\r\n\r\n") or b"\x00" in raw:
        raise EnvelopeError("malformed_header_block")
    if b"\r" in raw.replace(b"\r\n", b"") or b"\n" in raw.replace(b"\r\n", b""):
        raise EnvelopeError("malformed_header_block")
    status, separator, remainder = raw.partition(b"\r\n")
    status_wire = status + separator
    if not STATUS_LINE.fullmatch(status_wire):
        raise EnvelopeError("malformed_status_line")
    lines = remainder[:-4].split(b"\r\n") if remainder != b"\r\n" else []
    fields: list[tuple[bytes, bytes, int]] = []
    content_lengths: list[bytes] = []
    for line in lines:
        if not line or line[:1] in b" \t" or b":" not in line:
            raise EnvelopeError("malformed_header_block")
        name, value_with_space = line.split(b":", 1)
        if not FIELD_NAME.fullmatch(name) or not value_with_space.startswith(b" "):
            raise EnvelopeError("malformed_header_field")
        value = value_with_space[1:]
        if any(byte < 0x20 or byte == 0x7F for byte in value):
            raise EnvelopeError("malformed_header_value")
        lowered = name.lower()
        if lowered == b"content-length":
            if not value.isdigit():
                raise EnvelopeError("ambiguous_framing")
            content_lengths.append(value)
        if lowered == b"transfer-encoding":
            raise EnvelopeError("ambiguous_framing")
        fields.append((lowered, value, len(line) + 2))
    if content_lengths and any(
        value != content_lengths[0] for value in content_lengths
    ):
        raise EnvelopeError("ambiguous_framing")
    return fields


def measure(raw: bytes) -> dict[str, int | bool]:
    fields = parse_response_header_block(raw)
    cookies = [(value, wire) for name, value, wire in fields if name == b"set-cookie"]
    tokens = [
        value for name, value, _ in fields if name == b"x-auth-request-access-token"
    ]
    if len(cookies) > LIMITS["set_cookie_fields"]:
        raise EnvelopeError("set_cookie_fields")
    if sum(wire for _, wire in cookies) > LIMITS["set_cookie_total_bytes"]:
        raise EnvelopeError("set_cookie_total_bytes")
    if any(wire > LIMITS["set_cookie_field_bytes"] for _, wire in cookies):
        raise EnvelopeError("set_cookie_field_bytes")
    active_pairs = [
        pair
        for value, _ in cookies
        if _session_cookie_name((pair := _cookie_pair(value))[0])
    ]
    if len(active_pairs) > LIMITS["active_cookie_parts"]:
        raise EnvelopeError("active_cookie_parts")
    if any(
        len(name) + len(value) > LIMITS["cookie_name_value_bytes"]
        for name, value in active_pairs
    ):
        raise EnvelopeError("cookie_name_value_bytes")
    active_value_bytes = sum(len(value) for _, value in active_pairs)
    if active_value_bytes > LIMITS["active_cookie_value_bytes"]:
        raise EnvelopeError("active_cookie_value_bytes")
    if (
        len(tokens) != 1
        or not tokens[0]
        or len(tokens[0]) > LIMITS["gateway_token_bytes"]
    ):
        raise EnvelopeError("gateway_token")
    return {
        "auth_response_header_bytes": len(raw),
        "gateway_token_bytes": len(tokens[0]),
        "active_cookie_parts": len(active_pairs),
        "active_cookie_value_bytes": active_value_bytes,
        "set_cookie_fields": len(cookies),
        "set_cookie_total_bytes": sum(wire for _, wire in cookies),
        "within_limits": True,
    }


def measure_incoming_cookie(raw_field: bytes) -> dict[str, int | bool]:
    """Measure exactly one complete `Cookie: value\r\n` wire field."""

    if len(raw_field) > LIMITS["incoming_cookie_bytes"]:
        raise EnvelopeError("incoming_cookie_bytes")
    if not raw_field.startswith(b"Cookie: ") or not raw_field.endswith(b"\r\n"):
        raise EnvelopeError("malformed_cookie")
    value = raw_field[len(b"Cookie: ") : -2]
    if any(byte < 0x20 or byte == 0x7F for byte in value):
        raise EnvelopeError("malformed_cookie")
    pairs = [_cookie_pair(part.strip()) for part in value.split(b";") if part.strip()]
    active = [pair for pair in pairs if _session_cookie_name(pair[0])]
    if len(active) > LIMITS["active_cookie_parts"]:
        raise EnvelopeError("active_cookie_parts")
    if any(
        len(name) + len(item) > LIMITS["cookie_name_value_bytes"]
        for name, item in active
    ):
        raise EnvelopeError("cookie_name_value_bytes")
    total = sum(len(item) for _, item in active)
    if total > LIMITS["active_cookie_value_bytes"]:
        raise EnvelopeError("active_cookie_value_bytes")
    return {
        "incoming_cookie_bytes": len(raw_field),
        "active_cookie_parts": len(active),
        "active_cookie_value_bytes": total,
        "within_limits": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("header_block", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.header_block.read_bytes())
    except (EnvelopeError, OSError):
        print(json.dumps({"result": "FAIL", "failure_code": "COOKIE_ENVELOPE"}))
        raise SystemExit(1) from None
    print(
        json.dumps(
            {"result": "PASS", "measurements": result, "failure_code": "NONE"},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
