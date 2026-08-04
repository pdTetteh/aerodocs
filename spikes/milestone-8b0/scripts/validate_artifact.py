#!/usr/bin/env python3
"""Validate and secret-scan a sanitized Stage 8B-0 artifact."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

PATTERNS = {
    "authorization": re.compile(r"(?i)\bauthorization\s*[:=]"),
    "cookie": re.compile(r"(?i)\b(?:set-cookie|cookie)\s*[:=]"),
    "jwt": re.compile(
        r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
    ),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "secret": re.compile(r"(?i)\b(?:client[_-]?secret|password)\s*[:=]\s*[^\s,}\]]+"),
    "runtime_path": re.compile(r"(?:/tmp/|/private/tmp/|/home/|/Users/|[A-Za-z]:\\)"),
}


def scan(value: object, location: str = "$") -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            findings.extend(scan(item, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(scan(item, f"{location}[{index}]"))
    elif isinstance(value, str):
        for category, pattern in PATTERNS.items():
            if pattern.search(value):
                findings.append({"category": category, "location": location})
    return findings


def validate(schema_path: Path, artifact_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(artifact)
    findings = scan(artifact)
    if findings:
        # Never print the matched value.
        print(json.dumps({"result": "FAIL", "findings": findings}, sort_keys=True))
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("schema", type=Path, nargs="?")
    parser.add_argument("artifact", type=Path, nargs="?")
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()
    if args.scan_only:
        target = args.artifact or args.schema
        if target is None:
            raise SystemExit(2)
        artifact = json.loads(target.read_text(encoding="utf-8"))
        findings = scan(artifact)
        if findings:
            print(json.dumps({"result": "FAIL", "findings": findings}, sort_keys=True))
            raise SystemExit(1)
    else:
        if args.schema is None or args.artifact is None:
            raise SystemExit(2)
        validate(args.schema, args.artifact)
    print('{"result":"PASS"}')


if __name__ == "__main__":
    main()
