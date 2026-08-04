#!/usr/bin/env python3
"""Build, validate, and scan retained sanitized Stage 8B-0 FAIL artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from gate_registry import GATES
from jsonschema import Draft202012Validator, FormatChecker
from validate_artifact import scan


def _validate(schema_path: Path, value: object) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)
    if scan(value):
        raise ValueError("sensitive_artifact")


def build(
    spike: Path, state_path: Path, gate: str, code: str
) -> tuple[dict[str, object], dict[str, object]]:
    if gate not in GATES:
        raise ValueError("unknown_gate")
    states = json.loads(state_path.read_text(encoding="utf-8"))
    if set(states) != set(GATES):
        raise ValueError("invalid_registry")
    states[gate] = "FAIL"
    state_path.write_text(
        json.dumps(states, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    event: dict[str, object] = {
        "schema_version": 2,
        "test_id": "live_failure",
        "observed_at": now,
        "result": "FAIL",
        "measurements": {
            "failed_gate_count": sum(value == "FAIL" for value in states.values())
        },
        "failure_code": code,
    }
    attestation = json.loads(
        (spike / "examples/attestation.not-executed.json").read_text(encoding="utf-8")
    )
    attestation.update(
        status="FAIL",
        hard_gates=states,
        failed_gates=sorted(name for name, value in states.items() if value == "FAIL"),
        failure_code=code,
        review={"reviewed": False, "decision": "FAIL"},
    )
    _validate(spike / "schemas/evidence.schema.json", event)
    _validate(spike / "schemas/attestation.schema.json", attestation)
    return event, attestation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spike", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--failure-code", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    event, attestation = build(args.spike, args.state, args.gate, args.failure_code)
    args.output.mkdir(mode=0o700, parents=False, exist_ok=True)
    pending_event = args.output / ".failure-evidence.pending"
    pending_attestation = args.output / ".failure-attestation.pending"
    pending_event.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    pending_attestation.write_text(
        json.dumps(attestation, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    pending_event.chmod(0o600)
    pending_attestation.chmod(0o600)
    pending_event.replace(args.output / "failure-evidence.json")
    pending_attestation.replace(args.output / "failure-attestation.json")


if __name__ == "__main__":
    main()
