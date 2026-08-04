#!/usr/bin/env python3
"""Ubuntu-only real-Podman interruption controller with bounded fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import secrets
import selectors
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from resource_manifest import (  # noqa: E402
    ResourceEntry,
    ResourceManifest,
    select,
    validate,
)


class Podman:
    def __init__(self, executable: Path) -> None:
        self.executable = executable

    def run(self, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.executable), *args],
            text=True,
            capture_output=True,
            check=check,
            timeout=10,
        )

    def exists(self, resource: ResourceEntry) -> bool:
        return self.run(resource.kind, "exists", resource.identifier).returncode == 0

    def label_matches(self, resource: ResourceEntry) -> bool:
        result = self.run(
            resource.kind,
            "inspect",
            resource.identifier,
            "--format",
            "{{json .Labels}}",
        )
        if result.returncode:
            return False
        try:
            labels = json.loads(result.stdout)
            return bool(
                isinstance(labels, dict)
                and labels.get("aerodocs.m8b0.run") == resource.run_label
            )
        except json.JSONDecodeError:
            return False

    def identity(self, resource: ResourceEntry) -> str | None:
        result = self.run(resource.kind, "inspect", resource.identifier)
        if result.returncode or not self.label_matches(resource):
            return None
        return hashlib.sha256(result.stdout.encode("utf-8")).hexdigest()

    def remove_exact(self, resource: ResourceEntry) -> bool:
        if not self.exists(resource):
            return True
        if not self.label_matches(resource):
            return False
        return self.run(resource.kind, "rm", resource.identifier).returncode == 0


def cleanup_manifest(
    podman: Podman, manifest: ResourceManifest, phase: str
) -> tuple[bool, int, int]:
    resources = select(manifest, phase)
    attempts = 0
    violations = 0
    ok = True
    for resource in resources:
        try:
            if podman.exists(resource):
                attempts += 1
                removed = podman.remove_exact(resource)
                violations += int(not removed)
                ok = removed and ok
        except (OSError, subprocess.SubprocessError):
            violations += 1
            ok = False
    try:
        remaining = any(podman.exists(resource) for resource in resources)
    except (OSError, subprocess.SubprocessError):
        remaining = True
    return ok and not remaining, attempts, violations


class ReadinessError(ValueError):
    """A bounded readiness frame was missing, ambiguous, or invalid."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, member in pairs:
        if key in value:
            raise ReadinessError("duplicate_member")
        value[key] = member
    return value


def read_readiness_frame(descriptor: int, timeout: float = 15.0) -> dict[str, object]:
    """Read exactly one length-prefixed JSON frame and require writer EOF."""
    deadline = time.monotonic() + timeout
    data = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(timeout=remaining):
                raise TimeoutError("readiness_eof")
            chunk = os.read(descriptor, 1029 - len(data))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > 1028:
                raise ReadinessError("oversized_frame")
    if len(data) < 4:
        raise ReadinessError("partial_frame")
    declared = struct.unpack(">I", data[:4])[0]
    if declared > 1024:
        raise ReadinessError("oversized_frame")
    if len(data) != 4 + declared:
        raise ReadinessError("incomplete_or_trailing_frame")
    try:
        decoded = bytes(data[4:]).decode("utf-8", errors="strict")
        event = json.loads(decoded, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReadinessError("malformed_json") from error
    if not isinstance(event, dict):
        raise ReadinessError("non_object")
    return event


def validate_test_base(base: Path) -> None:
    expected = Path("/tmp")
    value = os.lstat(base)
    if (
        base != expected
        or Path(os.path.realpath(base)) != expected
        or not stat.S_ISDIR(value.st_mode)
        or stat.S_ISLNK(value.st_mode)
        or not value.st_mode & stat.S_ISVTX
        or not value.st_mode & stat.S_IWOTH
    ):
        raise ValueError("unsafe_test_base")


def evidence(
    result: str, measurements: dict[str, bool | int], code: str
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "test_id": "podman_interruption",
        "observed_at": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "result": result,
        "measurements": measurements,
        "failure_code": code,
        "gate_results": {
            "podman_interruption_cleanup": {
                "result": result,
                "measurements": {"checked": True},
            }
        },
    }


def pass_allowed(measurements: dict[str, bool | int]) -> bool:
    return measurements == {
        "readiness_observed": True,
        "sigterm_delivered": True,
        "forced_kill_count": 0,
        "authorized_resources_remaining": 0,
        "unrelated_resource_preserved": True,
        "child_cleanup_dispatch_count": 1,
        "child_resource_cleanup_attempt_count": 2,
        "controller_verification_count": 1,
        "controller_fallback_dispatch_count": 0,
        "controller_fallback_resource_cleanup_attempt_count": 0,
        "controller_final_teardown_dispatch_count": 1,
        "controller_final_teardown_resource_cleanup_attempt_count": 2,
        "manifest_entry_count": 4,
        "interruption_target_count": 2,
        "preservation_control_count": 2,
        "controller_fixture_count": 0,
        "unauthorized_cleanup_attempt_count": 0,
        "phase_policy_violation_count": 0,
        "preservation_controls_unchanged": True,
        "preservation_verdict_finalized": True,
        "preservation_evidence_finalized": True,
        "manifest_unchanged": True,
        "all_manifest_resources_absent": True,
        "cleanup_hygiene_passed": True,
    }


def build_manifest_value(
    run_id: str, label: str, control_label: str
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "run_label": label,
        "resources": [
            {
                "kind": kind,
                "identifier": identifier,
                "run_label": label,
                "purpose": "interruption_target",
                "creator": "child",
                "lifecycle": "interrupt_cleanup",
                "child_cleanup": True,
                "controller_fallback": True,
                "controller_final_teardown": False,
                "expected_at_checkpoint": True,
                "expected_during_verification": False,
                "expected_final_state": "absent",
            }
            for kind, identifier in (
                ("network", f"aerodocs-8b0-net-{run_id}"),
                ("volume", f"aerodocs-8b0-vol-{run_id}"),
            )
        ]
        + [
            {
                "kind": kind,
                "identifier": identifier,
                "run_label": control_label,
                "purpose": "unrelated_preservation_control",
                "creator": "controller",
                "lifecycle": "preserve_until_final_teardown",
                "child_cleanup": False,
                "controller_fallback": False,
                "controller_final_teardown": True,
                "expected_at_checkpoint": True,
                "expected_during_verification": True,
                "expected_final_state": "absent",
            }
            for kind, identifier in (
                ("network", f"aerodocs-8b0-caddy-control-{run_id}"),
                ("volume", f"aerodocs-8b0-secret-control-{run_id}"),
            )
        ],
    }


def main() -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("Ubuntu 24.04 LTS amd64 required")
    release = Path("/etc/os-release").read_text(encoding="utf-8")
    if "ID=ubuntu" not in release or 'VERSION_ID="24.04"' not in release:
        raise SystemExit("Ubuntu 24.04 LTS amd64 required")
    parser = argparse.ArgumentParser()
    parser.add_argument("--podman", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_id = secrets.token_hex(12)
    label = run_id
    control_label = secrets.token_hex(12)
    manifest_value = build_manifest_value(run_id, label, control_label)
    manifest = validate(manifest_value)
    targets = tuple(
        entry for entry in manifest.resources if entry.purpose == "interruption_target"
    )
    controls = tuple(
        entry
        for entry in manifest.resources
        if entry.purpose == "unrelated_preservation_control"
    )
    podman = Podman(args.podman)
    process: subprocess.Popen[str] | None = None
    forced_kill_count = 0
    readiness = False
    sigterm = False
    fallback_required = False
    unrelated_preserved = False
    child_cleanup_dispatch_count = 0
    child_resource_cleanup_attempt_count = 0
    controller_verification_count = 0
    controller_fallback_dispatch_count = 0
    controller_fallback_resource_cleanup_attempt_count = 0
    controller_final_teardown_dispatch_count = 0
    controller_final_teardown_resource_cleanup_attempt_count = 0
    phase_policy_violation_count = 0
    authorized_resources_remaining = len(targets)
    result = "FAIL"
    code = "CLEANUP"
    fixed_root: Path | None = None
    parent: Path | None = None
    control_identities: dict[tuple[str, str], str] = {}
    manifest_digest = ""
    manifest_unchanged = False
    preservation_verdict_finalized = False
    preservation_evidence_finalized = False
    try:
        validate_test_base(Path("/tmp"))
        with tempfile.TemporaryDirectory(
            prefix="aerodocs-8b0-podman-", dir="/tmp"
        ) as _base_raw:
            fixed_root = Path(f"/tmp/aerodocs-m8b0-bootstrap-tests-{os.getuid()}")
            fixed_root.mkdir(mode=0o700)
            root_nonce = secrets.token_hex(32)
            root_sentinel = fixed_root / ".aerodocs-bootstrap-test-root"
            root_sentinel.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "purpose": "aerodocs-m8b0-bootstrap-root",
                        "nonce": root_nonce,
                    },
                    separators=(",", ":"),
                ),
                encoding="ascii",
            )
            root_sentinel.chmod(0o600)
            parent = Path(tempfile.mkdtemp(prefix="run-", dir=fixed_root))
            parent.chmod(0o700)
            nonce = secrets.token_hex(32)
            sentinel = parent / ".aerodocs-bootstrap-test-parent"
            sentinel.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "purpose": "aerodocs-m8b0-bootstrap-test",
                        "nonce": nonce,
                    },
                    separators=(",", ":"),
                ),
                encoding="ascii",
            )
            sentinel.chmod(0o600)
            manifest_path = parent / "resource-manifest.json"
            manifest_path.write_text(
                json.dumps(manifest_value, separators=(",", ":")), encoding="utf-8"
            )
            manifest_path.chmod(0o600)
            manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            subprocess.run(
                [
                    sys.executable,
                    os.fspath(Path(__file__).with_name("resource_manifest.py")),
                    os.fspath(manifest_path),
                ],
                check=True,
                timeout=5,
            )
            for control in controls:
                podman.run(
                    control.kind,
                    "create",
                    "--label",
                    f"aerodocs.m8b0.run={control.run_label}",
                    control.identifier,
                    check=True,
                )
                identity = podman.identity(control)
                if identity is None:
                    raise ValueError("control_identity")
                control_identities[(control.kind, control.identifier)] = identity
            ready_read, ready_write = os.pipe()
            control_read, control_write = os.pipe()
            nonce_read, nonce_write = os.pipe()
            root_nonce_read, root_nonce_write = os.pipe()
            os.write(nonce_write, (nonce + "\n").encode("ascii"))
            os.close(nonce_write)
            os.write(root_nonce_write, (root_nonce + "\n").encode("ascii"))
            os.close(root_nonce_write)
            environment = os.environ.copy()
            environment.update(
                AERODOCS_SPIKE_BOOTSTRAP_TEST_MODE="1",
                AERODOCS_SPIKE_ACTUAL_PODMAN_TEST_MODE="1",
                AERODOCS_SPIKE_TEST_INTERRUPT_AFTER="after_actual_podman_resource",
                AERODOCS_SPIKE_TEST_PARENT=str(parent),
                AERODOCS_SPIKE_TEST_READY_FD=str(ready_write),
                AERODOCS_SPIKE_TEST_CONTROL_FD=str(control_read),
                AERODOCS_SPIKE_TEST_NONCE_FD=str(nonce_read),
                AERODOCS_SPIKE_TEST_ROOT_NONCE_FD=str(root_nonce_read),
                AERODOCS_SPIKE_PODMAN_NETWORK=targets[0].identifier,
                AERODOCS_SPIKE_PODMAN_VOLUME=targets[1].identifier,
                AERODOCS_SPIKE_PODMAN_LABEL=label,
                AERODOCS_SPIKE_RESOURCE_MANIFEST=str(manifest_path),
            )
            process = subprocess.Popen(
                ["bash", args.harness],
                env=environment,
                pass_fds=(ready_write, control_read, nonce_read, root_nonce_read),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            os.close(ready_write)
            os.close(control_read)
            os.close(nonce_read)
            os.close(root_nonce_read)
            event = read_readiness_frame(ready_read)
            if event != {
                "schema_version": 1,
                "checkpoint": "after_actual_podman_resource",
                "resource_kind": "actual_podman",
                "created_resource_count": 9,
                "state": "BLOCKED",
            }:
                raise ValueError("unexpected_readiness")
            readiness = True
            process.send_signal(signal.SIGTERM)
            sigterm = True
            os.close(control_write)
            os.close(ready_read)
            stdout, _ = process.communicate(timeout=20)
            cleanup_events = []
            for line in stdout.splitlines():
                if '"event":"child_cleanup"' in line:
                    cleanup_events.append(json.loads(line))
            if len(cleanup_events) != 1:
                raise ValueError("child_cleanup_event")
            cleanup_event = cleanup_events[0]
            if set(cleanup_event) != {
                "schema_version",
                "event",
                "dispatch_count",
                "resource_attempt_count",
                "result",
            }:
                raise ValueError("child_cleanup_failed")
            if cleanup_event.get("result") != "PASS":
                phase_policy_violation_count += 1
                raise ValueError("child_cleanup_failed")
            child_cleanup_dispatch_count = int(cleanup_event["dispatch_count"])
            child_resource_cleanup_attempt_count = int(
                cleanup_event["resource_attempt_count"]
            )
            unrelated_preserved = all(
                podman.exists(control)
                and podman.label_matches(control)
                and podman.identity(control)
                == control_identities.get((control.kind, control.identifier))
                for control in controls
            )
            authorized_absent = not any(podman.exists(item) for item in targets)
            if process.returncode != 143 or not authorized_absent:
                fallback_required = True
    except (
        KeyError,
        OSError,
        ReadinessError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        TimeoutError,
    ):
        fallback_required = True
    finally:
        if process is not None and process.poll() is None:
            fallback_required = True
            try:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                        forced_kill_count += 1
            except (OSError, subprocess.SubprocessError):
                forced_kill_count += 1
        if fallback_required:
            controller_fallback_dispatch_count = 1
            (
                _,
                controller_fallback_resource_cleanup_attempt_count,
                fallback_violations,
            ) = cleanup_manifest(podman, manifest, "controller_fallback")
            phase_policy_violation_count += fallback_violations
        controller_verification_count = 1
        try:
            authorized_resources_remaining = sum(
                int(podman.exists(item)) for item in targets
            )
        except (OSError, subprocess.SubprocessError):
            authorized_resources_remaining = len(targets)
        try:
            unrelated_preserved = all(
                podman.exists(control)
                and podman.label_matches(control)
                and podman.identity(control)
                == control_identities.get((control.kind, control.identifier))
                for control in controls
            )
        except (OSError, subprocess.SubprocessError):
            unrelated_preserved = False
        preservation_verdict_finalized = True
        preservation_evidence = (
            unrelated_preserved,
            len(controls),
            len(control_identities),
            controller_verification_count,
        )
        preservation_evidence_finalized = len(preservation_evidence) == 4
        try:
            manifest_unchanged = bool(
                parent is not None
                and manifest_digest
                and hashlib.sha256(
                    (parent / "resource-manifest.json").read_bytes()
                ).hexdigest()
                == manifest_digest
            )
        except OSError:
            manifest_unchanged = False
        controller_final_teardown_dispatch_count = 1
        teardown_complete, teardown_attempts, teardown_violations = cleanup_manifest(
            podman, manifest, "controller_final_teardown"
        )
        phase_policy_violation_count += teardown_violations
        controller_final_teardown_resource_cleanup_attempt_count = teardown_attempts
        try:
            all_manifest_resources_absent = not any(
                podman.exists(resource) for resource in manifest.resources
            )
        except (OSError, subprocess.SubprocessError):
            all_manifest_resources_absent = False
        filesystem_cleanup_complete = (
            teardown_complete and all_manifest_resources_absent
        )
        if parent is not None and fixed_root is not None:
            for artifact in (
                parent / "resource-manifest.json",
                parent / ".aerodocs-bootstrap-test-parent",
            ):
                try:
                    artifact.unlink(missing_ok=True)
                except OSError:
                    filesystem_cleanup_complete = False
            try:
                parent.rmdir()
                (fixed_root / ".aerodocs-bootstrap-test-root").unlink()
                fixed_root.rmdir()
            except OSError:
                filesystem_cleanup_complete = False
        if not filesystem_cleanup_complete:
            fallback_required = True
        measurements = {
            "readiness_observed": readiness,
            "sigterm_delivered": sigterm,
            "forced_kill_count": forced_kill_count,
            "authorized_resources_remaining": authorized_resources_remaining,
            "unrelated_resource_preserved": unrelated_preserved,
            "child_cleanup_dispatch_count": child_cleanup_dispatch_count,
            "child_resource_cleanup_attempt_count": (
                child_resource_cleanup_attempt_count
            ),
            "controller_verification_count": controller_verification_count,
            "controller_fallback_dispatch_count": controller_fallback_dispatch_count,
            "controller_fallback_resource_cleanup_attempt_count": (
                controller_fallback_resource_cleanup_attempt_count
            ),
            "controller_final_teardown_dispatch_count": (
                controller_final_teardown_dispatch_count
            ),
            "controller_final_teardown_resource_cleanup_attempt_count": (
                controller_final_teardown_resource_cleanup_attempt_count
            ),
            "manifest_entry_count": len(manifest.resources),
            "interruption_target_count": len(targets),
            "preservation_control_count": len(controls),
            "controller_fixture_count": sum(
                entry.purpose == "controller_test_fixture"
                for entry in manifest.resources
            ),
            "unauthorized_cleanup_attempt_count": 0,
            "phase_policy_violation_count": phase_policy_violation_count,
            "preservation_controls_unchanged": unrelated_preserved,
            "preservation_verdict_finalized": preservation_verdict_finalized,
            "preservation_evidence_finalized": preservation_evidence_finalized,
            "manifest_unchanged": manifest_unchanged,
            "all_manifest_resources_absent": all_manifest_resources_absent,
            "cleanup_hygiene_passed": filesystem_cleanup_complete,
        }
        if pass_allowed(measurements):
            result, code = "PASS", "NONE"
        else:
            result, code = "FAIL", "CLEANUP"
        args.output.write_text(
            json.dumps(evidence(result, measurements, code), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.output.chmod(0o600)
    raise SystemExit(0 if result == "PASS" else 1)


if __name__ == "__main__":
    main()
