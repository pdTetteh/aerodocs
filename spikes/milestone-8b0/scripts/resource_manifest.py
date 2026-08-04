#!/usr/bin/env python3
"""Validate and select the closed Stage 8B-0 Podman resource manifest."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

KINDS = {"container", "pod", "network", "volume", "secret", "image"}
PURPOSES = {
    "interruption_target",
    "unrelated_preservation_control",
    "controller_test_fixture",
}
CREATORS = {"child", "controller"}
LIFECYCLES = {"interrupt_cleanup", "preserve_until_final_teardown"}
PHASES = {
    "child_cleanup",
    "controller_fallback",
    "controller_final_teardown",
}
FINAL_STATES = {"absent"}
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
LABEL = re.compile(r"^[a-f0-9]{24}$")
RESOURCE_FIELDS = {
    "kind",
    "identifier",
    "run_label",
    "purpose",
    "creator",
    "lifecycle",
    "child_cleanup",
    "controller_fallback",
    "controller_final_teardown",
    "expected_at_checkpoint",
    "expected_during_verification",
    "expected_final_state",
}


@dataclass(frozen=True)
class ResourceEntry:
    kind: str
    identifier: str
    run_label: str
    purpose: str
    creator: str
    lifecycle: str
    child_cleanup: bool
    controller_fallback: bool
    controller_final_teardown: bool
    expected_at_checkpoint: bool
    expected_during_verification: bool
    expected_final_state: str

    def permits(self, phase: str) -> bool:
        if phase == "child_cleanup":
            return self.child_cleanup
        if phase == "controller_fallback":
            return self.controller_fallback
        if phase == "controller_final_teardown":
            return self.controller_final_teardown
        raise ValueError("cleanup_phase")


@dataclass(frozen=True)
class ResourceManifest:
    schema_version: int
    run_label: str
    resources: tuple[ResourceEntry, ...]


def _entry(value: object) -> ResourceEntry:
    if not isinstance(value, dict) or set(value) != RESOURCE_FIELDS:
        raise ValueError("resource_shape")
    kind = value["kind"]
    identifier = value["identifier"]
    run_label = value["run_label"]
    purpose = value["purpose"]
    creator = value["creator"]
    lifecycle = value["lifecycle"]
    final_state = value["expected_final_state"]
    boolean_fields = (
        "child_cleanup",
        "controller_fallback",
        "controller_final_teardown",
        "expected_at_checkpoint",
        "expected_during_verification",
    )
    if (
        kind not in KINDS
        or not isinstance(identifier, str)
        or not IDENTIFIER.fullmatch(identifier)
        or not isinstance(run_label, str)
        or not LABEL.fullmatch(run_label)
        or purpose not in PURPOSES
        or creator not in CREATORS
        or lifecycle not in LIFECYCLES
        or final_state not in FINAL_STATES
        or any(not isinstance(value[field], bool) for field in boolean_fields)
    ):
        raise ValueError("resource_identity_or_policy")
    entry = ResourceEntry(
        kind=kind,
        identifier=identifier,
        run_label=run_label,
        purpose=purpose,
        creator=creator,
        lifecycle=lifecycle,
        child_cleanup=value["child_cleanup"],
        controller_fallback=value["controller_fallback"],
        controller_final_teardown=value["controller_final_teardown"],
        expected_at_checkpoint=value["expected_at_checkpoint"],
        expected_during_verification=value["expected_during_verification"],
        expected_final_state=final_state,
    )
    target_policy = (
        entry.purpose == "interruption_target"
        and entry.creator == "child"
        and entry.lifecycle == "interrupt_cleanup"
        and entry.child_cleanup
        and entry.controller_fallback
        and not entry.controller_final_teardown
        and entry.expected_at_checkpoint
        and not entry.expected_during_verification
    )
    control_policy = (
        entry.purpose
        in {
            "unrelated_preservation_control",
            "controller_test_fixture",
        }
        and entry.creator == "controller"
        and entry.lifecycle == "preserve_until_final_teardown"
        and not entry.child_cleanup
        and not entry.controller_fallback
        and entry.controller_final_teardown
        and entry.expected_at_checkpoint
        and entry.expected_during_verification
    )
    if not (target_policy or control_policy):
        raise ValueError("conflicting_resource_policy")
    return entry


def validate(value: object) -> ResourceManifest:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "run_label",
        "resources",
    }:
        raise ValueError("manifest_shape")
    label = value["run_label"]
    resources = value["resources"]
    if (
        value["schema_version"] != 2
        or not isinstance(label, str)
        or not LABEL.fullmatch(label)
    ):
        raise ValueError("manifest_header")
    if not isinstance(resources, list) or not 1 <= len(resources) <= 16:
        raise ValueError("manifest_resources")
    entries = tuple(_entry(item) for item in resources)
    identities = [(entry.kind, entry.identifier) for entry in entries]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate_resource")
    return ResourceManifest(2, label, entries)


def load(path: Path) -> ResourceManifest:
    return validate(json.loads(path.read_text(encoding="utf-8")))


def select(manifest: ResourceManifest, phase: str) -> tuple[ResourceEntry, ...]:
    if phase not in PHASES:
        raise ValueError("cleanup_phase")
    return tuple(entry for entry in manifest.resources if entry.permits(phase))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--phase", choices=sorted(PHASES))
    args = parser.parse_args()
    manifest = load(args.manifest)
    if args.phase:
        for entry in select(manifest, args.phase):
            print(
                "\t".join(
                    (
                        entry.kind,
                        entry.identifier,
                        entry.run_label,
                        entry.purpose,
                        entry.creator,
                    )
                )
            )


if __name__ == "__main__":
    main()
