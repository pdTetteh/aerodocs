#!/usr/bin/env python3
"""Closed Stage 8B-0 gate state machine; omission after live start is FAIL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GATES = (
    "ubuntu_version",
    "ubuntu_architecture",
    "caddy_provenance",
    "caddy_exact_version",
    "caddy_stock_modules",
    "oauth2_proxy_provenance",
    "oauth2_proxy_exact_version",
    "keycloak_index_digest",
    "keycloak_platform_manifest_digest",
    "keycloak_config_digest",
    "keycloak_os",
    "keycloak_architecture",
    "keycloak_reported_version",
    "realm_import",
    "issuer_endpoint_confinement",
    "rs256",
    "exact_audience",
    "bounded_groups",
    "pkce_s256",
    "refresh_support",
    "no_experimental_features",
    "loopback_isolation",
    "adapted_caddy_validation",
    "health_route",
    "oauth2_routes",
    "hostile_header_stripping",
    "direct_bearer_separation",
    "duplicate_authorization_http1",
    "duplicate_authorization_http2",
    "initial_login",
    "forced_refresh",
    "clearing_refresh",
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
    "json_navigation_behavior",
    "operator_mutation_behavior",
    "unknown_route_behavior",
    "long_running_response",
    "data_plane_separation",
    "secret_scan",
    "protected_file_comparison",
    "regression_suite",
    "podman_interruption_cleanup",
    "cleanup",
)
STATES = {"PASS", "FAIL", "NOT_YET_EXECUTED"}


def initial() -> dict[str, str]:
    return dict.fromkeys(GATES, "NOT_YET_EXECUTED")


def decide(gates: dict[str, str], live_started: bool) -> str:
    if set(gates) != set(GATES) or any(value not in STATES for value in gates.values()):
        return "FAIL"
    if any(value == "FAIL" for value in gates.values()):
        return "FAIL"
    if all(value == "PASS" for value in gates.values()):
        return "PASS"
    return "FAIL" if live_started else "NOT_YET_EXECUTED"


def apply_probe_results(
    gates: dict[str, str], results: dict[str, dict[str, object]]
) -> dict[str, str]:
    if not results:
        raise ValueError("missing_probe_result")
    updated = dict(gates)
    for gate, item in results.items():
        if gate not in GATES:
            raise ValueError("unknown_gate")
        result = item.get("result")
        measurements = item.get("measurements")
        if (
            result not in {"PASS", "FAIL"}
            or not isinstance(measurements, dict)
            or not measurements
        ):
            raise ValueError("invalid_probe_result")
        previous = updated[gate]
        if previous != "NOT_YET_EXECUTED" and previous != result:
            raise ValueError("conflicting_probe_result")
        updated[gate] = str(result)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--set", nargs=2, metavar=("GATE", "STATE"))
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--live-started", action="store_true")
    parser.add_argument("--apply-result", type=Path)
    args = parser.parse_args()
    finalized_marker = args.state.with_name(args.state.name + ".finalized")
    if finalized_marker.exists():
        raise SystemExit(2)
    gates = (
        initial() if args.init else json.loads(args.state.read_text(encoding="utf-8"))
    )
    if args.set:
        gate, state = args.set
        if gate not in GATES or state not in STATES:
            raise SystemExit(2)
        gates[gate] = state
    if args.apply_result:
        document = json.loads(args.apply_result.read_text(encoding="utf-8"))
        results = document.get("gate_results") if isinstance(document, dict) else None
        if not isinstance(results, dict):
            raise SystemExit(2)
        try:
            gates = apply_probe_results(gates, results)
        except ValueError:
            raise SystemExit(2) from None
    args.state.write_text(
        json.dumps(gates, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    if args.finalize:
        decision = decide(gates, args.live_started)
        finalized_marker.write_text(decision + "\n", encoding="ascii")
        finalized_marker.chmod(0o600)
        print(decision)
        raise SystemExit(
            0 if decision == "PASS" else (20 if decision == "NOT_YET_EXECUTED" else 1)
        )


if __name__ == "__main__":
    main()
