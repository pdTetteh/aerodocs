# AeroDocs Milestone 8B-0 compatibility spike

This directory contains the secret-free materials for the stock-Caddy gateway
compatibility spike. It does not implement a gateway credential source in
AeroDocs. A result is valid only when the complete harness runs on a disposable
Ubuntu Server 24.04 LTS amd64 host.

The initial state is `NOT_YET_EXECUTED`. macOS checks validate only repository
structure and must never be cited as Linux compatibility evidence.
The local synthetic `caddy-data/bootstrap-test` interruption case proves only
filesystem cleanup. It is not Podman evidence. The separate Ubuntu-only
`scripts/probe_podman_interruption.py` gate creates and removes real Podman
network and volume resources; its committed state is `NOT_YET_EXECUTED`.
Its controller writes one mode-0600, closed resource manifest before any
probe-owned Podman resource is created. Interruption targets and unrelated
preservation controls carry explicit child, fallback, and final-teardown
permissions. Every cleanup phase accepts only exact kind, generated identifier,
run-label, creator, and purpose entries from that manifest. Checkpoint readiness is one bounded,
length-prefixed JSON frame followed by descriptor EOF; the child then blocks
on a separate control descriptor. Child removal, controller verification,
fallback removal, control teardown, and forced termination have independent
evidence counters.

## Fixed components

- stock Caddy 2.11.4;
- OAuth2 Proxy 7.15.3 official linux-amd64 release artifact;
- official `quay.io/keycloak/keycloak:26.7.0` image, resolved and pulled by a
  reviewed immutable linux/amd64 manifest digest;
- Ubuntu Server 24.04 LTS amd64.

No digest or checksum is embedded here. `scripts/resolve-provenance.sh` obtains
official checksum material and unresolved image metadata on the spike host.
The resulting provenance record must be reviewed before the execution phase.

## Directory layout

```text
config/                 exact non-secret candidate configurations
docs/                   architecture, procedures, and threat boundary
keycloak/               secret-free synthetic realm import
schemas/                evidence and attestation JSON Schemas
scripts/                Ubuntu harness, verification, probes, and cleanup
mock_aerodocs.py         bounded, credential-blind mock upstream
```

Runtime data belongs in a directory created by `mktemp -d` outside the
repository. Never place binaries, images, volumes, TLS keys, browser profiles,
cookies, tokens, passwords, client secrets, raw headers, or runtime logs here.

## Execution contract

Run `scripts/run-ubuntu-spike.sh` from a disposable Ubuntu host. The harness
first verifies the OS and architecture, prepares a proposed provenance record,
and stops with exit code 20. Reviewers must place the approved immutable values
in the runtime-only approval file before rerunning. The second run verifies and
pulls only those approved artifacts, then performs the Keycloak early gate.

The live compatibility tests are deliberately fail-closed. Any unimplemented
probe, missing tool, ambiguous observation, or hard-gate failure produces
`FAIL`; it can never be treated as a skip or PASS.

The reviewed stock-Caddy configuration cannot prove the cardinality of
`X-Auth-Request-Access-Token` response fields: its response placeholder is a
scalar representation, not evidence that exactly one wire field was received.
The candidate therefore uses the approved fail-safe outcome. It exercises the
synthetic envelope cases, records `gateway_token_limit=FAIL`, performs cleanup,
retains sanitized failure artifacts, and stops. Browser-authenticated branches
return a fixed 503 and never proxy to mock AeroDocs. Stage 8B-1 remains barred.

See `docs/EXECUTION.md` for the ordered procedure and evidence rules.
