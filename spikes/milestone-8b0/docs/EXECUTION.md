# Ubuntu execution and decision procedure

## Host preparation

Use a freshly installed disposable Ubuntu Server 24.04 LTS amd64 VM with at
least 4 vCPUs, 8 GiB RAM, and 30 GiB free storage. Install only `curl`, `jq`,
`openssl`, `podman`, `python3`, `skopeo`, `cosign`, `certutil`, Chromium,
ChromeDriver, and Selenium for Python. Verify Caddy's Sigstore certificate and
signature as well as its checksum; verify OAuth2 Proxy against the official
release `sha256sum.txt`. Do not reuse a developer workstation, persistent Podman
store, browser profile, or credential directory.

Create two synthetic DNS names resolving to the VM's Caddy listener:
`aerodocs-spike.invalid` and `idp.aerodocs-spike.invalid`. The dedicated IdP
site proxies only to Keycloak. The AeroDocs site never shares the IdP routing
block.

## Keycloak early gate

Resolve `quay.io/keycloak/keycloak:26.7.0` directly from Quay. Record the OCI
index digest, linux/amd64 child manifest digest, configuration digest,
architecture, creation timestamp, and official release and registry sources.
After review, pull by the approved child digest and require local inspection to
match exactly.

Mount the realm template read-only at `/opt/keycloak/data/import`. Supply the
bootstrap password, synthetic user password, and OAuth client secret using
runtime-only Podman secret files. Start standard server mode with realm import,
an ephemeral database/runtime volume, no preview or experimental feature
flags, and a loopback-published HTTP port.

Before any gateway test, verify exact version 26.7.0, realm import, issuer and
endpoint origin confinement, RS256, exact synthetic audience, bounded groups,
PKCE S256 enforcement, refresh behavior, absence of enabled experimental
features, and lack of any non-loopback listener. Any failure stops the spike.

## Measurements and hard gates

`scripts/measure_envelope.py` consumes a binary-safe HTTP/1 header block from a
runtime file and emits sanitized counts only. It rejects, without truncation:

| Envelope | Limit |
| --- | ---: |
| gateway access-token JWT | 3,072 bytes |
| active session-cookie values | 10,240 bytes |
| active session-cookie parts | 3 |
| cookie name plus value | 4,096 bytes |
| individual Set-Cookie field | 4,608 bytes |
| incoming Cookie field | 16,384 bytes |
| auth-response Set-Cookie bytes | 18,432 bytes |
| auth-response Set-Cookie fields | 4 |
| complete auth-response header block | 24,576 bytes |

Boundary and one-over-boundary cases are mandatory. Caddy must reject the
complete authentication envelope atomically: no partial cookies, first-cookie
fallback, silent truncation, or AeroDocs invocation is acceptable.

Run initial login and forced refresh through a fresh headless browser. Compare
the ordered synthetic Set-Cookie part identifiers, counts, and attributes seen at the Caddy
boundary with those stored by the browser. Never persist their values. Confirm
that the access-token response header is absent, all authentication responses
are non-cacheable, browser cookies never reach the mock, and the mock status
and body remain unchanged.

The canonical browser login entry and return target is the exact same-origin
relative path `/ui/`. The root `/`, `/ui` without its trailing slash, and
unknown UI-like paths remain fixed 404 responses.

Run `probe_authorization.py` over HTTP/1.1 and HTTP/2 where the client permits.
Two different fields, two identical fields, comma-combined values, repeated
schemes, malformed whitespace, empty fields, and allowed control variants must
be rejected by Caddy or arrive at the mock in a form marked deterministically
invalid. Normalization into one apparently valid bearer token is a hard fail.

## Decision and cleanup

Stock Caddy 2.11.4 has no reviewed primitive in this configuration that proves
the response-field cardinality required for the gateway access token. The
envelope probe therefore exercises every synthetic valid and invalid response
against a fail-safe browser branch, then records `FAIL` and stops. The fixed
503 branch copies neither tokens nor cookies and cannot invoke AeroDocs.

PASS requires every test and a reviewed attestation. Missing or inconclusive
observations are FAIL, not skipped. Stop immediately for lost/collapsed
Set-Cookie fields, unenforceable limits, partial forwarding, access-token
leakage, cookie leakage, unsafe Authorization normalization, OAuth2 Proxy in
the application data plane, JSON login HTML, or a required non-stock Caddy
module.

Always run `scripts/cleanup.sh`, even after failure. It removes spike
containers, pods, networks, images, secrets, volumes, temporary certificates,
browser profiles, credentials, and logs. Verify `git status --short` contains
none of those artifacts before retaining sanitized evidence.
