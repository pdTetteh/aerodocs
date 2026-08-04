#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
# shellcheck source=common.sh
source "$(dirname -- "$0")/common.sh"

require_ubuntu_amd64
require_runtime_root
for command_name in cosign curl jq sha256sum skopeo; do
  require_command "${command_name}"
done

downloads="${AERODOCS_SPIKE_RUNTIME}/downloads"
mkdir -m 700 -p "${downloads}"

caddy_base="https://github.com/caddyserver/caddy/releases/download/v2.11.4"
caddy_archive="caddy_2.11.4_linux_amd64.tar.gz"
caddy_checksums="caddy_2.11.4_checksums.txt"
caddy_certificate="caddy_2.11.4_linux_amd64.pem"
caddy_signature="${caddy_archive}.sig"
oauth_base="https://github.com/oauth2-proxy/oauth2-proxy/releases/download/v7.15.3"
oauth_archive="oauth2-proxy-v7.15.3.linux-amd64.tar.gz"
oauth_checksums="sha256sum.txt"
keycloak_ref="quay.io/keycloak/keycloak:26.7.0"

curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  --output "${downloads}/${caddy_checksums}" "${caddy_base}/${caddy_checksums}"
if [[ "${AERODOCS_SPIKE_TEST_INTERRUPT_AFTER:-}" == "after_first_download" ]]; then
  kill -TERM "$PPID"
  exit 143
fi
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  --output "${downloads}/${caddy_archive}" "${caddy_base}/${caddy_archive}"
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  --output "${downloads}/${caddy_certificate}" "${caddy_base}/${caddy_certificate}"
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  --output "${downloads}/${caddy_signature}" "${caddy_base}/${caddy_signature}"
caddy_matches="$(awk -v file="${caddy_archive}" '$2 == file {print $1}' "${downloads}/${caddy_checksums}")"
[[ "$(printf '%s\n' "${caddy_matches}" | awk 'NF {count++} END {print count+0}')" -eq 1 ]] || fail 4 "official Caddy checksum entry is not unique"
caddy_expected="${caddy_matches}"
[[ "${caddy_expected}" =~ ^[a-f0-9]{64}$ ]] || fail 4 "official Caddy checksum missing"
[[ "$(sha256_file "${downloads}/${caddy_archive}")" == "${caddy_expected}" ]] ||
  fail 4 "Caddy archive checksum mismatch"
cosign verify-blob \
  --certificate "${downloads}/${caddy_certificate}" \
  --signature "${downloads}/${caddy_signature}" \
  --certificate-identity-regexp '^https://github.com/caddyserver/caddy/.github/workflows/[^@]+@refs/tags/v2\.11\.4$' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  "${downloads}/${caddy_archive}" >"${downloads}/caddy-cosign-verification.txt" || fail 4 "Caddy provenance verification failed"

curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  --output "${downloads}/${oauth_checksums}" "${oauth_base}/${oauth_checksums}"
curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  --output "${downloads}/${oauth_archive}" "${oauth_base}/${oauth_archive}"
oauth_matches="$(awk -v file="${oauth_archive}" '$2 == file {print $1}' "${downloads}/${oauth_checksums}")"
[[ "$(printf '%s\n' "${oauth_matches}" | awk 'NF {count++} END {print count+0}')" -eq 1 ]] || fail 4 "official OAuth2 Proxy checksum entry is not unique"
oauth_expected="${oauth_matches}"
[[ "${oauth_expected}" =~ ^[a-f0-9]{64}$ ]] || fail 4 "official OAuth2 Proxy checksum malformed"
[[ "$(sha256_file "${downloads}/${oauth_archive}")" == "${oauth_expected}" ]] ||
  fail 4 "OAuth2 Proxy archive checksum mismatch"
for archive in "${downloads}/${caddy_archive}" "${downloads}/${oauth_archive}"; do
  tar -tzf "${archive}" | awk 'BEGIN {bad=0} /^\// || /(^|\/)\.\.($|\/)/ {bad=1} END {exit bad}' || fail 4 "archive contains an unsafe path"
  tar -tvzf "${archive}" | awk 'substr($1,1,1) !~ /[-d]/ {bad=1} END {exit bad}' || fail 4 "archive contains a link or unsupported member"
done
tar -tzf "${downloads}/${caddy_archive}" | awk '$0 !~ /^(caddy|LICENSE|README\.md)$/ {bad=1} END {exit bad}' || fail 4 "Caddy archive contains an unexpected member"
tar -tzf "${downloads}/${oauth_archive}" | awk '$0 !~ /^oauth2-proxy-v7\.15\.3\.linux-amd64\/(oauth2-proxy|LICENSE|README\.md|CHANGELOG\.md)?$/ {bad=1} END {exit bad}' || fail 4 "OAuth2 Proxy archive contains an unexpected member"
proposed_bin="${AERODOCS_SPIKE_RUNTIME}/proposed-bin"
mkdir -m 700 "${proposed_bin}"
tar -xzf "${downloads}/${caddy_archive}" -C "${proposed_bin}" caddy
tar -xzf "${downloads}/${oauth_archive}" -C "${proposed_bin}" --strip-components=1 oauth2-proxy-v7.15.3.linux-amd64/oauth2-proxy
caddy_binary_sha256="$(sha256_file "${proposed_bin}/caddy")"
oauth_binary_sha256="$(sha256_file "${proposed_bin}/oauth2-proxy")"

raw_index="${downloads}/keycloak-index.json"
skopeo inspect --raw "docker://${keycloak_ref}" >"${raw_index}"
index_digest="sha256:$(sha256_file "${raw_index}")"
amd64_digest="$(jq -er '.manifests[] | select(.platform.os == "linux" and .platform.architecture == "amd64") | .digest' "${raw_index}")"
[[ "$(printf '%s\n' "${amd64_digest}" | wc -l)" -eq 1 ]] || fail 4 "Keycloak amd64 manifest is not unique"
[[ "${amd64_digest}" =~ ^sha256:[a-f0-9]{64}$ ]] || fail 4 "Keycloak amd64 digest malformed"

raw_manifest="${downloads}/keycloak-amd64-manifest.json"
skopeo inspect --raw "docker://quay.io/keycloak/keycloak@${amd64_digest}" >"${raw_manifest}"
[[ "sha256:$(sha256_file "${raw_manifest}")" == "${amd64_digest}" ]] || fail 4 "Keycloak platform manifest digest mismatch"
config_digest="$(jq -er '.config.digest' "${raw_manifest}")"
[[ "${config_digest}" =~ ^sha256:[a-f0-9]{64}$ ]] || fail 4 "Keycloak configuration digest malformed"
image_metadata="${downloads}/keycloak-amd64-inspect.json"
skopeo inspect "docker://quay.io/keycloak/keycloak@${amd64_digest}" >"${image_metadata}"
created="$(jq -er '.Created' "${image_metadata}")"

jq -n \
  --arg caddy_sha256 "${caddy_expected}" \
  --arg caddy_binary_sha256 "${caddy_binary_sha256}" \
  --arg oauth_sha256 "${oauth_expected}" \
  --arg oauth_binary_sha256 "${oauth_binary_sha256}" \
  --arg index_digest "${index_digest}" \
  --arg amd64_digest "${amd64_digest}" \
  --arg config_digest "${config_digest}" \
  --arg created "${created}" \
  '{
    schema_version: 1,
    review_status: "PROPOSED",
    platform: {os_release: "Ubuntu 24.04 LTS", architecture: "amd64"},
    caddy: {version: "2.11.4", sha256: $caddy_sha256, binary_sha256: $caddy_binary_sha256, provenance_verified: true,
      sigstore_identity: "UNRESOLVED",
      sigstore_issuer: "https://token.actions.githubusercontent.com", transparency_log_verified: true,
      source: "https://github.com/caddyserver/caddy/releases/tag/v2.11.4"},
    oauth2_proxy: {version: "7.15.3", sha256: $oauth_sha256, binary_sha256: $oauth_binary_sha256, official_checksum_verified: true,
      source: "https://github.com/oauth2-proxy/oauth2-proxy/releases/tag/v7.15.3"},
    keycloak: {version: "26.7.0", source_reference: "quay.io/keycloak/keycloak:26.7.0",
      oci_index_digest: $index_digest, amd64_manifest_digest: $amd64_digest,
      config_digest: $config_digest, os: "linux", architecture: "amd64", created: $created,
      provenance: "https://www.keycloak.org/2026/07/keycloak-2670-released"}
  }' >"${AERODOCS_SPIKE_RUNTIME}/proposed-provenance.json"
chmod 600 "${AERODOCS_SPIKE_RUNTIME}/proposed-provenance.json"
printf '%s\n' "Proposed provenance written for independent review."
printf '%s\n' "Do not execute the spike until review_status is APPROVED in a separate runtime copy."
exit 20
