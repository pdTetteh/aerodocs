#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
# shellcheck source=common.sh
source "$(dirname -- "$0")/common.sh"

require_ubuntu_amd64
require_runtime_root
for command_name in jq podman sha256sum tar; do
  require_command "${command_name}"
done
tar -tzf "${downloads}/caddy_2.11.4_linux_amd64.tar.gz" | awk '$0 !~ /^(caddy|LICENSE|README\.md)$/ {bad=1} END {exit bad}' || fail 5 "Caddy archive contains an unexpected member"
tar -tzf "${downloads}/oauth2-proxy-v7.15.3.linux-amd64.tar.gz" | awk '$0 !~ /^oauth2-proxy-v7\.15\.3\.linux-amd64\/(oauth2-proxy|LICENSE|README\.md|CHANGELOG\.md)?$/ {bad=1} END {exit bad}' || fail 5 "OAuth2 Proxy archive contains an unexpected member"

approval="${1:-${AERODOCS_SPIKE_RUNTIME}/approved-provenance.json}"
[[ -f "${approval}" ]] || fail 5 "approved provenance file missing"
[[ "$(stat -c '%a' "${approval}")" == "600" ]] || fail 5 "approved provenance must have mode 0600"
[[ "$(jq -er '.review_status' "${approval}")" == "APPROVED" ]] || fail 5 "provenance is not approved"
[[ "$(jq -er '.keycloak.version' "${approval}")" == "26.7.0" ]] || fail 5 "Keycloak version not approved"
[[ "$(jq -er '.keycloak.source_reference' "${approval}")" == "quay.io/keycloak/keycloak:26.7.0" ]] || fail 5 "Keycloak source is not official"
[[ "$(jq -er '.caddy.sigstore_identity' "${approval}")" =~ ^https://github\.com/caddyserver/caddy/\.github/workflows/[^@]+@refs/tags/v2\.11\.4$ ]] || fail 5 "Caddy identity not approved"
[[ "$(jq -er '.caddy.sigstore_issuer' "${approval}")" == "https://token.actions.githubusercontent.com" && "$(jq -er '.caddy.transparency_log_verified' "${approval}")" == true ]] || fail 5 "Caddy Sigstore provenance not approved"
amd64_digest="$(jq -er '.keycloak.amd64_manifest_digest' "${approval}")"
[[ "${amd64_digest}" =~ ^sha256:[a-f0-9]{64}$ ]] || fail 5 "approved Keycloak digest malformed"
index_digest="$(jq -er '.keycloak.oci_index_digest' "${approval}")"
config_digest="$(jq -er '.keycloak.config_digest' "${approval}")"
[[ "${index_digest}" =~ ^sha256:[a-f0-9]{64}$ && "${config_digest}" =~ ^sha256:[a-f0-9]{64}$ ]] || fail 5 "approved Keycloak provenance malformed"
raw_index="${AERODOCS_SPIKE_RUNTIME}/downloads/approved-keycloak-index.json"
skopeo inspect --raw "docker://quay.io/keycloak/keycloak@${index_digest}" >"${raw_index}"
[[ "sha256:$(sha256_file "${raw_index}")" == "${index_digest}" ]] || fail 5 "approved Keycloak index digest mismatch"
[[ "$(jq -r --arg digest "${amd64_digest}" '[.manifests[] | select(.digest == $digest and .platform.os == "linux" and .platform.architecture == "amd64")] | length' "${raw_index}")" == 1 ]] || fail 5 "approved Keycloak platform is not bound by index"
raw_manifest="${AERODOCS_SPIKE_RUNTIME}/downloads/approved-keycloak-manifest.json"
skopeo inspect --raw "docker://quay.io/keycloak/keycloak@${amd64_digest}" >"${raw_manifest}"
[[ "sha256:$(sha256_file "${raw_manifest}")" == "${amd64_digest}" ]] || fail 5 "approved Keycloak manifest digest mismatch"
[[ "$(jq -er '.config.digest' "${raw_manifest}")" == "${config_digest}" ]] || fail 5 "approved Keycloak configuration digest mismatch"

downloads="${AERODOCS_SPIKE_RUNTIME}/downloads"
[[ "$(sha256_file "${downloads}/caddy_2.11.4_linux_amd64.tar.gz")" == "$(jq -er '.caddy.sha256' "${approval}")" ]] || fail 5 "Caddy approval mismatch"
[[ "$(sha256_file "${downloads}/oauth2-proxy-v7.15.3.linux-amd64.tar.gz")" == "$(jq -er '.oauth2_proxy.sha256' "${approval}")" ]] || fail 5 "OAuth2 Proxy approval mismatch"

image_ref="quay.io/keycloak/keycloak@${amd64_digest}"
podman pull --arch amd64 --os linux "${image_ref}" >/dev/null
local_digest="$(podman image inspect "${image_ref}" --format '{{.Digest}}')"
local_arch="$(podman image inspect "${image_ref}" --format '{{.Architecture}}')"
local_os="$(podman image inspect "${image_ref}" --format '{{.Os}}')"
local_created="$(podman image inspect "${image_ref}" --format '{{.Created}}')"
[[ "${local_digest}" == "${amd64_digest}" ]] || fail 5 "pulled Keycloak digest differs from approval"
[[ "${local_arch}" == "amd64" ]] || fail 5 "pulled Keycloak architecture differs from approval"
[[ "${local_os}" == "linux" ]] || fail 5 "pulled Keycloak OS differs from approval"
[[ "${local_created}" == "$(jq -er '.keycloak.created' "${approval}")" ]] || fail 5 "pulled Keycloak creation metadata differs from approval"

bin_dir="${AERODOCS_SPIKE_RUNTIME}/bin"
mkdir -m 700 -p "${bin_dir}"
for archive in "${downloads}/caddy_2.11.4_linux_amd64.tar.gz" "${downloads}/oauth2-proxy-v7.15.3.linux-amd64.tar.gz"; do
  tar -tzf "${archive}" | awk 'BEGIN {bad=0} /^\// || /(^|\/)\.\.($|\/)/ {bad=1} END {exit bad}' || fail 5 "archive contains an unsafe path"
  tar -tvzf "${archive}" | awk 'substr($1,1,1) !~ /[-d]/ {bad=1} END {exit bad}' || fail 5 "archive contains a link or unsupported member"
done
[[ "$(tar -tzf "${downloads}/caddy_2.11.4_linux_amd64.tar.gz" | awk '$0 == "caddy" {n++} END {print n+0}')" == 1 ]] || fail 5 "Caddy binary archive entry is not unique"
oauth_entry="oauth2-proxy-v7.15.3.linux-amd64/oauth2-proxy"
[[ "$(tar -tzf "${downloads}/oauth2-proxy-v7.15.3.linux-amd64.tar.gz" | awk -v entry="${oauth_entry}" '$0 == entry {n++} END {print n+0}')" == 1 ]] || fail 5 "OAuth2 Proxy binary archive entry is not unique"
tar -xzf "${downloads}/caddy_2.11.4_linux_amd64.tar.gz" -C "${bin_dir}" caddy
tar -xzf "${downloads}/oauth2-proxy-v7.15.3.linux-amd64.tar.gz" -C "${bin_dir}" \
  --strip-components=1 oauth2-proxy-v7.15.3.linux-amd64/oauth2-proxy
chmod 700 "${bin_dir}/caddy" "${bin_dir}/oauth2-proxy"
[[ "$("${bin_dir}/caddy" version)" == "v2.11.4 h1:"* ]] || fail 5 "Caddy version mismatch"
[[ "$("${bin_dir}/oauth2-proxy" --version 2>&1)" =~ ^oauth2-proxy\ v7\.15\.3([[:space:]]|$) ]] || fail 5 "OAuth2 Proxy version mismatch"
[[ "$(sha256_file "${bin_dir}/caddy")" == "$(jq -er '.caddy.binary_sha256' "${approval}")" ]] || fail 5 "Caddy binary checksum differs from approval"
[[ "$(sha256_file "${bin_dir}/oauth2-proxy")" == "$(jq -er '.oauth2_proxy.binary_sha256' "${approval}")" ]] || fail 5 "OAuth2 Proxy binary checksum differs from approval"
module_inventory="${AERODOCS_SPIKE_RUNTIME}/caddy-stock-modules.txt"
"${bin_dir}/caddy" list-modules --packages | LC_ALL=C sort >"${module_inventory}"
[[ -s "${module_inventory}" ]] || fail 5 "Caddy module inventory is empty"
awk 'NF >= 2 && $NF !~ /^github\.com\/caddyserver\/caddy\/v2\// {bad=1} END {exit bad}' "${module_inventory}" || fail 5 "Caddy contains an unreviewed external module package"
sha256_file "${module_inventory}" >"${AERODOCS_SPIKE_RUNTIME}/caddy-stock-modules.sha256"
printf '%s\n' "Verified approved binaries and immutable Keycloak image."
