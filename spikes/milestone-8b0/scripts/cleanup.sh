#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
# shellcheck source=common.sh
source "$(dirname -- "$0")/common.sh"

require_runtime_root
cleanup_failed=0
resource_attempts=0
manifest="${AERODOCS_SPIKE_RESOURCE_MANIFEST:-}"
cleanup_phase=controller_final_teardown
if [[ "${AERODOCS_SPIKE_BOOTSTRAP_TEST_MODE:-0}" == 1 ]]; then cleanup_phase=child_cleanup; fi

if [[ -n "${manifest}" ]]; then
  [[ -f "${manifest}" && ! -L "${manifest}" && "$(mode_bits "${manifest}")" == 600 ]] || cleanup_failed=1
  python3 "${SPIKE_DIR}/scripts/resource_manifest.py" "${manifest}" || cleanup_failed=1
  if [[ "${cleanup_failed}" -eq 0 ]] && command -v podman >/dev/null 2>&1; then
    while IFS=$'\t' read -r kind identifier label purpose creator; do
      if [[ "${cleanup_phase}" == child_cleanup ]]; then
        [[ "${purpose}" == interruption_target && "${creator}" == child ]] || { cleanup_failed=1; continue; }
      else
        [[ "${purpose}" =~ ^(unrelated_preservation_control|controller_test_fixture)$ && "${creator}" == controller ]] || { cleanup_failed=1; continue; }
      fi
      if podman "${kind}" exists "${identifier}"; then
        actual_label="$(podman "${kind}" inspect "${identifier}" --format '{{ index .Labels "aerodocs.m8b0.run" }}' 2>/dev/null)" || actual_label=""
        if [[ "${actual_label}" != "${label}" ]]; then
          cleanup_failed=1
          continue
        fi
        resource_attempts=$((resource_attempts + 1))
        podman "${kind}" rm "${identifier}" >/dev/null || cleanup_failed=1
      fi
    done < <(python3 "${SPIKE_DIR}/scripts/resource_manifest.py" "${manifest}" --phase "${cleanup_phase}")
  fi
fi

for child in downloads bin secrets caddy-data caddy-config browser-profile; do
  target="${AERODOCS_SPIKE_RUNTIME}/${child}"
  [[ ! -e "${target}" && ! -L "${target}" ]] || rm -rf -- "${target}" || cleanup_failed=1
done
for file in approved-provenance.json proposed-provenance.json resource-manifest.json aerodocs-spike-realm.runtime.json caddy.log mock.log oauth2-proxy.log auth-mock.log oidc-gate.json browser-flow.json envelope-gate.json authorization-gate.json podman-interruption.json environment-gate.json configuration-hashes.json json-error caddy.adapted.json caddy-stock-modules.txt caddy-stock-modules.sha256 gate-state.json endpoints.json; do
  target="${AERODOCS_SPIKE_RUNTIME}/${file}"
  [[ ! -e "${target}" && ! -L "${target}" ]] || rm -f -- "${target}" || cleanup_failed=1
done
rm -f -- "${AERODOCS_SPIKE_RUNTIME}/.sentinel" || cleanup_failed=1
rmdir -- "${AERODOCS_SPIKE_RUNTIME}" || cleanup_failed=1

if [[ "${AERODOCS_SPIKE_BOOTSTRAP_TEST_MODE:-0}" == 1 && "${AERODOCS_SPIKE_TEST_FORCE_CLEANUP_FAILURE:-0}" == 1 ]]; then cleanup_failed=1; fi
if [[ "${AERODOCS_SPIKE_BOOTSTRAP_TEST_MODE:-0}" == 1 ]]; then
  printf '{"schema_version":1,"event":"child_cleanup","dispatch_count":1,"resource_attempt_count":%s,"result":"%s"}\n' \
    "${resource_attempts}" "$([[ "${cleanup_failed}" -eq 0 ]] && printf PASS || printf FAIL)"
fi
[[ "${cleanup_failed}" -eq 0 ]] || { printf 'FAIL: cleanup incomplete\n' >&2; exit 1; }
printf '%s\n' "Stage 8B-0 runtime artifacts removed."
