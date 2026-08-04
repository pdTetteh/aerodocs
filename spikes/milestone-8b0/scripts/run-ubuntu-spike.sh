#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
# shellcheck source=common.sh
source "$(dirname -- "$0")/common.sh"

AERODOCS_SPIKE_RUNTIME=""
AERODOCS_SPIKE_REVIEW=""
gate_state=""
live_started=0
ACTIVE_GATE="ubuntu_version"
interrupt_checkpoint() {
  [[ "${AERODOCS_SPIKE_TEST_INTERRUPT_AFTER:-}" != "$1" ]] || {
    if [[ "${AERODOCS_SPIKE_BOOTSTRAP_TEST_MODE:-0}" == 1 ]]; then
      local resource_kind created_count
      case "$1" in
        before_runtime) resource_kind=none; created_count=0 ;;
        after_runtime) [[ -d "${AERODOCS_SPIKE_RUNTIME}" && ! -e "${AERODOCS_SPIKE_RUNTIME}/.sentinel" ]] || exit 2; resource_kind=runtime; created_count=1 ;;
        after_sentinel) [[ -f "${AERODOCS_SPIKE_RUNTIME}/.sentinel" ]] || exit 2; resource_kind=runtime_sentinel; created_count=2 ;;
        after_review) [[ -d "${AERODOCS_SPIKE_REVIEW}" && ! -e "${AERODOCS_SPIKE_REVIEW}/.sentinel" ]] || exit 2; resource_kind=review; created_count=3 ;;
        after_review_sentinel) [[ -f "${AERODOCS_SPIKE_REVIEW}/.sentinel" ]] || exit 2; resource_kind=review_sentinel; created_count=4 ;;
        after_gate_state) [[ -f "${gate_state}" ]] || exit 2; resource_kind=gate_state; created_count=5 ;;
        after_first_download) [[ -f "${AERODOCS_SPIKE_RUNTIME}/downloads/bootstrap-test" ]] || exit 2; resource_kind=download; created_count=6 ;;
        after_first_container_resource) [[ -f "${AERODOCS_SPIKE_RUNTIME}/caddy-data/bootstrap-test" ]] || exit 2; resource_kind=synthetic_container_like; created_count=7 ;;
        after_actual_podman_resource) podman network exists "${AERODOCS_SPIKE_PODMAN_NETWORK}" && podman volume exists "${AERODOCS_SPIKE_PODMAN_VOLUME}" || exit 2; resource_kind=actual_podman; created_count=9 ;;
        *) exit 2 ;;
      esac
      python3 - "$1" "${resource_kind}" "${created_count}" "${AERODOCS_SPIKE_TEST_READY_FD}" <<'PY'
import json
import os
import struct
import sys

payload = json.dumps(
    {
        "schema_version": 1,
        "checkpoint": sys.argv[1],
        "resource_kind": sys.argv[2],
        "created_resource_count": int(sys.argv[3]),
        "state": "BLOCKED",
    },
    separators=(",", ":"),
).encode("utf-8")
if len(payload) > 1024:
    raise SystemExit(2)
descriptor = int(sys.argv[4])
frame = struct.pack(">I", len(payload)) + payload
while frame:
    written = os.write(descriptor, frame)
    frame = frame[written:]
os.close(descriptor)
PY
      eval "exec ${AERODOCS_SPIKE_TEST_READY_FD}>&-"
      IFS= read -r checkpoint_control <&"${AERODOCS_SPIKE_TEST_CONTROL_FD}" || exit 143
      [[ "${checkpoint_control}" == RELEASE ]] || exit 2
      return
    fi
    kill -TERM "$$"
  }
}
cleanup_on_exit() {
  local original_status=$?
  local cleanup_status=0
  trap - EXIT INT TERM
  if [[ "${explicit_cleanup_complete:-0}" -eq 1 ]]; then
    exit "${original_status}"
  fi
  if [[ "${original_status}" -ne 0 && "${live_started}" -eq 1 && -n "${gate_state}" && -f "${gate_state}" ]]; then
    python3 "${SPIKE_DIR}/scripts/build_failure_artifacts.py" \
      --spike "${SPIKE_DIR}" --state "${gate_state}" --gate "${ACTIVE_GATE}" \
      --failure-code INCONCLUSIVE --output "${AERODOCS_SPIKE_REVIEW}" || cleanup_status=1
  fi
  if declare -F cleanup_processes >/dev/null 2>&1; then
    cleanup_processes || cleanup_status=1
  fi
  if [[ -n "${AERODOCS_SPIKE_RUNTIME}" && -d "${AERODOCS_SPIKE_RUNTIME}" ]]; then
    if [[ -n "${AERODOCS_SPIKE_NONCE:-}" ]]; then
      bash "${SPIKE_DIR}/scripts/cleanup.sh" || cleanup_status=1
    elif [[ "${AERODOCS_SPIKE_RUNTIME_CREATED:-0}" -eq 1 ]]; then
      bootstrap_remove_empty_child "${AERODOCS_SPIKE_RUNTIME}" "${FIXED_RUNTIME_PARENT}" run || cleanup_status=1
    fi
  fi
  if [[ "${AERODOCS_SPIKE_BOOTSTRAP_TEST_MODE:-0}" == 1 ]]; then
    if [[ "${cleanup_status}" -eq 0 ]]; then cleanup_result=PASS; else cleanup_result=FAIL; fi
    printf '{"schema_version":1,"event":"cleanup_complete","result":"%s"}\n' "${cleanup_result}"
  fi
  if [[ "${cleanup_status}" -ne 0 ]]; then
    if [[ -n "${gate_state}" && -f "${gate_state}" && -n "${AERODOCS_SPIKE_REVIEW}" ]]; then
      python3 "${SPIKE_DIR}/scripts/build_failure_artifacts.py" \
        --spike "${SPIKE_DIR}" --state "${gate_state}" --gate cleanup \
        --failure-code CLEANUP --output "${AERODOCS_SPIKE_REVIEW}" || exit 1
    fi
    exit 1
  fi
  if [[ "${live_started}" -eq 1 && -n "${gate_state}" && -f "${gate_state}" ]]; then
    python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set cleanup PASS || exit 1
    if [[ "${original_status}" -ne 0 ]]; then
      python3 "${SPIKE_DIR}/scripts/build_failure_artifacts.py" \
        --spike "${SPIKE_DIR}" --state "${gate_state}" --gate "${ACTIVE_GATE}" \
        --failure-code INCONCLUSIVE --output "${AERODOCS_SPIKE_REVIEW}" || exit 1
    fi
  fi
  [[ -z "${gate_state}" || ! -f "${gate_state}" ]] || rm -f -- "${gate_state}"
  if [[ -n "${AERODOCS_SPIKE_REVIEW}" ]] && validate_review_root; then
    rm -f -- "${AERODOCS_SPIKE_REVIEW}/.sentinel"
    if [[ -z "$(find "${AERODOCS_SPIKE_REVIEW}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      rmdir -- "${AERODOCS_SPIKE_REVIEW}" || exit 1
    fi
  elif [[ "${AERODOCS_SPIKE_REVIEW_CREATED:-0}" -eq 1 && -n "${AERODOCS_SPIKE_REVIEW}" ]]; then
    bootstrap_remove_empty_child "${AERODOCS_SPIKE_REVIEW}" "${FIXED_REVIEW_PARENT}" review || exit 1
  fi
  exit "${original_status}"
}
trap cleanup_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${AERODOCS_SPIKE_BOOTSTRAP_TEST_MODE:-0}" == 1 ]]; then
  test_parent="${AERODOCS_SPIKE_TEST_PARENT:-}"
  case "$(uname -s)" in Darwin) test_base=/private/tmp ;; Linux) test_base=/tmp ;; *) fail 2 "unsupported bootstrap test platform" ;; esac
  python3 -c 'import os,stat,sys; p=sys.argv[1]; s=os.lstat(p); assert os.path.realpath(p) == p and stat.S_ISDIR(s.st_mode) and not stat.S_ISLNK(s.st_mode) and s.st_mode & stat.S_ISVTX and s.st_mode & stat.S_IWOTH' "${test_base}" || fail 2 "unsafe canonical bootstrap test base"
  test_root="${test_base}/aerodocs-m8b0-bootstrap-tests-${UID}"
  [[ -d "${test_root}" && ! -L "${test_root}" && "${test_root}" == /* ]] || fail 2 "unsafe fixed bootstrap test root"
  [[ "$(CDPATH= cd -- "${test_root}" && pwd -P)" == "${test_root}" ]] || fail 2 "fixed bootstrap test root is not canonical"
  python3 -c 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); assert all(not q.is_symlink() for q in (p, *p.parents))' "${test_root}" || fail 2 "fixed bootstrap test root chain contains a symlink"
  [[ "$(owner_id "${test_root}")" == "$(id -u)" && "$(mode_bits "${test_root}")" == 700 ]] || fail 2 "fixed bootstrap test root ownership or mode is unsafe"
  [[ "${test_root}" != / && "${test_root}" != "${HOME}" && "${test_root}" != "${REPO_DIR}" && "${test_root}" != "${FIXED_RUNTIME_PARENT}" ]] || fail 2 "fixed bootstrap test root is protected"
  [[ "${HOME}/" != "${test_root}/"* && "${test_root}/" != "${HOME}/"* ]] || fail 2 "fixed bootstrap test root overlaps home"
  [[ "${REPO_DIR}/" != "${test_root}/"* && "${test_root}/" != "${REPO_DIR}/"* ]] || fail 2 "fixed bootstrap test root overlaps repository"
  [[ "${AERODOCS_SPIKE_TEST_NONCE_FD:-}" =~ ^[0-9]+$ ]] || fail 2 "bootstrap nonce descriptor invalid"
  [[ "${AERODOCS_SPIKE_TEST_ROOT_NONCE_FD:-}" =~ ^[0-9]+$ ]] || fail 2 "bootstrap root nonce descriptor invalid"
  IFS= read -r root_nonce <&"${AERODOCS_SPIKE_TEST_ROOT_NONCE_FD}" || fail 2 "bootstrap root nonce unavailable"
  [[ "${root_nonce}" =~ ^[a-f0-9]{64}$ && -f "${test_root}/.aerodocs-bootstrap-test-root" && ! -L "${test_root}/.aerodocs-bootstrap-test-root" && "$(mode_bits "${test_root}/.aerodocs-bootstrap-test-root")" == 600 ]] || fail 2 "bootstrap root sentinel missing"
  root_sentinel_json="$(< "${test_root}/.aerodocs-bootstrap-test-root")"
  python3 -c 'import json,sys; v=json.loads(sys.argv[1]); assert v == {"schema_version":1,"purpose":"aerodocs-m8b0-bootstrap-root","nonce":sys.argv[2]}' "${root_sentinel_json}" "${root_nonce}" || fail 2 "bootstrap root sentinel invalid"
  IFS= read -r test_nonce <&"${AERODOCS_SPIKE_TEST_NONCE_FD}" || fail 2 "bootstrap nonce unavailable"
  [[ ! -e "/proc/$$/fd/${AERODOCS_SPIKE_TEST_NONCE_FD}" || -r "/proc/$$/fd/${AERODOCS_SPIKE_TEST_NONCE_FD}" ]] || fail 2 "bootstrap nonce descriptor invalid"
  [[ -n "${test_parent}" && "${test_parent}" == /* && -d "${test_parent}" && ! -L "${test_parent}" ]] || fail 2 "unsafe bootstrap test parent"
  [[ "$(dirname -- "${test_parent}")" == "${test_root}" && "${test_parent}" != "${test_root}" ]] || fail 2 "bootstrap test parent is not an immediate child"
  [[ "$(CDPATH= cd -- "${test_parent}" && pwd -P)" == "${test_parent}" ]] || fail 2 "bootstrap test parent is not canonical"
  [[ "$(owner_id "${test_parent}")" == "$(id -u)" && "$(mode_bits "${test_parent}")" == 700 ]] || fail 2 "bootstrap test parent ownership or mode is unsafe"
  [[ "${test_nonce}" =~ ^[a-f0-9]{64}$ && -f "${test_parent}/.aerodocs-bootstrap-test-parent" && ! -L "${test_parent}/.aerodocs-bootstrap-test-parent" ]] || fail 2 "bootstrap test parent sentinel missing"
  sentinel_json="$(< "${test_parent}/.aerodocs-bootstrap-test-parent")"
  [[ "$(mode_bits "${test_parent}/.aerodocs-bootstrap-test-parent")" == 600 ]] || fail 2 "bootstrap test parent sentinel mode invalid"
  python3 -c 'import json,sys; v=json.loads(sys.argv[1]); assert v == {"schema_version":1,"purpose":"aerodocs-m8b0-bootstrap-test","nonce":sys.argv[2]}' "${sentinel_json}" "${test_nonce}" || fail 2 "bootstrap test parent sentinel invalid"
  while IFS= read -r initial_name; do
    case "${initial_name}" in
      .aerodocs-bootstrap-test-parent|unrelated-file|unrelated-directory|runtime-parent-similar|runtime-parent-link|symlink-target) ;;
      *) fail 2 "bootstrap test parent contains an unregistered fixture" ;;
    esac
  done < <(find "${test_parent}" -mindepth 1 -maxdepth 1 -exec basename {} \;)
  case "${REPO_DIR}/" in "${test_parent}/"*) fail 2 "bootstrap test parent contains repository" ;; esac
  [[ "${test_parent}" != "${HOME}" && "${test_parent}/" != "${HOME}/"* && "${HOME}/" != "${test_parent}/"* ]] || fail 2 "bootstrap test parent overlaps home"
  [[ "${AERODOCS_SPIKE_TEST_READY_FD:-}" =~ ^[0-9]+$ && "${AERODOCS_SPIKE_TEST_CONTROL_FD:-}" =~ ^[0-9]+$ ]] || fail 2 "bootstrap test descriptors invalid"
  FIXED_RUNTIME_PARENT="${test_parent}/runtime-parent"
  FIXED_REVIEW_PARENT="${test_parent}/review-parent"
  export FIXED_RUNTIME_PARENT FIXED_REVIEW_PARENT
  install -d -m 700 "${FIXED_RUNTIME_PARENT}" "${FIXED_REVIEW_PARENT}"
  interrupt_checkpoint before_runtime
  create_runtime_root
  create_review_root
  gate_state="${AERODOCS_SPIKE_REVIEW}/.gate-state.json"
  python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --init
  interrupt_checkpoint after_gate_state
  install -d -m 700 "${AERODOCS_SPIKE_RUNTIME}/downloads"
  printf '%s\n' synthetic >"${AERODOCS_SPIKE_RUNTIME}/downloads/bootstrap-test"
  interrupt_checkpoint after_first_download
  install -d -m 700 "${AERODOCS_SPIKE_RUNTIME}/caddy-data"
  printf '%s\n' synthetic >"${AERODOCS_SPIKE_RUNTIME}/caddy-data/bootstrap-test"
  interrupt_checkpoint after_first_container_resource
  if [[ "${AERODOCS_SPIKE_ACTUAL_PODMAN_TEST_MODE:-0}" == 1 ]]; then
    require_ubuntu_amd64
    require_command podman
    manifest="${AERODOCS_SPIKE_RESOURCE_MANIFEST:?}"
    python3 "${SPIKE_DIR}/scripts/resource_manifest.py" "${manifest}" || fail 2 "resource manifest invalid"
    while IFS=$'\t' read -r kind identifier label purpose creator; do
      [[ "${purpose}" == interruption_target && "${creator}" == child && "${kind}" =~ ^(network|volume)$ ]] || fail 2 "interruption manifest policy invalid"
      podman "${kind}" exists "${identifier}" && fail 2 "authorized Podman resource already exists"
      podman "${kind}" create --label "aerodocs.m8b0.run=${label}" "${identifier}" >/dev/null
    done < <(python3 "${SPIKE_DIR}/scripts/resource_manifest.py" "${manifest}" --phase child_cleanup)
    interrupt_checkpoint after_actual_podman_resource
  fi
  exit 0
fi

require_ubuntu_amd64
for command_name in certutil curl getcap getent jq npm openssl podman python3 skopeo ss uv; do
  require_command "${command_name}"
done
interrupt_checkpoint before_runtime
create_runtime_root
create_review_root
gate_state="${AERODOCS_SPIKE_REVIEW}/.gate-state.json"
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --init
interrupt_checkpoint after_gate_state
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set ubuntu_version PASS
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set ubuntu_architecture PASS
endpoint_record="${AERODOCS_SPIKE_RUNTIME}/endpoints.json"
printf '%s\n' '{"caddy_url":"https://aerodocs-spike.invalid","mock_control_url":"http://127.0.0.1:18000","oauth_control_url":"http://127.0.0.1:14180","keycloak_port":18080,"mock_port":18000,"oauth_port":14180}' >"${endpoint_record}"
chmod 600 "${endpoint_record}"
mock_port="$(jq -er '.mock_port' "${endpoint_record}")"
oauth_port="$(jq -er '.oauth_port' "${endpoint_record}")"
keycloak_port="$(jq -er '.keycloak_port' "${endpoint_record}")"
caddy_url="$(jq -er '.caddy_url' "${endpoint_record}")"
mock_control_url="$(jq -er '.mock_control_url' "${endpoint_record}")"
oauth_control_url="$(jq -er '.oauth_control_url' "${endpoint_record}")"

approval="${AERODOCS_SPIKE_RUNTIME}/approved-provenance.json"
if [[ -z "${AERODOCS_SPIKE_APPROVAL:-}" ]]; then
  set +e
  "${SPIKE_DIR}/scripts/resolve-provenance.sh"
  provenance_status=$?
  set -e
  [[ "${provenance_status}" -eq 20 ]] || fail 4 "provenance resolution failed"
  cp -- "${AERODOCS_SPIKE_RUNTIME}/proposed-provenance.json" "${AERODOCS_SPIKE_REVIEW}/proposed-provenance.json"
  chmod 600 "${AERODOCS_SPIKE_REVIEW}/proposed-provenance.json"
  printf '%s\n' "NOT_YET_EXECUTED: provenance approval required; review output: ${AERODOCS_SPIKE_REVIEW}"
  exit 20
fi
[[ -f "${AERODOCS_SPIKE_APPROVAL}" && ! -L "${AERODOCS_SPIKE_APPROVAL}" ]] || fail 5 "approval file is missing or a symlink"
cp -- "${AERODOCS_SPIKE_APPROVAL}" "${approval}"
chmod 600 "${approval}"

live_started=1
ACTIVE_GATE="caddy_provenance"
"${SPIKE_DIR}/scripts/verify-provenance.sh" "${approval}"
for gate in caddy_provenance caddy_exact_version caddy_stock_modules oauth2_proxy_provenance oauth2_proxy_exact_version keycloak_index_digest keycloak_platform_manifest_digest keycloak_config_digest keycloak_os keycloak_architecture; do
  python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set "${gate}" PASS
done

ACTIVE_GATE="podman_interruption_cleanup"
podman_evidence="${AERODOCS_SPIKE_RUNTIME}/podman-interruption.json"
set +e
python3 "${SPIKE_DIR}/scripts/probe_podman_interruption.py" \
  --podman "$(command -v podman)" \
  --harness "${SPIKE_DIR}/scripts/run-ubuntu-spike.sh" \
  --output "${podman_evidence}"
podman_probe_status=$?
set -e
[[ -s "${podman_evidence}" ]] || fail 6 "Podman interruption evidence missing"
python3 "${SPIKE_DIR}/scripts/validate_artifact.py" \
  "${SPIKE_DIR}/schemas/evidence.schema.json" "${podman_evidence}" >/dev/null ||
  fail 6 "Podman interruption evidence invalid"
python3 "${SPIKE_DIR}/scripts/validate_artifact.py" --scan-only "${podman_evidence}" >/dev/null ||
  fail 6 "Podman interruption evidence secret scan failed"
jq -e '.gate_results | keys == ["podman_interruption_cleanup"]' "${podman_evidence}" >/dev/null ||
  fail 6 "Podman interruption evidence gate mismatch"
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --apply-result "${podman_evidence}" ||
  fail 6 "Podman interruption result conflicts with registry"
[[ "${podman_probe_status}" -eq 0 ]] && jq -e '.result == "PASS" and .gate_results.podman_interruption_cleanup.result == "PASS"' "${podman_evidence}" >/dev/null ||
  fail 6 "Podman interruption gate failed"

install -d -m 700 "${AERODOCS_SPIKE_RUNTIME}/secrets"
read -r -p "Synthetic username: " synthetic_username
read -r -s -p "Synthetic user password: " synthetic_user_password
printf '\n' >&2
read -r -s -p "Synthetic OAuth client secret: " synthetic_client_secret
printf '\n' >&2
read -r -s -p "Synthetic Keycloak admin password: " synthetic_admin_password
printf '\n' >&2
umask 077
printf '%s\n' "${synthetic_username}" >"${AERODOCS_SPIKE_RUNTIME}/secrets/username"
printf '%s\n' "${synthetic_user_password}" >"${AERODOCS_SPIKE_RUNTIME}/secrets/user-password"
printf '%s\n' "${synthetic_client_secret}" >"${AERODOCS_SPIKE_RUNTIME}/secrets/client-secret"
printf '%s\n' "${synthetic_admin_password}" >"${AERODOCS_SPIKE_RUNTIME}/secrets/admin-password"
unset synthetic_username synthetic_user_password synthetic_client_secret synthetic_admin_password

for secret_name in admin-password client-secret user-password username; do
  secret_file="${AERODOCS_SPIKE_RUNTIME}/secrets/${secret_name}"
  [[ -f "${secret_file}" ]] || fail 6 "runtime secret file missing"
  [[ "$(stat -c '%a' "${secret_file}")" == "600" ]] || fail 6 "runtime secret file mode must be 0600"
  [[ -s "${secret_file}" ]] || fail 6 "runtime secret file is empty"
done

for host in aerodocs-spike.invalid idp.aerodocs-spike.invalid; do
  addresses="$(getent ahostsv4 "${host}" | awk '{print $1}' | sort -u)"
  [[ "${addresses}" == "127.0.0.1" ]] || fail 6 "synthetic hostname must resolve only to loopback"
done

caddy_bin="${AERODOCS_SPIKE_RUNTIME}/bin/caddy"
oauth_bin="${AERODOCS_SPIKE_RUNTIME}/bin/oauth2-proxy"
getcap "${caddy_bin}" | grep -q 'cap_net_bind_service=ep' ||
  fail 6 "runtime Caddy binary requires only cap_net_bind_service before launch"

secrets_dir="${AERODOCS_SPIKE_RUNTIME}/secrets"
cookie_secret_file="${secrets_dir}/cookie-secret"
if [[ ! -f "${cookie_secret_file}" ]]; then
  openssl rand -base64 32 >"${cookie_secret_file}"
  chmod 600 "${cookie_secret_file}"
fi

rendered_realm="${AERODOCS_SPIKE_RUNTIME}/aerodocs-spike-realm.runtime.json"
jq --rawfile client_secret "${secrets_dir}/client-secret" \
  --rawfile user_password "${secrets_dir}/user-password" '
    .clients[0].secret = ($client_secret | rtrimstr("\n")) |
    .users[0].credentials = [{type: "password", value: ($user_password | rtrimstr("\n")), temporary: false}]
  ' "${SPIKE_DIR}/keycloak/aerodocs-spike-realm.json" >"${rendered_realm}"
chmod 600 "${rendered_realm}"

amd64_digest="$(jq -er '.keycloak.amd64_manifest_digest' "${approval}")"
keycloak_image="quay.io/keycloak/keycloak@${amd64_digest}"
version_output="$(podman run --rm --arch amd64 --os linux "${keycloak_image}" --version 2>&1)"
ACTIVE_GATE="keycloak_reported_version"
[[ "${version_output}" == *"Keycloak 26.7.0"* ]] || fail 7 "Keycloak runtime version mismatch"
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set keycloak_reported_version PASS

pilot_label="${AERODOCS_SPIKE_NONCE:0:24}"
keycloak_container="aerodocs-8b0-keycloak-${pilot_label}"
keycloak_volume="aerodocs-8b0-keycloak-data-${pilot_label}"
resource_manifest="${AERODOCS_SPIKE_RUNTIME}/resource-manifest.json"
printf '%s\n' "$(jq -cn --arg label "${pilot_label}" --arg container "${keycloak_container}" --arg volume "${keycloak_volume}" '{schema_version:2,run_label:$label,resources:[{kind:"container",identifier:$container,run_label:$label,purpose:"controller_test_fixture",creator:"controller",lifecycle:"preserve_until_final_teardown",child_cleanup:false,controller_fallback:false,controller_final_teardown:true,expected_at_checkpoint:true,expected_during_verification:true,expected_final_state:"absent"},{kind:"volume",identifier:$volume,run_label:$label,purpose:"controller_test_fixture",creator:"controller",lifecycle:"preserve_until_final_teardown",child_cleanup:false,controller_fallback:false,controller_final_teardown:true,expected_at_checkpoint:true,expected_during_verification:true,expected_final_state:"absent"}]}')" >"${resource_manifest}"
chmod 600 "${resource_manifest}"
python3 "${SPIKE_DIR}/scripts/resource_manifest.py" "${resource_manifest}" || fail 7 "pilot resource manifest invalid"
export AERODOCS_SPIKE_RESOURCE_MANIFEST="${resource_manifest}"
podman volume create --label "aerodocs.m8b0.run=${pilot_label}" "${keycloak_volume}" >/dev/null
interrupt_checkpoint after_first_container_resource
podman run --detach --name "${keycloak_container}" --label "aerodocs.m8b0.run=${pilot_label}" \
  --publish "127.0.0.1:${keycloak_port}:8080" \
  --volume "${keycloak_volume}:/opt/keycloak/data:Z" \
  --volume "${rendered_realm}:/opt/keycloak/data/import/aerodocs-spike-realm.json:ro,Z" \
  --volume "${secrets_dir}/admin-password:/run/secrets/admin-password:ro,Z" \
  --entrypoint /bin/bash "${keycloak_image}" -ec '
    export KC_BOOTSTRAP_ADMIN_USERNAME=synthetic-spike-admin
    export KC_BOOTSTRAP_ADMIN_PASSWORD="$(< /run/secrets/admin-password)"
    exec /opt/keycloak/bin/kc.sh start --http-enabled=true \
      --hostname=https://idp.aerodocs-spike.invalid --hostname-strict=true \
      --proxy-headers=xforwarded --import-realm --db=dev-file \
      --health-enabled=false --metrics-enabled=false
  ' >/dev/null

if podman inspect "${keycloak_container}" --format '{{json .Args}}' | grep -Eq -- '--features($|[= -])'; then
  fail 7 "Keycloak feature override detected"
fi
for _ in {1..60}; do
  curl --silent --fail --max-time 1 "http://127.0.0.1:${keycloak_port}/" >/dev/null 2>&1 && break
  sleep 1
done
curl --silent --fail --max-time 2 "http://127.0.0.1:${keycloak_port}/" >/dev/null || fail 7 "Keycloak did not start"
ss -ltnH 'sport = :18080' | awk '{print $4}' | grep -Eq '^127\.0\.0\.1:18080$' || fail 7 "Keycloak listener is not loopback-only"
for gate in realm_import no_experimental_features loopback_isolation; do
  python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set "${gate}" PASS
done

export XDG_DATA_HOME="${AERODOCS_SPIKE_RUNTIME}/caddy-data"
export XDG_CONFIG_HOME="${AERODOCS_SPIKE_RUNTIME}/caddy-config"
mkdir -m 700 -p "${XDG_DATA_HOME}" "${XDG_CONFIG_HOME}"
ACTIVE_GATE="adapted_caddy_validation"
"${caddy_bin}" validate --config "${SPIKE_DIR}/config/Caddyfile" --adapter caddyfile >/dev/null
adapted_caddy_json="${AERODOCS_SPIKE_RUNTIME}/caddy.adapted.json"
"${caddy_bin}" adapt --config "${SPIKE_DIR}/config/Caddyfile" --adapter caddyfile --pretty >"${adapted_caddy_json}"
chmod 600 "${adapted_caddy_json}"
printf '{"adapted_caddy_json_sha256":"%s","caddyfile_sha256":"%s","oauth2_proxy_sha256":"%s","realm_sha256":"%s","stock_module_inventory_sha256":"%s"}\n' \
  "$(sha256_file "${adapted_caddy_json}")" "$(sha256_file "${SPIKE_DIR}/config/Caddyfile")" \
  "$(sha256_file "${SPIKE_DIR}/config/oauth2-proxy.cfg")" "$(sha256_file "${SPIKE_DIR}/keycloak/aerodocs-spike-realm.json")" \
  "$(< "${AERODOCS_SPIKE_RUNTIME}/caddy-stock-modules.sha256")" >"${AERODOCS_SPIKE_RUNTIME}/configuration-hashes.json"
chmod 600 "${AERODOCS_SPIKE_RUNTIME}/configuration-hashes.json"
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set adapted_caddy_validation PASS
"${caddy_bin}" run --config "${SPIKE_DIR}/config/Caddyfile" --adapter caddyfile \
  >"${AERODOCS_SPIKE_RUNTIME}/caddy.log" 2>&1 &
caddy_pid=$!

mock_pid=""
oauth_pid=""
auth_mock_pid=""
cleanup_processes() {
  for process_id in "${auth_mock_pid}" "${oauth_pid}" "${mock_pid}" "${caddy_pid}"; do
    if [[ -n "${process_id}" ]]; then
      if kill -0 "${process_id}" >/dev/null 2>&1; then
        kill "${process_id}" || return 1
      fi
      wait "${process_id}" >/dev/null 2>&1 || [[ "$?" -eq 143 ]] || return 1
    fi
  done
}

finalize_failure() {
  local failed_gate="$1"
  local failure_code="$2"
  local cleanup_failed=0
  python3 "${SPIKE_DIR}/scripts/build_failure_artifacts.py" \
    --spike "${SPIKE_DIR}" --state "${gate_state}" --gate "${failed_gate}" \
    --failure-code "${failure_code}" --output "${AERODOCS_SPIKE_REVIEW}"
  cleanup_processes || cleanup_failed=1
  mock_pid=""; oauth_pid=""; auth_mock_pid=""; caddy_pid=""
  bash "${SPIKE_DIR}/scripts/cleanup.sh" || cleanup_failed=1
  if [[ "${cleanup_failed}" -eq 0 ]]; then
    python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set cleanup PASS
  else
    python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set cleanup FAIL
    failed_gate=cleanup
    failure_code=CLEANUP
  fi
  python3 "${SPIKE_DIR}/scripts/build_failure_artifacts.py" \
    --spike "${SPIKE_DIR}" --state "${gate_state}" --gate "${failed_gate}" \
    --failure-code "${failure_code}" --output "${AERODOCS_SPIKE_REVIEW}"
  decision="$(python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --finalize --live-started)" || decision=FAIL
  [[ "${decision}" == FAIL ]] || fail 9 "failed run produced a non-FAIL decision"
  explicit_cleanup_complete=1
  exit 1
}

ca_file="${XDG_DATA_HOME}/caddy/pki/authorities/local/root.crt"
for _ in {1..30}; do
  [[ -s "${ca_file}" ]] && break
  sleep 1
done
[[ -s "${ca_file}" ]] || fail 7 "Caddy test CA was not created"
curl --silent --fail --cacert "${ca_file}" --max-time 5 \
  "https://idp.aerodocs-spike.invalid/realms/aerodocs-spike/.well-known/openid-configuration" >/dev/null ||
  fail 7 "OIDC discovery through dedicated IdP site failed"

ACTIVE_GATE="issuer_endpoint_confinement"
python3 "${SPIKE_DIR}/scripts/oidc_gate.py" \
  --ca-file "${ca_file}" \
  --username-file "${secrets_dir}/username" \
  --password-file "${secrets_dir}/user-password" \
  --client-secret-file "${secrets_dir}/client-secret" \
  >"${AERODOCS_SPIKE_RUNTIME}/oidc-gate.json"
python3 "${SPIKE_DIR}/scripts/validate_artifact.py" "${SPIKE_DIR}/schemas/evidence.schema.json" "${AERODOCS_SPIKE_RUNTIME}/oidc-gate.json" >/dev/null || fail 7 "OIDC evidence invalid"
jq -e '.result == "PASS" and .measurements.issuer_exact and .measurements.endpoint_origins_exact and .measurements.algorithm_rs256 and .measurements.audience_exact and .measurements.groups_count == 1 and .measurements.pkce_s256_advertised and .measurements.refresh_succeeded' "${AERODOCS_SPIKE_RUNTIME}/oidc-gate.json" >/dev/null || fail 7 "OIDC evidence incomplete"
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --apply-result "${AERODOCS_SPIKE_RUNTIME}/oidc-gate.json"

python3 "${SPIKE_DIR}/mock_aerodocs.py" --port "${mock_port}" >"${AERODOCS_SPIKE_RUNTIME}/mock.log" 2>&1 &
mock_pid=$!
export OAUTH2_PROXY_CLIENT_ID=aerodocs-spike-browser
export OAUTH2_PROXY_CLIENT_SECRET="$(< "${secrets_dir}/client-secret")"
export OAUTH2_PROXY_COOKIE_SECRET="$(< "${cookie_secret_file}")"
export SSL_CERT_FILE="${ca_file}"
"${oauth_bin}" --config "${SPIKE_DIR}/config/oauth2-proxy.cfg" --config-test >/dev/null
"${oauth_bin}" --config "${SPIKE_DIR}/config/oauth2-proxy.cfg" \
  >"${AERODOCS_SPIKE_RUNTIME}/oauth2-proxy.log" 2>&1 &
oauth_pid=$!
for _ in {1..30}; do
  curl --silent --fail --max-time 1 "${oauth_control_url}/ping" >/dev/null 2>&1 && break
  sleep 1
done
curl --silent --fail --max-time 2 "${oauth_control_url}/ping" >/dev/null || fail 8 "OAuth2 Proxy did not start"

# Hard gate: the reviewed stock-Caddy primitives cannot prove response-header
# field cardinality. Exercise every envelope vector against the fail-safe site,
# retain a sanitized FAIL, clean explicitly, and stop before browser data-plane
# tests or any AeroDocs browser invocation.
kill "${oauth_pid}"
wait "${oauth_pid}" || true
oauth_pid=""
python3 "${SPIKE_DIR}/mock_auth_envelope.py" --port "${oauth_port}" >"${AERODOCS_SPIKE_RUNTIME}/auth-mock.log" 2>&1 &
auth_mock_pid=$!
sleep 1
ACTIVE_GATE="gateway_token_limit"
python3 "${SPIKE_DIR}/scripts/probe_envelope.py" --ca-file "${ca_file}" \
  --endpoints-file "${endpoint_record}" \
  >"${AERODOCS_SPIKE_RUNTIME}/envelope-gate.json"
python3 "${SPIKE_DIR}/scripts/validate_artifact.py" "${SPIKE_DIR}/schemas/evidence.schema.json" "${AERODOCS_SPIKE_RUNTIME}/envelope-gate.json" >/dev/null || fail 8 "envelope evidence invalid"
jq -e '.result == "FAIL" and .gate_results.gateway_token_limit.result == "FAIL" and (.measurements | length) > 0' "${AERODOCS_SPIKE_RUNTIME}/envelope-gate.json" >/dev/null || fail 8 "envelope failure evidence incomplete"
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --apply-result "${AERODOCS_SPIKE_RUNTIME}/envelope-gate.json"
finalize_failure gateway_token_limit COOKIE_ENVELOPE

ACTIVE_GATE="health_route"
curl --silent --fail --request POST "${mock_control_url}/__control/reset" >/dev/null
health_body="$(curl --silent --fail --cacert "${ca_file}" \
  --header 'Authorization: hostile' --header 'Cookie: hostile=value' \
  --header 'X-AeroDocs-Gateway-Access-Token: hostile' \
  --header 'X-Auth-Request-Access-Token: hostile' --header 'X-Forwarded-User: hostile' \
  --header 'X-Forwarded-Groups: hostile' --header 'X-Forwarded-Email: hostile' \
  --header 'X-Forwarded-Preferred-Username: hostile' --header 'X-Forwarded-Access-Token: hostile' \
  --header 'X-Forwarded-Basic-Auth: hostile' --header 'Remote-User: hostile' \
  --header 'X_AeroDocs_Gateway_Access_Token: hostile' --header 'X_Auth_Request_Access_Token: hostile' \
  https://aerodocs-spike.invalid/health)"
[[ "${health_body}" == '{"mock":"AERODOCS_RESPONSE_SENTINEL"}' ]] || fail 8 "health response changed"
health_snapshot="$(curl --silent --fail "${mock_control_url}/__control/snapshot")"
jq -e '.invocations == 1 and .last.route_id == "health" and .last.authorization_field_count == 0 and .last.browser_cookie_field_count == 0 and .last.gateway_header_count == 0 and .last.hostile_header_names == []' <<<"${health_snapshot}" >/dev/null || fail 8 "health headers were not stripped"
[[ "$(curl --silent --output /dev/null --write-out '%{http_code}' --request POST --cacert "${ca_file}" https://aerodocs-spike.invalid/health)" == 404 ]] || fail 8 "non-GET health was not rejected"
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set health_route PASS
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set hostile_header_stripping PASS

json_status="$(curl --silent --output "${AERODOCS_SPIKE_RUNTIME}/json-error" --write-out '%{http_code}' \
  --cacert "${ca_file}" https://aerodocs-spike.invalid/search)"
[[ "${json_status}" == "401" ]] || fail 8 "JSON API did not return 401"
jq -e '.detail == "invalid_credentials" and (keys | length) == 1' \
  "${AERODOCS_SPIKE_RUNTIME}/json-error" >/dev/null || fail 8 "JSON API returned a non-fixed error"
navigation_status="$(curl --silent --output /dev/null --write-out '%{http_code}' \
  --cacert "${ca_file}" https://aerodocs-spike.invalid/ui/)"
[[ "${navigation_status}" == "302" ]] || fail 8 "UI navigation did not redirect"
[[ "$(curl --silent --output /dev/null --write-out '%{http_code}' --cacert "${ca_file}" https://aerodocs-spike.invalid/)" == 404 ]] || fail 8 "root route was not fixed 404"
[[ "$(curl --silent --output /dev/null --write-out '%{http_code}' --cacert "${ca_file}" https://aerodocs-spike.invalid/ui)" == 404 ]] || fail 8 "/ui route was not deliberate 404"
[[ "$(curl --silent --output /dev/null --write-out '%{http_code}' --cacert "${ca_file}" https://aerodocs-spike.invalid/ui/unknown)" == 404 ]] || fail 8 "unknown UI-like route was not 404"
for operator_path in documents indexes/rebuild; do
  [[ "$(curl --silent --output /dev/null --write-out '%{http_code}' --request POST --cacert "${ca_file}" "https://aerodocs-spike.invalid/${operator_path}")" == 401 ]] || fail 8 "operator mutation did not return 401"
done
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set json_navigation_behavior PASS
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set operator_mutation_behavior PASS
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set unknown_route_behavior PASS

ACTIVE_GATE="initial_login"
profile_dir="${AERODOCS_SPIKE_RUNTIME}/browser-profile"
mkdir -m 700 "${profile_dir}"
certutil -N --empty-password -d "sql:${profile_dir}"
certutil -A -d "sql:${profile_dir}" -n aerodocs-spike-ca -t 'C,,' -i "${ca_file}"
python3 "${SPIKE_DIR}/scripts/oauth_browser_flow.py" \
  --username-file "${secrets_dir}/username" \
  --password-file "${secrets_dir}/user-password" \
  --profile-dir "${profile_dir}" --endpoints-file "${endpoint_record}" \
  >"${AERODOCS_SPIKE_RUNTIME}/browser-flow.json"
python3 "${SPIKE_DIR}/scripts/validate_artifact.py" "${SPIKE_DIR}/schemas/evidence.schema.json" "${AERODOCS_SPIKE_RUNTIME}/browser-flow.json" >/dev/null || fail 8 "browser evidence invalid"
jq -e '.result == "PASS" and .measurements.initial_set_cookie_fields >= 1 and .measurements.refresh_set_cookie_fields >= 1 and .measurements.clearing_set_cookie_fields >= 1 and .measurements.access_token_header_absent and .measurements.mock_not_invoked_after_clear' "${AERODOCS_SPIKE_RUNTIME}/browser-flow.json" >/dev/null || fail 8 "browser evidence incomplete"
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --apply-result "${AERODOCS_SPIKE_RUNTIME}/browser-flow.json"

kill "${oauth_pid}"
wait "${oauth_pid}" || true
oauth_pid=""
python3 "${SPIKE_DIR}/mock_auth_envelope.py" --port "${oauth_port}" >"${AERODOCS_SPIKE_RUNTIME}/auth-mock.log" 2>&1 &
auth_mock_pid=$!
sleep 1
for oauth_path in /oauth2 /oauth2/ping; do
  [[ "$(curl --silent --output /dev/null --write-out '%{http_code}' --cacert "${ca_file}" "https://aerodocs-spike.invalid${oauth_path}")" == 202 ]] || fail 8 "exact OAuth control route failed"
done
[[ "$(curl --silent --output /dev/null --write-out '%{http_code}' --cacert "${ca_file}" https://aerodocs-spike.invalid/oauth2evil)" == 404 ]] || fail 8 "OAuth sibling route escaped fixed 404"
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set oauth2_routes PASS
curl --silent --fail --request POST "${mock_control_url}/__control/reset" >/dev/null
curl --silent --fail --request POST "${oauth_control_url}/__spike/reset" >/dev/null
direct_body="$(curl --silent --fail --cacert "${ca_file}" \
  --header 'Authorization: Bearer aaa.bbb.ccc' \
  --header 'Cookie: hostile=value' \
  --header 'X-AeroDocs-Gateway-Access-Token: hostile' \
  https://aerodocs-spike.invalid/search)"
[[ "${direct_body}" == '{"mock":"AERODOCS_RESPONSE_SENTINEL"}' ]] || fail 8 "direct public response changed"
direct_snapshot="$(curl --silent --fail "${mock_control_url}/__control/snapshot")"
auth_snapshot="$(curl --silent --fail "${oauth_control_url}/__spike/snapshot")"
jq -e '.invocations == 1 and .last.authorization_field_count == 1 and .last.authorization_field_lengths == [18] and .last.browser_cookie_field_count == 0 and .last.gateway_header_count == 0 and .last.phase8a_parser_verdict == "ACCEPTED"' \
  <<<"${direct_snapshot}" >/dev/null || fail 8 "direct bearer snapshot failed"
jq -e '.invocations == 0' <<<"${auth_snapshot}" >/dev/null || fail 8 "direct bearer invoked authentication service"
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set direct_bearer_separation PASS
ACTIVE_GATE="gateway_token_limit"
python3 "${SPIKE_DIR}/scripts/probe_envelope.py" --ca-file "${ca_file}" \
  --endpoints-file "${endpoint_record}" \
  >"${AERODOCS_SPIKE_RUNTIME}/envelope-gate.json"
python3 "${SPIKE_DIR}/scripts/validate_artifact.py" "${SPIKE_DIR}/schemas/evidence.schema.json" "${AERODOCS_SPIKE_RUNTIME}/envelope-gate.json" >/dev/null || fail 8 "envelope evidence invalid"
jq -e '.result == "PASS" and (.measurements | length) > 0' "${AERODOCS_SPIKE_RUNTIME}/envelope-gate.json" >/dev/null || fail 8 "envelope evidence incomplete"
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --apply-result "${AERODOCS_SPIKE_RUNTIME}/envelope-gate.json"
ACTIVE_GATE="duplicate_authorization_http1"
python3 "${SPIKE_DIR}/scripts/probe_authorization.py" --ca-file "${ca_file}" \
  --caddy-url "${caddy_url}" \
  --mock-control-url "${mock_control_url}" \
  --auth-control-url "${oauth_control_url}" \
  --protocol both --phase8a-repository "${REPO_DIR}" \
  --private-runtime "${AERODOCS_SPIKE_RUNTIME}" \
  >"${AERODOCS_SPIKE_RUNTIME}/authorization-gate.json"
python3 "${SPIKE_DIR}/scripts/validate_artifact.py" "${SPIKE_DIR}/schemas/evidence.schema.json" "${AERODOCS_SPIKE_RUNTIME}/authorization-gate.json" >/dev/null || fail 8 "Authorization evidence invalid"
jq -e '.result == "PASS" and .measurements.http1_case_count == 8 and .measurements.http2_case_count == 8 and .measurements.all_safe' "${AERODOCS_SPIKE_RUNTIME}/authorization-gate.json" >/dev/null || fail 8 "Authorization evidence incomplete"
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --apply-result "${AERODOCS_SPIKE_RUNTIME}/authorization-gate.json"

ACTIVE_GATE="protected_file_comparison"
git -C "${REPO_DIR}" diff --quiet milestone-8a -- app app/security pyproject.toml uv.lock package.json package-lock.json || fail 9 "protected files differ from milestone-8a"
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set protected_file_comparison PASS
ACTIVE_GATE="regression_suite"
(cd "${REPO_DIR}" && uv run pytest -q && npm run check && uv lock --check) || fail 9 "regression suite failed"
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set regression_suite PASS
ACTIVE_GATE="secret_scan"
for evidence_file in oidc-gate.json browser-flow.json envelope-gate.json authorization-gate.json configuration-hashes.json; do
  python3 "${SPIKE_DIR}/scripts/validate_artifact.py" --scan-only "${AERODOCS_SPIKE_RUNTIME}/${evidence_file}" >/dev/null || fail 9 "sanitized evidence scan failed"
done
python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set secret_scan PASS

ACTIVE_GATE="cleanup"
cleanup_failed=0
cleanup_processes || cleanup_failed=1
mock_pid=""; oauth_pid=""; auth_mock_pid=""; caddy_pid=""
bash "${SPIKE_DIR}/scripts/cleanup.sh" || cleanup_failed=1
if [[ "${cleanup_failed}" -eq 0 ]]; then
  python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set cleanup PASS
else
  python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --set cleanup FAIL
fi
explicit_cleanup_complete=1
decision="$(python3 "${SPIKE_DIR}/scripts/gate_registry.py" "${gate_state}" --finalize --live-started)" || decision=FAIL
[[ "${decision}" == PASS ]] || fail 9 "one or more mandatory live gates failed or were not executed"
