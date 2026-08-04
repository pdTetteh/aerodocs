#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SPIKE_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
REPO_DIR="$(CDPATH= cd -- "${SPIKE_DIR}/../.." && pwd -P)"
FIXED_RUNTIME_PARENT="${FIXED_RUNTIME_PARENT:-/tmp/aerodocs-8b0-$UID}"
FIXED_REVIEW_PARENT="${FIXED_REVIEW_PARENT:-/tmp/aerodocs-8b0-review-$UID}"
RUNTIME_MARKER="aerodocs-8b0-runtime-v2"

fail() {
  local code="$1"
  local message="$2"
  printf 'FAIL: %s\n' "${message}" >&2
  exit "${code}"
}

owner_id() {
  stat -c '%u' "$1" 2>/dev/null || stat -f '%u' "$1"
}

mode_bits() {
  stat -c '%a' "$1" 2>/dev/null || stat -f '%Lp' "$1"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail 2 "required command unavailable: $1"
}

assert_safe_parent() {
  local parent="$1"
  local expected="$2"
  [[ -n "${parent}" && "${parent}" == "${expected}" ]] || return 1
  if [[ "${AERODOCS_SPIKE_BOOTSTRAP_TEST_MODE:-0}" != 1 ]]; then
    [[ ! -L /tmp ]] || return 1
  fi
  [[ ! -L "${parent}" ]] || return 1
  [[ -d "${parent}" ]] || return 1
  [[ "$(owner_id "${parent}")" == "$(id -u)" ]] || return 1
  [[ "$(mode_bits "${parent}")" == "700" ]] || return 1
}

create_runtime_root() {
  [[ "${AERODOCS_SPIKE_BOOTSTRAP_TEST_MODE:-0}" == 1 || ! -L /tmp ]] || fail 2 "fixed runtime parent chain contains a symlink"
  install -d -m 700 "${FIXED_RUNTIME_PARENT}"
  assert_safe_parent "${FIXED_RUNTIME_PARENT}" "${FIXED_RUNTIME_PARENT}" ||
    fail 2 "fixed runtime parent is unsafe"
  AERODOCS_SPIKE_RUNTIME="$(mktemp -d "${FIXED_RUNTIME_PARENT}/run.XXXXXXXXXX")"
  AERODOCS_SPIKE_RUNTIME_CREATED=1
  export AERODOCS_SPIKE_RUNTIME AERODOCS_SPIKE_RUNTIME_CREATED
  interrupt_checkpoint after_runtime
  [[ ! -L "${AERODOCS_SPIKE_RUNTIME}" ]] || fail 2 "runtime directory is a symlink"
  chmod 700 "${AERODOCS_SPIKE_RUNTIME}"
  AERODOCS_SPIKE_NONCE="$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')"
  AERODOCS_SPIKE_CREATOR_PID="$$"
  export AERODOCS_SPIKE_NONCE AERODOCS_SPIKE_CREATOR_PID
  printf '%s\n%s\n%s\n' "${RUNTIME_MARKER}" "${AERODOCS_SPIKE_CREATOR_PID}" "${AERODOCS_SPIKE_NONCE}" \
    >"${AERODOCS_SPIKE_RUNTIME}/.sentinel"
  chmod 600 "${AERODOCS_SPIKE_RUNTIME}/.sentinel"
  interrupt_checkpoint after_sentinel
  export AERODOCS_SPIKE_RUNTIME AERODOCS_SPIKE_NONCE AERODOCS_SPIKE_CREATOR_PID AERODOCS_SPIKE_RUNTIME_CREATED
}

create_review_root() {
  [[ "${AERODOCS_SPIKE_BOOTSTRAP_TEST_MODE:-0}" == 1 || ! -L /tmp ]] || fail 2 "fixed review parent chain contains a symlink"
  install -d -m 700 "${FIXED_REVIEW_PARENT}"
  assert_safe_parent "${FIXED_REVIEW_PARENT}" "${FIXED_REVIEW_PARENT}" ||
    fail 2 "fixed review parent is unsafe"
  AERODOCS_SPIKE_REVIEW="$(mktemp -d "${FIXED_REVIEW_PARENT}/review.XXXXXXXXXX")"
  AERODOCS_SPIKE_REVIEW_CREATED=1
  export AERODOCS_SPIKE_REVIEW AERODOCS_SPIKE_REVIEW_CREATED
  interrupt_checkpoint after_review
  chmod 700 "${AERODOCS_SPIKE_REVIEW}"
  [[ ! -L "${AERODOCS_SPIKE_REVIEW}" ]] || fail 2 "review directory is a symlink"
  [[ "$(dirname -- "${AERODOCS_SPIKE_REVIEW}")" == "${FIXED_REVIEW_PARENT}" ]] || fail 2 "review directory escaped fixed parent"
  [[ "$(owner_id "${AERODOCS_SPIKE_REVIEW}")" == "$(id -u)" && "$(mode_bits "${AERODOCS_SPIKE_REVIEW}")" == "700" ]] || fail 2 "review directory ownership or mode is unsafe"
  AERODOCS_SPIKE_REVIEW_NONCE="$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')"
  export AERODOCS_SPIKE_REVIEW_NONCE
  printf '%s\n%s\n%s\n' "aerodocs-8b0-review-v2" "$$" "${AERODOCS_SPIKE_REVIEW_NONCE}" >"${AERODOCS_SPIKE_REVIEW}/.sentinel"
  chmod 600 "${AERODOCS_SPIKE_REVIEW}/.sentinel"
  interrupt_checkpoint after_review_sentinel
  export AERODOCS_SPIKE_REVIEW AERODOCS_SPIKE_REVIEW_NONCE AERODOCS_SPIKE_REVIEW_CREATED
}

bootstrap_remove_empty_child() {
  local target="$1"
  local parent="$2"
  local prefix="$3"
  [[ -n "${target}" && "$(dirname -- "${target}")" == "${parent}" && "$(basename -- "${target}")" =~ ^${prefix}\.[A-Za-z0-9]{10}$ ]] || return 1
  [[ -d "${target}" && ! -L "${target}" && "$(owner_id "${target}")" == "$(id -u)" && "$(mode_bits "${target}")" == "700" ]] || return 1
  [[ ! -e "${target}/.sentinel" ]] || rm -f -- "${target}/.sentinel" || return 1
  rmdir -- "${target}"
}

validate_review_root() {
  [[ -n "${AERODOCS_SPIKE_REVIEW:-}" && -n "${AERODOCS_SPIKE_REVIEW_NONCE:-}" ]] || return 1
  [[ "$(dirname -- "${AERODOCS_SPIKE_REVIEW}")" == "${FIXED_REVIEW_PARENT}" && ! -L "${AERODOCS_SPIKE_REVIEW}" ]] || return 1
  [[ "$(owner_id "${AERODOCS_SPIKE_REVIEW}")" == "$(id -u)" && "$(mode_bits "${AERODOCS_SPIKE_REVIEW}")" == "700" ]] || return 1
  { IFS= read -r review_marker && IFS= read -r review_pid && IFS= read -r review_nonce && ! IFS= read -r review_extra; } <"${AERODOCS_SPIKE_REVIEW}/.sentinel" || return 1
  [[ "${review_marker}" == "aerodocs-8b0-review-v2" && "${review_pid}" == "$$" && "${review_nonce}" == "${AERODOCS_SPIKE_REVIEW_NONCE}" ]]
}

validate_runtime_root() {
  : "${AERODOCS_SPIKE_RUNTIME:?runtime not created}"
  : "${AERODOCS_SPIKE_NONCE:?runtime nonce unavailable}"
  : "${AERODOCS_SPIKE_CREATOR_PID:?runtime creator unavailable}"
  [[ -n "${AERODOCS_SPIKE_RUNTIME}" && "${AERODOCS_SPIKE_RUNTIME}" != / && "${AERODOCS_SPIKE_RUNTIME}" != "${HOME}" ]] || return 1
  [[ "${AERODOCS_SPIKE_BOOTSTRAP_TEST_MODE:-0}" == 1 || ! -L /tmp ]] || return 1
  [[ ! -L "${FIXED_RUNTIME_PARENT}" && ! -L "${AERODOCS_SPIKE_RUNTIME}" ]] || return 1
  [[ "$(dirname -- "${AERODOCS_SPIKE_RUNTIME}")" == "${FIXED_RUNTIME_PARENT}" ]] || return 1
  [[ "$(basename -- "${AERODOCS_SPIKE_RUNTIME}")" =~ ^run\.[A-Za-z0-9]{10}$ ]] || return 1
  [[ -d "${AERODOCS_SPIKE_RUNTIME}" ]] || return 1
  [[ "$(owner_id "${AERODOCS_SPIKE_RUNTIME}")" == "$(id -u)" ]] || return 1
  [[ "$(mode_bits "${AERODOCS_SPIKE_RUNTIME}")" == "700" ]] || return 1
  case "${REPO_DIR}/" in "${AERODOCS_SPIKE_RUNTIME}/"*) return 1 ;; esac
  { IFS= read -r runtime_marker && IFS= read -r runtime_pid && IFS= read -r runtime_nonce && ! IFS= read -r runtime_extra; } <"${AERODOCS_SPIKE_RUNTIME}/.sentinel" || return 1
  [[ "${runtime_marker}" == "${RUNTIME_MARKER}" && "${runtime_pid}" == "${AERODOCS_SPIKE_CREATOR_PID}" && "${runtime_nonce}" == "${AERODOCS_SPIKE_NONCE}" ]]
}

require_runtime_root() {
  validate_runtime_root || fail 2 "runtime directory validation failed"
}

require_ubuntu_amd64() {
  [[ "$(uname -s)" == "Linux" ]] || fail 3 "Ubuntu Linux execution required"
  [[ "$(uname -m)" == "x86_64" ]] || fail 3 "amd64 execution required"
  # shellcheck disable=SC1091
  source /etc/os-release
  [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" && "${PRETTY_NAME:-}" == "Ubuntu 24.04 LTS" ]] ||
    fail 3 "exact Ubuntu 24.04 LTS execution required"
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}
