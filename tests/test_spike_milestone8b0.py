from __future__ import annotations

import importlib.util
import json
import os
import secrets
import selectors
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from jsonschema import (  # type: ignore[import-untyped]
    Draft202012Validator,
    FormatChecker,
    ValidationError,
)

ROOT = Path(__file__).parents[1]
SPIKE = ROOT / "spikes" / "milestone-8b0"


def read_framed_event(descriptor: int) -> dict[str, object]:
    data = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            assert selector.select(timeout=10), "bootstrap readiness EOF timeout"
            chunk = os.read(descriptor, 1029 - len(data))
            if not chunk:
                break
            data.extend(chunk)
            assert len(data) <= 1028
    assert len(data) >= 4
    length = struct.unpack(">I", data[:4])[0]
    assert len(data) == 4 + length
    value = json.loads(data[4:].decode("utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def bootstrap_fixed_root() -> Generator[tuple[Path, str]]:
    base = Path("/private/tmp" if sys.platform == "darwin" else "/tmp")
    root = base / f"aerodocs-m8b0-bootstrap-tests-{os.getuid()}"
    root.mkdir(mode=0o700)
    nonce = secrets.token_hex(32)
    sentinel = root / ".aerodocs-bootstrap-test-root"
    sentinel.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "aerodocs-m8b0-bootstrap-root",
                "nonce": nonce,
            },
            separators=(",", ":"),
        ),
        encoding="ascii",
    )
    sentinel.chmod(0o600)
    yield root, nonce
    assert json.loads(sentinel.read_text(encoding="ascii"))["nonce"] == nonce
    shutil.rmtree(root)


def load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def validator(name: str) -> Draft202012Validator:
    schema = json.loads((SPIKE / "schemas" / name).read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def test_protected_files_match_milestone_8a() -> None:
    assert (
        subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                "milestone-8a",
                "--",
                "app",
                "pyproject.toml",
                "uv.lock",
                "package.json",
                "package-lock.json",
            ],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )


def test_no_stage_8b1_source() -> None:
    changed = subprocess.run(
        ["git", "diff", "--name-only", "milestone-8a"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    assert all(
        path == ".gitignore"
        or path.startswith("spikes/milestone-8b0/")
        or path == "tests/test_spike_milestone8b0.py"
        for path in changed
    )
    assert "gateway_session" not in "\n".join(
        path.read_text(errors="ignore") for path in SPIKE.rglob("*") if path.is_file()
    )


def test_caddy_closed_route_and_header_policy() -> None:
    text = (SPIKE / "config/Caddyfile").read_text()
    assert text.count("https://idp.aerodocs-spike.invalid {") == 1
    assert text.count("https://aerodocs-spike.invalid {") == 1
    assert "@direct header Authorization" in text
    assert "header Authorization *" not in text
    assert "{rp.header.X-Auth-Request-Access-Token}" not in text
    assert "{http.reverse_proxy.header." not in text
    assert "copy_response_headers" not in text
    assert text.count("authentication_envelope_unavailable") == 2
    assert "max_response_header 24KiB" in text
    for value in (
        "/health",
        "/oauth2 /oauth2/*",
        "path('/documents', '/indexes/active', '/ui/api/index-status')",
        "/indexes/active",
        "/indexes/rebuild",
    ):
        assert value in text
    assert "/documents*" not in text and "/indexes*" not in text
    assert "^/documents/[^/]+(?:/pages|/chunks)?$" in text
    for header in (
        "X-AeroDocs-Gateway-Access-Token",
        "X-Auth-Request-Access-Token",
        "X-Forwarded-User",
        "X-Forwarded-Groups",
        "X-Forwarded-Email",
        "X-Forwarded-Preferred-Username",
        "X-Forwarded-Access-Token",
        "X-Forwarded-Basic-Auth",
        "Remote-User",
        "X_AeroDocs_Gateway_Access_Token",
        "X_Auth_Request_Access_Token",
    ):
        assert f"request_header -{header}" in text
    assert (
        text.rstrip().endswith("}") and 'respond `{"detail":"not_found"}` 404' in text
    )


def test_oauth2_proxy_configuration_is_explicit_and_loopback_trusted() -> None:
    lines = set((SPIKE / "config/oauth2-proxy.cfg").read_text().splitlines())
    assert 'trusted_proxy_ips = ["127.0.0.1/32", "::1/128"]' in lines
    for setting in (
        'upstreams = ["static://202"]',
        "pass_access_token = true",
        "set_xauthrequest = true",
        "pass_authorization_header = false",
        "set_authorization_header = false",
        "pass_basic_auth = false",
        "set_basic_auth = false",
        "pass_user_headers = false",
        "skip_jwt_bearer_tokens = false",
        'session_store_type = "cookie"',
        'code_challenge_method = "S256"',
        "auth_logging = false",
        "request_logging = false",
        "standard_logging = false",
    ):
        assert setting in lines


def _response(fields: list[tuple[bytes, bytes]], target: int | None = None) -> bytes:
    prefix = b"HTTP/1.1 202 Accepted\r\n" + b"".join(
        name + b": " + value + b"\r\n" for name, value in fields
    )
    if target is not None:
        fixed = len(prefix) + len(b"X-Padding: \r\n\r\n")
        prefix += b"X-Padding: " + b"x" * (target - fixed) + b"\r\n"
    return prefix + b"\r\n"


@pytest.mark.parametrize("size,passes", [(3072, True), (3073, False)])
def test_gateway_token_wire_boundary(size: int, passes: bool) -> None:
    module = load(SPIKE / "scripts/measure_envelope.py")
    raw = _response([(b"X-Auth-Request-Access-Token", b"x" * size)])
    if passes:
        assert module.measure(raw)["gateway_token_bytes"] == size
    else:
        with pytest.raises(module.EnvelopeError):
            module.measure(raw)


@pytest.mark.parametrize("target,passes", [(24576, True), (24577, False)])
def test_complete_response_header_boundary(target: int, passes: bool) -> None:
    module = load(SPIKE / "scripts/measure_envelope.py")
    raw = _response([(b"X-Auth-Request-Access-Token", b"a.b.c")], target)
    assert len(raw) == target
    if passes:
        assert module.measure(raw)["auth_response_header_bytes"] == target
    else:
        with pytest.raises(module.EnvelopeError):
            module.measure(raw)


def test_every_cookie_boundary_and_one_over_is_exact() -> None:
    module = load(SPIKE / "mock_auth_envelope.py")
    measure = load(SPIKE / "scripts/measure_envelope.py")
    pairs = {
        "set-cookie-field": "set_cookie_field_bytes",
        "set-cookie-total": "set_cookie_total_bytes",
        "active-parts": "active_cookie_parts",
        "active-values": "active_cookie_value_bytes",
        "name-value": "cookie_name_value_bytes",
        "complete-header": "auth_response_header_bytes",
        "gateway-token": "gateway_token",
        "set-cookie-count": "set_cookie_fields",
    }
    for prefix, error in pairs.items():
        boundary = module.serialize(module.fields_for(f"{prefix}-boundary".encode()))
        over = module.serialize(module.fields_for(f"{prefix}-over".encode()))
        assert measure.measure(boundary)["within_limits"] is True
        with pytest.raises(measure.EnvelopeError, match=error):
            measure.measure(over)


def test_gateway_token_invalid_envelopes_and_compact_boundaries_exist() -> None:
    mock = load(SPIKE / "mock_auth_envelope.py")
    probe = load(SPIKE / "scripts/probe_envelope.py")
    names = {case.name for case in probe.CASES}
    invalid = {
        "missing-token",
        "empty-token",
        "identical-token-fields",
        "different-token-fields",
        "comma-token",
        "leading-space-token",
        "trailing-space-token",
        "control-token",
        "non-ascii-token",
        "one-segment-token",
        "two-segment-token",
        "empty-segment-token",
        "gateway-token-over",
    }
    assert invalid <= names
    assert all(not case.should_pass for case in probe.CASES if case.name in invalid)
    boundary = dict(mock.fields_for(b"gateway-token-boundary"))[
        b"X-Auth-Request-Access-Token"
    ]
    over = dict(mock.fields_for(b"gateway-token-over"))[b"X-Auth-Request-Access-Token"]
    assert len(boundary) == 3072 and len(boundary.split(b".")) == 3
    assert len(over) == 3073 and len(over.split(b".")) == 3


def test_exact_cookie_wire_generator() -> None:
    module = load(SPIKE / "scripts/probe_envelope.py")
    assert len(module.complete_cookie_field(16384)) == 16384
    assert len(module.complete_cookie_field(16385)) == 16385
    with pytest.raises(module.ProbeFailure, match="too_small"):
        module.complete_cookie_field(4)
    assert module.complete_cookie_field(16384).count(b"\r\n") == 1
    exact = module.complete_cookie_field(16384)
    with pytest.raises(module.ProbeFailure, match="control"):
        module.validate_cookie_wire([exact[:-3] + b"\t\r\n"], 16384)
    with pytest.raises(module.ProbeFailure, match="multiple"):
        module.validate_cookie_wire([exact, exact], 16384)
    with pytest.raises(module.ProbeFailure, match="rewrite"):
        module.validate_cookie_wire([exact + b"x"], 16384)
    with pytest.raises(module.ProbeFailure):
        module.validate_cookie_wire([exact + b"\r\n"], 16386)


def test_incoming_cookie_wire_boundary_and_one_over() -> None:
    module = load(SPIKE / "scripts/measure_envelope.py")
    prefix = b"Cookie: unrelated="
    exact = prefix + b"x" * (16384 - len(prefix) - 2) + b"\r\n"
    assert (
        len(exact) == 16384 and module.measure_incoming_cookie(exact)["within_limits"]
    )
    with pytest.raises(module.EnvelopeError, match="incoming_cookie_bytes"):
        module.measure_incoming_cookie(exact[:-2] + b"x\r\n")


@pytest.mark.parametrize(
    "raw",
    [
        b"HTTP/1.1 202 Accepted\r\n Folded: x\r\n\r\n",
        b"HTTP/1.1 202 Accepted\r\nX: a\nvalue\r\n\r\n",
        b"HTTP/1.1 202 Accepted\r\nBad Name: x\r\n\r\n",
        b"HTTP/1.1 202 Accepted\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\n",
        b"HTTP/1.1 202 Accepted\r\nTransfer-Encoding: chunked\r\n\r\n",
        b"HTTP/1.1 202 Accepted\r\nX: y\r\n",
    ],
)
def test_malformed_wire_envelopes_fail(raw: bytes) -> None:
    module = load(SPIKE / "scripts/measure_envelope.py")
    with pytest.raises(module.EnvelopeError):
        module.measure(raw)


def test_mock_uses_real_phase8a_parser_and_retains_only_lengths() -> None:
    module = load(SPIKE / "mock_aerodocs.py")
    invalid = [
        (b"authorization", b"Bearer aaa.bbb.ccc"),
        (b"authorization", b"Bearer ddd.eee.fff"),
    ]
    data = module.build_metadata(99, invalid)
    assert data["phase8a_parser_verdict"] == "REJECTED"
    assert data["authorization_field_lengths"] == [18, 18]
    encoded = json.dumps(data)
    assert "aaa.bbb" not in encoded and "ddd.eee" not in encoded
    assert module.BoundedThreadingHTTPServer.request_queue_size == module.MAX_WORKERS


def test_authorization_probe_is_not_status_only() -> None:
    text = (SPIKE / "scripts/probe_authorization.py").read_text()
    assert "phase8a_parser_verdict" in text
    assert "auth_delta == 0" in text and "mock_delta == 0" in text
    assert "status != 200" not in text
    assert '"empty_field"' in text and '"valid_control"' in text
    assert '"http1", "http2"' in text


def test_http2_trace_counts_structural_header_events() -> None:
    module = load(SPIKE / "scripts/probe_authorization.py")
    trace = (
        b"=> Send header, 80 bytes\n"
        b"0000: authorization: Bearer same.value.here\n"
        b"002a: authorization: Bearer same.value.here\n"
        b"== Info: Bearer same.value.here appears elsewhere\n"
    )
    assert module.parse_http2_trace(trace) == 2
    assert module.parse_http2_trace(b"unparseable same.value.here") == 0
    source = (SPIKE / "scripts/probe_authorization.py").read_text()
    assert '"--trace-ascii",\n            str(trace_file),' in source
    assert "completed.stderr" not in source
    for private_name in ("trace", "body", "status", "stderr"):
        assert f'work / "{private_name}"' in source


def test_authorization_probe_requires_explicit_shared_endpoints() -> None:
    text = (SPIKE / "scripts/probe_authorization.py").read_text()
    assert 'parser.add_argument("--caddy-url", required=True)' in text
    assert 'parser.add_argument("--mock-control-url", required=True)' in text
    assert 'parser.add_argument("--auth-control-url", required=True)' in text
    assert "default=18001" not in text


def test_browser_probe_requires_token_equality_refresh_and_clearing() -> None:
    text = (SPIKE / "scripts/oauth_browser_flow.py").read_text()
    assert "gateway_token_matches_expected" in text
    assert "initial_cookie_fields_missing_or_collapsed" in text
    assert "refresh_cookie_fields_missing_or_collapsed" in text
    assert "clearing_refresh_failed" in text
    assert "/oauth2/sign_out" in text
    assert "mock_not_invoked_after_clear" in text
    assert 'APP_URL = "https://aerodocs-spike.invalid/ui/"' in text
    assert 'str(endpoints["caddy_url"]).rstrip("/") + "/ui/"' in text


def test_harness_uses_public_sentinel_and_loopback_snapshot() -> None:
    text = (SPIKE / "scripts/run-ubuntu-spike.sh").read_text()
    assert "direct public response changed" in text
    assert '"${mock_control_url}/__control/snapshot"' in text
    assert '.last.phase8a_parser_verdict == "ACCEPTED"' in text


def test_gate_registry_is_closed_and_live_omissions_fail() -> None:
    module = load(SPIKE / "scripts/gate_registry.py")
    gates = module.initial()
    assert len(gates) == 52
    assert module.decide(gates, False) == "NOT_YET_EXECUTED"
    assert module.decide(gates, True) == "FAIL"
    assert module.decide(dict.fromkeys(module.GATES, "PASS"), True) == "PASS"
    failed = dict.fromkeys(module.GATES, "PASS")
    failed["cleanup"] = "FAIL"
    assert module.decide(failed, True) == "FAIL"


def test_gate_registry_finalizes_once(tmp_path: Path) -> None:
    module = load(SPIKE / "scripts/gate_registry.py")
    state = tmp_path / "gates.json"
    state.write_text(json.dumps(dict.fromkeys(module.GATES, "PASS")), encoding="utf-8")
    first = subprocess.run(
        [
            "python3",
            SPIKE / "scripts/gate_registry.py",
            state,
            "--finalize",
            "--live-started",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    second = subprocess.run(
        [
            "python3",
            SPIKE / "scripts/gate_registry.py",
            state,
            "--finalize",
            "--live-started",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert first.returncode == 0 and first.stdout.strip() == "PASS"
    assert second.returncode == 2


@pytest.mark.parametrize(
    ("live_gate", "cleanup", "expected"),
    [
        ("PASS", "PASS", "PASS"),
        ("PASS", "FAIL", "FAIL"),
        ("FAIL", "PASS", "FAIL"),
        ("FAIL", "FAIL", "FAIL"),
        ("NOT_YET_EXECUTED", "PASS", "FAIL"),
    ],
)
def test_cleanup_is_terminal_before_live_decision(
    live_gate: str, cleanup: str, expected: str
) -> None:
    module = load(SPIKE / "scripts/gate_registry.py")
    gates = dict.fromkeys(module.GATES, "PASS")
    gates["gateway_token_limit"] = live_gate
    gates["cleanup"] = cleanup
    assert module.decide(gates, True) == expected


def test_harness_finalizes_after_explicit_cleanup_and_preserves_gate() -> None:
    text = (SPIKE / "scripts/run-ubuntu-spike.sh").read_text()
    failure_function = text[text.index("finalize_failure()") : text.index("ca_file=")]
    assert failure_function.index('scripts/cleanup.sh"') < failure_function.index(
        "--finalize --live-started"
    )
    assert '--gate "${failed_gate}"' in failure_function
    assert "explicit_cleanup_complete=1" in failure_function
    trap_start = text.index("cleanup_on_exit()")
    trap_end = text.index("trap cleanup_on_exit")
    trap_function = text[trap_start:trap_end]
    assert "explicit_cleanup_complete" in trap_function


def test_gate_result_mapping_rejects_unknown_missing_and_conflicts() -> None:
    module = load(SPIKE / "scripts/gate_registry.py")
    states = module.initial()
    with pytest.raises(ValueError, match="missing"):
        module.apply_probe_results(states, {})
    with pytest.raises(ValueError, match="unknown"):
        module.apply_probe_results(
            states, {"fabricated": {"result": "PASS", "measurements": {"ok": True}}}
        )
    passed = module.apply_probe_results(
        states, {"health_route": {"result": "PASS", "measurements": {"status": 200}}}
    )
    with pytest.raises(ValueError, match="conflicting"):
        module.apply_probe_results(
            passed,
            {"health_route": {"result": "FAIL", "measurements": {"status": 500}}},
        )
    with pytest.raises(ValueError, match="invalid"):
        module.apply_probe_results(
            states, {"health_route": {"result": "PASS", "measurements": {}}}
        )


def test_bootstrap_trap_precedes_runtime_creation() -> None:
    text = (SPIKE / "scripts/run-ubuntu-spike.sh").read_text()
    common = (SPIKE / "scripts/common.sh").read_text()
    startup_text = text + common
    trap = text.index("trap cleanup_on_exit EXIT")
    assert trap < text.index("create_runtime_root")
    assert trap < text.index("create_review_root")
    for point in (
        "before_runtime",
        "after_runtime",
        "after_sentinel",
        "after_review",
        "after_review_sentinel",
        "after_gate_state",
        "after_first_container_resource",
    ):
        assert f"interrupt_checkpoint {point}" in startup_text
    assert (
        "after_first_download" in (SPIKE / "scripts/resolve-provenance.sh").read_text()
    )
    assert (
        common.index("AERODOCS_SPIKE_RUNTIME_CREATED=1")
        < common.index("interrupt_checkpoint after_runtime")
        < common.index("AERODOCS_SPIKE_NONCE=")
    )
    sentinel_chmod = 'chmod 600 "${AERODOCS_SPIKE_RUNTIME}/.sentinel"'
    assert common.index(sentinel_chmod) < common.index(
        "interrupt_checkpoint after_sentinel"
    )


@pytest.mark.parametrize(
    ("checkpoint", "interrupt_signal"),
    [
        ("before_runtime", signal.SIGTERM),
        ("after_runtime", signal.SIGTERM),
        ("after_sentinel", signal.SIGTERM),
        ("after_review", signal.SIGTERM),
        ("after_review_sentinel", signal.SIGTERM),
        ("after_gate_state", signal.SIGTERM),
        ("after_first_download", signal.SIGTERM),
        ("after_first_container_resource", signal.SIGTERM),
        ("after_gate_state", signal.SIGINT),
    ],
)
def test_real_subprocess_interruption_cleans_only_created_resources(
    tmp_path: Path,
    bootstrap_fixed_root: tuple[Path, str],
    checkpoint: str,
    interrupt_signal: signal.Signals,
) -> None:
    tmp_path.chmod(0o700)
    fixed_root, root_nonce = bootstrap_fixed_root
    parent = Path(tempfile.mkdtemp(prefix="case-", dir=fixed_root))
    parent.chmod(0o700)
    unrelated_file = parent / "unrelated-file"
    unrelated_directory = parent / "unrelated-directory"
    unrelated_file.write_text("preserve", encoding="utf-8")
    unrelated_directory.mkdir(mode=0o700)
    (unrelated_directory / "preserve").write_text("preserve", encoding="utf-8")
    sibling = parent / "runtime-parent-similar"
    sibling.mkdir(mode=0o700)
    symlink_target = parent / "symlink-target"
    symlink_target.mkdir(mode=0o700)
    symlink = parent / "runtime-parent-link"
    symlink.symlink_to(symlink_target, target_is_directory=True)
    nonce = secrets.token_hex(32)
    parent_sentinel = parent / ".aerodocs-bootstrap-test-parent"
    parent_sentinel.write_text(
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
    parent_sentinel.chmod(0o600)
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
        AERODOCS_SPIKE_TEST_INTERRUPT_AFTER=checkpoint,
        AERODOCS_SPIKE_TEST_ROOT=str(tmp_path / "hostile-root"),
        TMPDIR=str(tmp_path),
        TEMP=str(tmp_path),
        TMP=str(tmp_path),
        AERODOCS_SPIKE_TEST_PARENT=str(parent),
        AERODOCS_SPIKE_TEST_NONCE_FD=str(nonce_read),
        AERODOCS_SPIKE_TEST_ROOT_NONCE_FD=str(root_nonce_read),
        AERODOCS_SPIKE_TEST_READY_FD=str(ready_write),
        AERODOCS_SPIKE_TEST_CONTROL_FD=str(control_read),
    )
    process = subprocess.Popen(
        ["bash", SPIKE / "scripts/run-ubuntu-spike.sh"],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        pass_fds=(ready_write, control_read, nonce_read, root_nonce_read),
    )
    os.close(ready_write)
    os.close(control_read)
    os.close(nonce_read)
    os.close(root_nonce_read)
    readiness = read_framed_event(ready_read)
    expected_kinds = {
        "before_runtime": ("none", 0),
        "after_runtime": ("runtime", 1),
        "after_sentinel": ("runtime_sentinel", 2),
        "after_review": ("review", 3),
        "after_review_sentinel": ("review_sentinel", 4),
        "after_gate_state": ("gate_state", 5),
        "after_first_download": ("download", 6),
        "after_first_container_resource": ("synthetic_container_like", 7),
    }
    resource_kind, resource_count = expected_kinds[checkpoint]
    assert readiness == {
        "schema_version": 1,
        "checkpoint": checkpoint,
        "resource_kind": resource_kind,
        "created_resource_count": resource_count,
        "state": "BLOCKED",
    }
    runtime_children = list((parent / "runtime-parent").glob("run.*"))
    review_children = list((parent / "review-parent").glob("review.*"))
    assert bool(runtime_children) == (resource_count >= 1)
    assert bool(review_children) == (resource_count >= 3)
    if resource_count >= 2:
        assert (runtime_children[0] / ".sentinel").is_file()
    if resource_count >= 4:
        assert (review_children[0] / ".sentinel").is_file()
    if resource_count >= 5:
        assert (review_children[0] / ".gate-state.json").is_file()
    if runtime_children:
        assert (runtime_children[0] / "downloads/bootstrap-test").is_file() == (
            resource_count >= 6
        )
        assert (runtime_children[0] / "caddy-data/bootstrap-test").is_file() == (
            resource_count >= 7
        )
    process.send_signal(interrupt_signal)
    os.close(control_write)
    os.close(ready_read)
    stdout, stderr = process.communicate(timeout=10)
    expected_returncode = 130 if interrupt_signal == signal.SIGINT else 143
    assert process.returncode == expected_returncode, (stdout, stderr)
    cleanup_events = [
        json.loads(line)
        for line in stdout.splitlines()
        if '"event":"cleanup_complete"' in line
    ]
    assert cleanup_events == [
        {"schema_version": 1, "event": "cleanup_complete", "result": "PASS"}
    ]
    assert "PASS attestation" not in stdout
    assert not list((parent / "runtime-parent").glob("run.*"))
    assert not list((parent / "review-parent").glob("review.*"))
    assert unrelated_file.read_text(encoding="utf-8") == "preserve"
    assert (unrelated_directory / "preserve").read_text(encoding="utf-8") == "preserve"
    assert sibling.is_dir() and symlink.is_symlink() and symlink_target.is_dir()
    assert not (tmp_path / f"aerodocs-m8b0-bootstrap-tests-{os.getuid()}").exists()


def test_interrupted_cleanup_failure_is_retained_as_fail(
    tmp_path: Path, bootstrap_fixed_root: tuple[Path, str]
) -> None:
    tmp_path.chmod(0o700)
    fixed_root, root_nonce = bootstrap_fixed_root
    parent = Path(tempfile.mkdtemp(prefix="failure-", dir=fixed_root))
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
        AERODOCS_SPIKE_TEST_INTERRUPT_AFTER="after_gate_state",
        AERODOCS_SPIKE_TEST_ROOT=str(tmp_path / "hostile-root"),
        TMPDIR=str(tmp_path),
        TEMP=str(tmp_path),
        TMP=str(tmp_path),
        AERODOCS_SPIKE_TEST_PARENT=str(parent),
        AERODOCS_SPIKE_TEST_NONCE_FD=str(nonce_read),
        AERODOCS_SPIKE_TEST_ROOT_NONCE_FD=str(root_nonce_read),
        AERODOCS_SPIKE_TEST_READY_FD=str(ready_write),
        AERODOCS_SPIKE_TEST_CONTROL_FD=str(control_read),
        AERODOCS_SPIKE_TEST_FORCE_CLEANUP_FAILURE="1",
    )
    process = subprocess.Popen(
        ["bash", SPIKE / "scripts/run-ubuntu-spike.sh"],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        pass_fds=(ready_write, control_read, nonce_read, root_nonce_read),
    )
    os.close(ready_write)
    os.close(control_read)
    os.close(nonce_read)
    os.close(root_nonce_read)
    readiness = read_framed_event(ready_read)
    assert readiness["checkpoint"] == "after_gate_state"
    process.send_signal(signal.SIGTERM)
    os.close(control_write)
    os.close(ready_read)
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode != 0, (stdout, stderr)
    cleanup_events = [
        json.loads(line)
        for line in stdout.splitlines()
        if '"event":"cleanup_complete"' in line
    ]
    assert cleanup_events == [
        {"schema_version": 1, "event": "cleanup_complete", "result": "FAIL"}
    ]
    review_children = list((parent / "review-parent").glob("review.*"))
    assert len(review_children) == 1
    attestation = json.loads(
        (review_children[0] / "failure-attestation.json").read_text(encoding="utf-8")
    )
    assert attestation["status"] == "FAIL"
    assert attestation["hard_gates"]["cleanup"] == "FAIL"
    assert attestation["review"] == {"reviewed": False, "decision": "FAIL"}


@pytest.mark.parametrize(
    "unsafe_kind",
    [
        "relative",
        "root",
        "home",
        "repository",
        "repository_ancestor",
        "repository_descendant",
        "production_parent",
        "outside_root",
        "fixed_root",
        "nested_grandchild",
        "symlink",
        "wrong_mode",
        "missing_sentinel",
        "malformed_sentinel",
        "nonce_mismatch",
        "reused_nonce",
    ],
)
def test_bootstrap_test_mode_refuses_unsafe_parent(
    tmp_path: Path, bootstrap_fixed_root: tuple[Path, str], unsafe_kind: str
) -> None:
    tmp_path.chmod(0o700)
    fixed_root, root_nonce = bootstrap_fixed_root
    parent = Path(tempfile.mkdtemp(prefix="unsafe-", dir=fixed_root))
    created_parent = parent
    parent.chmod(0o700)
    nonce = secrets.token_hex(32)
    sentinel_nonce = nonce
    sentinel = parent / ".aerodocs-bootstrap-test-parent"
    sentinel.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "aerodocs-m8b0-bootstrap-test",
                "nonce": sentinel_nonce,
            },
            separators=(",", ":"),
        ),
        encoding="ascii",
    )
    sentinel.chmod(0o600)
    if unsafe_kind == "symlink":
        target = fixed_root / "target"
        target.mkdir(mode=0o700)
        parent = fixed_root / "parent-link"
        parent.symlink_to(target, target_is_directory=True)
    elif unsafe_kind == "wrong_mode":
        parent.chmod(0o755)
    elif unsafe_kind == "missing_sentinel":
        sentinel.unlink()
    elif unsafe_kind == "malformed_sentinel":
        sentinel.write_text("not-json", encoding="ascii")
    elif unsafe_kind in {"nonce_mismatch", "reused_nonce"}:
        nonce = secrets.token_hex(32)
    elif unsafe_kind == "relative":
        parent = Path(parent.name)
    elif unsafe_kind == "root":
        parent = Path("/")
    elif unsafe_kind == "home":
        parent = Path.home()
    elif unsafe_kind == "repository":
        parent = ROOT
    elif unsafe_kind == "repository_ancestor":
        parent = ROOT.parent
    elif unsafe_kind == "repository_descendant":
        parent = SPIKE
    elif unsafe_kind == "production_parent":
        parent = Path(f"/tmp/aerodocs-8b0-{os.getuid()}")
    elif unsafe_kind == "outside_root":
        parent = tmp_path
    elif unsafe_kind == "fixed_root":
        parent = fixed_root
    elif unsafe_kind == "nested_grandchild":
        nested = parent / "nested"
        nested.mkdir(mode=0o700)
        parent = nested
    nonce_read, nonce_write = os.pipe()
    root_nonce_read, root_nonce_write = os.pipe()
    os.write(nonce_write, (nonce + "\n").encode("ascii"))
    os.close(nonce_write)
    os.write(root_nonce_write, (root_nonce + "\n").encode("ascii"))
    os.close(root_nonce_write)
    environment = os.environ.copy()
    environment.update(
        AERODOCS_SPIKE_BOOTSTRAP_TEST_MODE="1",
        AERODOCS_SPIKE_TEST_INTERRUPT_AFTER="before_runtime",
        AERODOCS_SPIKE_TEST_ROOT=str(tmp_path / "hostile-root"),
        TMPDIR=str(tmp_path),
        TEMP=str(tmp_path),
        TMP=str(tmp_path),
        AERODOCS_SPIKE_TEST_PARENT=str(parent),
        AERODOCS_SPIKE_TEST_NONCE_FD=str(nonce_read),
        AERODOCS_SPIKE_TEST_ROOT_NONCE_FD=str(root_nonce_read),
        AERODOCS_SPIKE_TEST_READY_FD="8",
        AERODOCS_SPIKE_TEST_CONTROL_FD="9",
    )
    completed = subprocess.run(
        ["bash", SPIKE / "scripts/run-ubuntu-spike.sh"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        pass_fds=(nonce_read, root_nonce_read),
    )
    os.close(nonce_read)
    os.close(root_nonce_read)
    assert completed.returncode != 0
    assert not any((created_parent / "runtime-parent").glob("run.*"))
    assert not any((created_parent / "review-parent").glob("review.*"))
    assert not (tmp_path / f"aerodocs-m8b0-bootstrap-tests-{os.getuid()}").exists()


def test_actual_podman_interruption_gate_is_ubuntu_only_and_unexecuted() -> None:
    registry = load(SPIKE / "scripts/gate_registry.py")
    assert registry.initial()["podman_interruption_cleanup"] == "NOT_YET_EXECUTED"
    source = (SPIKE / "scripts/probe_podman_interruption.py").read_text()
    assert 'platform.system() != "Linux"' in source
    assert 'VERSION_ID="24.04"' in source
    assert '"purpose": "interruption_target"' in source
    assert '"purpose": "unrelated_preservation_control"' in source
    assert "process.send_signal(signal.SIGTERM)" in source
    harness = (SPIKE / "scripts/run-ubuntu-spike.sh").read_text()
    assert 'scripts/probe_podman_interruption.py"' in harness
    assert '--apply-result "${podman_evidence}"' in harness
    assert 'keys == ["podman_interruption_cleanup"]' in harness
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["python3", SPIKE / "scripts/probe_podman_interruption.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode != 0
        assert "Ubuntu 24.04 LTS amd64 required" in completed.stderr


def manifest_value() -> dict[str, Any]:
    target = {
        "kind": "network",
        "identifier": "authorized-target",
        "run_label": "a" * 24,
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
    control = {
        "kind": "volume",
        "identifier": "aerodocs-8b0-caddy-control",
        "run_label": "b" * 24,
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
    return {"schema_version": 2, "run_label": "a" * 24, "resources": [target, control]}


class FakePodman:
    def __init__(self, labels: dict[str, str] | None = None) -> None:
        self.labels = labels or {
            "authorized-target": "a" * 24,
            "aerodocs-8b0-caddy-control": "b" * 24,
            "aerodocs-8b0-caddy": "preexisting",
            "aerodocs-8b0-caddy-control-similar": "b" * 24,
        }
        self.present = set(self.labels)
        self.removed: list[tuple[str, str]] = []

    def exists(self, resource: Any) -> bool:
        return resource.identifier in self.present

    def label_matches(self, resource: Any) -> bool:
        return self.labels.get(str(resource.identifier)) == str(resource.run_label)

    def remove_exact(self, resource: Any) -> bool:
        if not self.label_matches(resource):
            return False
        self.removed.append((resource.kind, resource.identifier))
        self.present.discard(resource.identifier)
        return True


def test_podman_manifest_cleanup_is_phase_aware_and_preserves_unrelated() -> None:
    controller = load(SPIKE / "scripts/probe_podman_interruption.py")
    manifest_module = load(SPIKE / "scripts/resource_manifest.py")
    manifest = manifest_module.validate(manifest_value())
    fake = FakePodman()
    assert controller.cleanup_manifest(fake, manifest, "child_cleanup") == (True, 1, 0)
    assert fake.removed == [("network", "authorized-target")]
    assert "aerodocs-8b0-caddy-control" in fake.present
    assert "aerodocs-8b0-caddy" in fake.present
    assert "aerodocs-8b0-caddy-control-similar" in fake.present
    assert controller.cleanup_manifest(fake, manifest, "controller_fallback") == (
        True,
        0,
        0,
    )
    assert controller.cleanup_manifest(fake, manifest, "controller_final_teardown") == (
        True,
        1,
        0,
    )
    assert fake.removed[-1] == ("volume", "aerodocs-8b0-caddy-control")
    assert fake.present == {
        "aerodocs-8b0-caddy",
        "aerodocs-8b0-caddy-control-similar",
    }


def test_controller_manifest_contains_every_created_resource_once() -> None:
    controller = load(SPIKE / "scripts/probe_podman_interruption.py")
    manifest_module = load(SPIKE / "scripts/resource_manifest.py")
    value = controller.build_manifest_value("1" * 24, "a" * 24, "b" * 24)
    manifest = manifest_module.validate(value)
    assert len(manifest.resources) == 4
    assert (
        len({(resource.kind, resource.identifier) for resource in manifest.resources})
        == 4
    )
    assert sum(resource.creator == "child" for resource in manifest.resources) == 2
    assert (
        sum(
            resource.purpose == "unrelated_preservation_control"
            for resource in manifest.resources
        )
        == 2
    )
    assert len(manifest_module.select(manifest, "child_cleanup")) == 2
    assert len(manifest_module.select(manifest, "controller_fallback")) == 2
    assert len(manifest_module.select(manifest, "controller_final_teardown")) == 2


def test_resource_manifest_is_closed_unique_and_policy_consistent() -> None:
    module = load(SPIKE / "scripts/resource_manifest.py")
    value = manifest_value()
    manifest = module.validate(value)
    assert len(manifest.resources) == 2
    assert {entry.purpose for entry in manifest.resources} == {
        "interruption_target",
        "unrelated_preservation_control",
    }
    first = value["resources"][0]
    for mutation in (
        {**value, "unexpected": True},
        {**value, "run_label": "wrong"},
        {
            **value,
            "resources": [{**first, "kind": "command"}, value["resources"][1]],
        },
        {
            **value,
            "resources": [first, {**first}],
        },
        {
            **value,
            "resources": [
                {**first, "controller_final_teardown": True},
                value["resources"][1],
            ],
        },
    ):
        with pytest.raises(ValueError):
            module.validate(mutation)


def test_cleanup_has_no_fixed_name_or_alternative_controls_path() -> None:
    controller = (SPIKE / "scripts/probe_podman_interruption.py").read_text()
    cleanup = (SPIKE / "scripts/cleanup.sh").read_text(encoding="utf-8")
    for forbidden in (
        "aerodocs-8b0-caddy",
        "aerodocs-8b0-admin-password",
        "system prune",
        "--filter",
        "podman image rm",
    ):
        assert forbidden not in cleanup
    assert "cleanup_manifest(podman, controls)" not in controller
    assert '"controller_final_teardown"' in controller


def test_podman_controller_refuses_label_mismatch_and_reports_residual() -> None:
    controller = load(SPIKE / "scripts/probe_podman_interruption.py")
    manifest_module = load(SPIKE / "scripts/resource_manifest.py")
    manifest = manifest_module.validate(manifest_value())
    fake = FakePodman({"authorized-target": "wrong-label"})
    assert controller.cleanup_manifest(fake, manifest, "child_cleanup") == (
        False,
        1,
        1,
    )
    assert fake.removed == []


def test_final_teardown_cannot_remove_targets_or_unmanifested_resources() -> None:
    controller = load(SPIKE / "scripts/probe_podman_interruption.py")
    manifest_module = load(SPIKE / "scripts/resource_manifest.py")
    manifest = manifest_module.validate(manifest_value())
    fake = FakePodman()
    assert controller.cleanup_manifest(fake, manifest, "controller_final_teardown") == (
        True,
        1,
        0,
    )
    assert "authorized-target" in fake.present
    assert "aerodocs-8b0-caddy" in fake.present
    assert fake.removed == [("volume", "aerodocs-8b0-caddy-control")]


def test_podman_controller_has_bounded_signal_escalation_and_no_broad_delete() -> None:
    source = (SPIKE / "scripts/probe_podman_interruption.py").read_text()
    assert "process.send_signal(signal.SIGTERM)" in source
    assert "process.send_signal(signal.SIGINT)" in source
    assert "process.kill()" in source
    assert "process.wait(timeout=5)" in source
    assert "forced_kill_count += 1" in source
    assert "finally:" in source
    assert "controller_fallback_dispatch_count = 1" in source
    for forbidden in (
        "system prune",
        "container prune",
        "network prune",
        "volume prune",
    ):
        assert forbidden not in source


def readiness_frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


@pytest.mark.parametrize(
    "wire",
    [
        readiness_frame(b'{"schema_version":1}') * 2,
        readiness_frame(b'{"schema_version":1}') + b"trailing",
        b"\x00\x00\x00",
        struct.pack(">I", 10) + b"{}",
        struct.pack(">I", 1025) + (b"x" * 1025),
        readiness_frame(b"{"),
        readiness_frame(b'{"schema_version":1,"schema_version":1}'),
    ],
)
def test_readiness_framing_rejects_ambiguous_or_malformed_input(wire: bytes) -> None:
    module = load(SPIKE / "scripts/probe_podman_interruption.py")
    read_descriptor, write_descriptor = os.pipe()
    try:
        os.write(write_descriptor, wire)
        os.close(write_descriptor)
        write_descriptor = -1
        with pytest.raises((module.ReadinessError, TimeoutError)):
            module.read_readiness_frame(read_descriptor, timeout=0.1)
    finally:
        os.close(read_descriptor)
        if write_descriptor >= 0:
            os.close(write_descriptor)


def test_readiness_framing_accepts_one_frame_only_after_eof() -> None:
    module = load(SPIKE / "scripts/probe_podman_interruption.py")
    payload = b'{"schema_version":1,"checkpoint":"expected"}'
    read_descriptor, write_descriptor = os.pipe()
    try:
        os.write(write_descriptor, readiness_frame(payload))
        os.close(write_descriptor)
        write_descriptor = -1
        assert module.read_readiness_frame(read_descriptor, timeout=0.1) == {
            "schema_version": 1,
            "checkpoint": "expected",
        }
    finally:
        os.close(read_descriptor)
        if write_descriptor >= 0:
            os.close(write_descriptor)


def test_readiness_framing_requires_eof() -> None:
    module = load(SPIKE / "scripts/probe_podman_interruption.py")
    read_descriptor, write_descriptor = os.pipe()
    try:
        os.write(write_descriptor, readiness_frame(b'{"schema_version":1}'))
        with pytest.raises(TimeoutError):
            module.read_readiness_frame(read_descriptor, timeout=0.01)
    finally:
        os.close(read_descriptor)
        os.close(write_descriptor)


def test_readiness_framing_rejects_a_later_second_frame() -> None:
    module = load(SPIKE / "scripts/probe_podman_interruption.py")
    frame = readiness_frame(b'{"schema_version":1}')
    read_descriptor, write_descriptor = os.pipe()
    try:
        os.write(write_descriptor, frame)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(module.read_readiness_frame, read_descriptor, 1.0)
            os.write(write_descriptor, frame)
            os.close(write_descriptor)
            write_descriptor = -1
            with pytest.raises(module.ReadinessError):
                future.result(timeout=2)
    finally:
        os.close(read_descriptor)
        if write_descriptor >= 0:
            os.close(write_descriptor)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("readiness_observed", False),
        ("sigterm_delivered", False),
        ("forced_kill_count", 1),
        ("authorized_resources_remaining", 1),
        ("unrelated_resource_preserved", False),
        ("child_cleanup_dispatch_count", 0),
        ("child_cleanup_dispatch_count", 2),
        ("child_resource_cleanup_attempt_count", 1),
        ("controller_verification_count", 0),
        ("controller_verification_count", 2),
        ("controller_fallback_dispatch_count", 1),
        ("controller_fallback_resource_cleanup_attempt_count", 1),
        ("controller_final_teardown_dispatch_count", 0),
        ("controller_final_teardown_resource_cleanup_attempt_count", 1),
        ("manifest_entry_count", 3),
        ("interruption_target_count", 1),
        ("preservation_control_count", 1),
        ("controller_fixture_count", 1),
        ("unauthorized_cleanup_attempt_count", 1),
        ("phase_policy_violation_count", 1),
        ("preservation_controls_unchanged", False),
        ("preservation_verdict_finalized", False),
        ("preservation_evidence_finalized", False),
        ("manifest_unchanged", False),
        ("all_manifest_resources_absent", False),
        ("cleanup_hygiene_passed", False),
    ],
)
def test_podman_abnormal_paths_cannot_pass(field: str, bad_value: bool | int) -> None:
    module = load(SPIKE / "scripts/probe_podman_interruption.py")
    measurements: dict[str, bool | int] = {
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
    assert module.pass_allowed(measurements)
    measurements[field] = bad_value
    assert not module.pass_allowed(measurements)


def test_cleanup_refuses_an_arbitrary_existing_directory() -> None:
    with tempfile.TemporaryDirectory(prefix="aerodocs-cleanup-refusal-") as directory:
        target = Path(directory)
        marker = target / "must-survive"
        marker.write_text("synthetic", encoding="utf-8")
        completed = subprocess.run(
            ["bash", SPIKE / "scripts/cleanup.sh"],
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": str(Path.home()),
                "AERODOCS_SPIKE_RUNTIME": str(target),
                "AERODOCS_SPIKE_NONCE": "synthetic-nonce",
                "AERODOCS_SPIKE_CREATOR_PID": "1",
            },
            capture_output=True,
            check=False,
        )
        assert completed.returncode != 0
        assert marker.read_text(encoding="utf-8") == "synthetic"


def test_ordinary_failure_artifacts_are_retained_and_valid(tmp_path: Path) -> None:
    registry = load(SPIKE / "scripts/gate_registry.py")
    state = tmp_path / "state.json"
    state.write_text(json.dumps(registry.initial()), encoding="utf-8")
    output = tmp_path / "review"
    subprocess.run(
        [
            "python3",
            SPIKE / "scripts/build_failure_artifacts.py",
            "--spike",
            SPIKE,
            "--state",
            state,
            "--gate",
            "health_route",
            "--failure-code",
            "ROUTING",
            "--output",
            output,
        ],
        cwd=ROOT,
        check=True,
    )
    evidence = json.loads((output / "failure-evidence.json").read_text())
    attestation = json.loads((output / "failure-attestation.json").read_text())
    validator("evidence.schema.json").validate(evidence)
    validator("attestation.schema.json").validate(attestation)
    assert attestation["hard_gates"]["health_route"] == "FAIL"
    assert attestation["hard_gates"]["cleanup"] == "NOT_YET_EXECUTED"


def test_unresolved_attestation_validates_with_format_checker() -> None:
    value = json.loads((SPIKE / "examples/attestation.not-executed.json").read_text())
    validator("attestation.schema.json").validate(value)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_caddy",
        "unknown_gate",
        "missing_gate",
        "unresolved_pass",
        "failed_pass",
        "pending_pass",
        "pass_review_on_fail",
        "contradictory",
        "missing_adapted_hash",
    ],
)
def test_attestation_rejects_false_pass_and_contradictions(mutation: str) -> None:
    value = json.loads((SPIKE / "examples/attestation.not-executed.json").read_text())
    digest = "a" * 64
    value["status"] = "PASS"
    value["hard_gates"] = dict.fromkeys(value["hard_gates"], "PASS")
    value["configuration_hashes"] = dict.fromkeys(value["configuration_hashes"], digest)
    value["components"]["caddy"].update(
        archive_sha256=digest,
        binary_sha256=digest,
        sigstore_identity="https://github.com/caddyserver/caddy/.github/workflows/release.yml@refs/tags/v2.11.4",
        sigstore_issuer="https://token.actions.githubusercontent.com",
        transparency_log_verified=True,
    )
    value["components"]["oauth2_proxy"].update(
        archive_sha256=digest, binary_sha256=digest
    )
    value["components"]["keycloak"].update(
        oci_index_digest="sha256:" + digest,
        amd64_manifest_digest="sha256:" + digest,
        config_digest="sha256:" + digest,
        created="2026-08-03T00:00:00Z",
    )
    value["review"] = {"reviewed": True, "decision": "PASS"}
    if mutation == "wrong_caddy":
        value["components"]["caddy"]["version"] = "2.11.5"
    elif mutation == "unknown_gate":
        value["hard_gates"]["fabricated"] = "PASS"
    elif mutation == "missing_gate":
        value["hard_gates"].pop("cleanup")
    elif mutation == "unresolved_pass":
        value["components"]["keycloak"]["config_digest"] = "UNRESOLVED"
    elif mutation == "failed_pass":
        value["hard_gates"]["cleanup"] = "FAIL"
    elif mutation == "pending_pass":
        value["review"] = {"reviewed": False, "decision": "PENDING"}
    elif mutation == "pass_review_on_fail":
        value.update(status="FAIL", failed_gates=["cleanup"], failure_code="CLEANUP")
    elif mutation == "contradictory":
        value.update(status="NOT_YET_EXECUTED")
    elif mutation == "missing_adapted_hash":
        value["configuration_hashes"]["adapted_caddy_json_sha256"] = "UNRESOLVED"
    with pytest.raises(ValidationError):
        validator("attestation.schema.json").validate(value)


def test_valid_fail_attestation_and_evidence() -> None:
    base = json.loads((SPIKE / "examples/attestation.not-executed.json").read_text())
    base.update(
        status="FAIL",
        failed_gates=["configuration_hashes" if False else "adapted_caddy_validation"],
        failure_code="CONFIGURATION",
        review={"reviewed": False, "decision": "FAIL"},
    )
    base["hard_gates"]["adapted_caddy_validation"] = "FAIL"
    validator("attestation.schema.json").validate(base)
    event = {
        "schema_version": 2,
        "test_id": "synthetic.fail",
        "observed_at": "2026-08-03T00:00:00Z",
        "result": "FAIL",
        "measurements": {"field_count": 4, "safe": False},
        "failure_code": "INCONCLUSIVE",
    }
    validator("evidence.schema.json").validate(event)


def test_secret_scanner_reports_category_and_location_only() -> None:
    module = load(SPIKE / "scripts/validate_artifact.py")
    value = "Bearer synthetic-sensitive-value"
    findings = module.scan({"bad": "Authorization: " + value})
    assert findings == [{"category": "authorization", "location": "$.bad"}]
    assert value not in json.dumps(findings)


def test_realm_is_secret_free_pkce_rs256_and_synthetic() -> None:
    realm = json.loads((SPIKE / "keycloak/aerodocs-spike-realm.json").read_text())
    encoded = json.dumps(realm).lower()
    assert '"secret"' not in encoded and '"credentials"' not in encoded
    assert realm["defaultSignatureAlgorithm"] == "RS256"
    assert realm["clients"][0]["attributes"]["pkce.code.challenge.method"] == "S256"
    assert realm["eventsEnabled"] is False and realm["adminEventsEnabled"] is False


def test_shell_and_python_scripts_parse() -> None:
    for script in (SPIKE / "scripts").glob("*.sh"):
        subprocess.run(["bash", "-n", script], check=True)
    for script in list((SPIKE / "scripts").glob("*.py")) + list(SPIKE.glob("*.py")):
        subprocess.run(["python3", "-m", "py_compile", script], check=True)


def test_harness_cannot_claim_partial_pass() -> None:
    text = (SPIKE / "scripts/run-ubuntu-spike.sh").read_text()
    assert "All automated gates passed" not in text
    assert "--finalize --live-started" in text
    assert (
        "caddy.adapted.json" in text and "caddy adapt" not in text
    )  # invoked via variable
    assert "--config-test" in text
