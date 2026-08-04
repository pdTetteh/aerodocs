#!/usr/bin/env python3
"""Run the secret-safe Keycloak 26.7.0 OIDC early compatibility gate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html.parser
import http.client
import http.cookiejar
import json
import secrets
import ssl
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Final

EXPECTED_ISSUER: Final = "https://idp.aerodocs-spike.invalid/realms/aerodocs-spike"
EXPECTED_AUDIENCE: Final = "urn:aerodocs:spike:api"
EXPECTED_GROUPS: Final = ["aerodocs-spike-viewers"]
CLIENT_ID: Final = "aerodocs-spike-browser"
REDIRECT_URI: Final = "https://aerodocs-spike.invalid/oauth2/callback"
MAX_DOCUMENT_BYTES: Final = 256 * 1024


class GateFailure(RuntimeError):
    """A non-disclosing early-gate failure."""


class LoginFormParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.action: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "form":
            return
        values = dict(attrs)
        if values.get("id") == "kc-form-login" and values.get("action"):
            self.action = values["action"]


class CallbackCapture(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        if newurl.startswith(REDIRECT_URI):
            raise GateFailure("callback_captured:" + newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _b64_json(segment: str) -> dict[str, object]:
    try:
        raw = base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
        value = json.loads(raw)
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateFailure("token_structure") from exc
    if not isinstance(value, dict):
        raise GateFailure("token_structure")
    return value


def _read_json(response: object) -> dict[str, object]:
    raw = response.read(MAX_DOCUMENT_BYTES + 1)  # type: ignore[attr-defined]
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise GateFailure("response_too_large")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GateFailure("invalid_json") from exc
    if not isinstance(value, dict):
        raise GateFailure("invalid_json")
    return value


def _same_origin(url: object) -> bool:
    if not isinstance(url, str):
        return False
    expected = urllib.parse.urlsplit(EXPECTED_ISSUER)
    actual = urllib.parse.urlsplit(url)
    return (actual.scheme, actual.hostname, actual.port) == (
        expected.scheme,
        expected.hostname,
        expected.port,
    )


def _assert_pkce_rejected(
    opener: urllib.request.OpenerDirector,
    authorization_endpoint: str,
    method: str | None,
) -> None:
    parameters = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid",
        "state": secrets.token_urlsafe(16),
    }
    if method is not None:
        parameters["code_challenge"] = "synthetic-plain-challenge"
        parameters["code_challenge_method"] = method
    try:
        response = opener.open(
            authorization_endpoint + "?" + urllib.parse.urlencode(parameters),
            timeout=5,
        )
    except urllib.error.HTTPError as exc:
        if 400 <= exc.code < 500:
            return
        raise GateFailure("pkce_enforcement") from exc
    if "error=" in response.geturl():
        return
    raise GateFailure("pkce_enforcement")


def run(
    cafile: Path, username_file: Path, password_file: Path, secret_file: Path
) -> dict[str, object]:
    context = ssl.create_default_context(cafile=str(cafile))
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar),
        urllib.request.HTTPSHandler(context=context),
        CallbackCapture(),
    )
    discovery = _read_json(
        opener.open(EXPECTED_ISSUER + "/.well-known/openid-configuration", timeout=5)
    )
    if discovery.get("issuer") != EXPECTED_ISSUER:
        raise GateFailure("issuer")
    endpoint_names = ("authorization_endpoint", "token_endpoint", "jwks_uri")
    if not all(_same_origin(discovery.get(name)) for name in endpoint_names):
        raise GateFailure("endpoint_origin")
    methods = discovery.get("code_challenge_methods_supported")
    if not isinstance(methods, list) or "S256" not in methods:
        raise GateFailure("pkce_metadata")

    authorization_endpoint = str(discovery["authorization_endpoint"])
    _assert_pkce_rejected(opener, authorization_endpoint, None)
    _assert_pkce_rejected(opener, authorization_endpoint, "plain")

    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    state = secrets.token_urlsafe(24)
    query = urllib.parse.urlencode(
        {
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "scope": "openid profile groups offline_access",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    login = opener.open(authorization_endpoint + "?" + query, timeout=5)
    raw_login = login.read(MAX_DOCUMENT_BYTES + 1)
    if len(raw_login) > MAX_DOCUMENT_BYTES:
        raise GateFailure("login_too_large")
    parser = LoginFormParser()
    parser.feed(raw_login.decode("utf-8", "strict"))
    if parser.action is None or not _same_origin(parser.action):
        raise GateFailure("login_form")

    username = username_file.read_text(encoding="utf-8").strip()
    password = password_file.read_text(encoding="utf-8").strip()
    if not username or not password:
        raise GateFailure("runtime_secret")
    body = urllib.parse.urlencode({"username": username, "password": password}).encode()
    callback_url = ""
    try:
        opener.open(urllib.request.Request(parser.action, data=body), timeout=5)
    except GateFailure as exc:
        marker = "callback_captured:"
        if not str(exc).startswith(marker):
            raise
        callback_url = str(exc)[len(marker) :]
    callback = urllib.parse.parse_qs(urllib.parse.urlsplit(callback_url).query)
    if callback.get("state") != [state] or len(callback.get("code", [])) != 1:
        raise GateFailure("authorization_callback")

    client_secret = secret_file.read_text(encoding="utf-8").strip()
    token_body = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "client_secret": client_secret,
            "redirect_uri": REDIRECT_URI,
            "code": callback["code"][0],
            "code_verifier": verifier,
        }
    ).encode()
    token_response = _read_json(
        opener.open(
            urllib.request.Request(str(discovery["token_endpoint"]), data=token_body),
            timeout=5,
        )
    )
    access_token = token_response.get("access_token")
    refresh_token = token_response.get("refresh_token")
    if not isinstance(access_token, str) or not isinstance(refresh_token, str):
        raise GateFailure("token_response")
    parts = access_token.split(".")
    if len(parts) != 3:
        raise GateFailure("token_structure")
    header = _b64_json(parts[0])
    claims = _b64_json(parts[1])
    if header.get("alg") != "RS256":
        raise GateFailure("token_algorithm")
    audience = claims.get("aud")
    audiences = audience if isinstance(audience, list) else [audience]
    if EXPECTED_AUDIENCE not in audiences:
        raise GateFailure("token_audience")
    if claims.get("groups") != EXPECTED_GROUPS:
        raise GateFailure("token_groups")
    if len(access_token.encode("ascii")) > 3072:
        raise GateFailure("token_size")

    refresh_body = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        }
    ).encode()
    refreshed = _read_json(
        opener.open(
            urllib.request.Request(str(discovery["token_endpoint"]), data=refresh_body),
            timeout=5,
        )
    )
    if not isinstance(refreshed.get("access_token"), str):
        raise GateFailure("refresh")
    return {
        "issuer_exact": True,
        "endpoint_origins_exact": True,
        "algorithm_rs256": True,
        "audience_exact": True,
        "groups_count": len(EXPECTED_GROUPS),
        "pkce_s256_advertised": True,
        "refresh_succeeded": True,
        "token_bytes": len(access_token.encode("ascii")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ca-file", type=Path, required=True)
    parser.add_argument("--username-file", type=Path, required=True)
    parser.add_argument("--password-file", type=Path, required=True)
    parser.add_argument("--client-secret-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        measurements = run(
            args.ca_file,
            args.username_file,
            args.password_file,
            args.client_secret_file,
        )
    except (GateFailure, OSError, urllib.error.URLError):
        print(
            json.dumps(
                {
                    "schema_version": 2,
                    "test_id": "oidc_early_gate",
                    "observed_at": datetime.now(UTC)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "result": "FAIL",
                    "measurements": {"checked": True},
                    "failure_code": "OIDC",
                    "gate_results": {
                        "issuer_endpoint_confinement": {
                            "result": "FAIL",
                            "measurements": {"checked": True},
                        }
                    },
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None
    gates = (
        "issuer_endpoint_confinement",
        "rs256",
        "exact_audience",
        "bounded_groups",
        "pkce_s256",
        "refresh_support",
    )
    print(
        json.dumps(
            {
                "schema_version": 2,
                "test_id": "oidc_early_gate",
                "observed_at": datetime.now(UTC)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "result": "PASS",
                "measurements": measurements,
                "failure_code": "NONE",
                "gate_results": {
                    gate: {"result": "PASS", "measurements": {"checked": True}}
                    for gate in gates
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
