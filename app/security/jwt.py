"""Duplicate-safe RS256 access-token validation."""

import base64
import binascii
import json
import math
import re
import unicodedata
from collections.abc import Callable
from typing import cast

import jwt

from app.core.config import Settings
from app.security.jwks import JwksResolver
from app.security.models import (
    CredentialSource,
    InvalidCredentialsError,
    Principal,
    Role,
)

JsonObject = dict[str, object]
_SAFE_KID = re.compile(r"[A-Za-z0-9._~-]{1,128}")
_HEADER_KEYS = frozenset({"alg", "kid", "typ"})
MAXIMUM_JWT_JSON_BYTES = 12_288
MAXIMUM_JSON_NESTING_DEPTH = 16
MAXIMUM_JSON_OBJECT_MEMBERS = 128
MAXIMUM_JSON_ARRAY_ELEMENTS = 128
MAXIMUM_JSON_CONTAINERS = 64


def duplicate_safe_json(
    data: bytes,
    *,
    maximum_bytes: int = MAXIMUM_JWT_JSON_BYTES,
    maximum_object_members: int = MAXIMUM_JSON_OBJECT_MEMBERS,
) -> object:
    """Decode one structurally bounded UTF-8 JSON value without duplicates."""

    def object_pairs(pairs: list[tuple[str, object]]) -> JsonObject:
        result: JsonObject = {}
        for key, value in pairs:
            if key in result:
                raise InvalidCredentialsError
            result[key] = value
        return result

    def invalid_constant(_value: str) -> object:
        raise InvalidCredentialsError

    try:
        if len(data) > maximum_bytes:
            raise InvalidCredentialsError
        text = data.decode("utf-8")
        _scan_json_structure(
            text,
            maximum_object_members=maximum_object_members,
        )
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except InvalidCredentialsError:
        raise
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidCredentialsError from error


def _scan_json_structure(text: str, *, maximum_object_members: int) -> None:
    """Scan JSON containers iteratively before recursive standard decoding."""
    # Each entry is [container delimiter, array is waiting for its next value].
    stack: list[list[str | bool]] = []
    containers = 0
    object_members = 0
    array_elements = 0
    quoted = False
    escaped = False

    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue

        if stack and stack[-1][0] == "[" and stack[-1][1]:
            if not character.isspace() and character not in {",", "]"}:
                array_elements += 1
                if array_elements > MAXIMUM_JSON_ARRAY_ELEMENTS:
                    raise InvalidCredentialsError
                stack[-1][1] = False

        if character == '"':
            quoted = True
        elif character in "[{":
            containers += 1
            if containers > MAXIMUM_JSON_CONTAINERS:
                raise InvalidCredentialsError
            if len(stack) >= MAXIMUM_JSON_NESTING_DEPTH:
                raise InvalidCredentialsError
            stack.append([character, character == "["])
        elif character in "]}":
            if stack and (
                (character == "]" and stack[-1][0] == "[")
                or (character == "}" and stack[-1][0] == "{")
            ):
                stack.pop()
        elif character == "," and stack and stack[-1][0] == "[":
            stack[-1][1] = True
        elif character == ":" and stack and stack[-1][0] == "{":
            object_members += 1
            if object_members > maximum_object_members:
                raise InvalidCredentialsError


def _segment_json(segment: str) -> JsonObject:
    try:
        padding = "=" * ((4 - len(segment) % 4) % 4)
        decoded = base64.b64decode(
            (segment + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise InvalidCredentialsError from error
    parsed = duplicate_safe_json(decoded)
    if not isinstance(parsed, dict):
        raise InvalidCredentialsError
    return parsed


def preparse_access_token(token: str) -> tuple[JsonObject, JsonObject]:
    """Preparse protected header and payload before cryptographic acceptance."""
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
    except ValueError as error:
        raise InvalidCredentialsError from error
    if not header_segment or not payload_segment or not signature_segment:
        raise InvalidCredentialsError
    header = _segment_json(header_segment)
    payload = _segment_json(payload_segment)
    if not header.keys() <= _HEADER_KEYS:
        raise InvalidCredentialsError
    if header.get("alg") != "RS256":
        raise InvalidCredentialsError
    kid = header.get("kid")
    if not isinstance(kid, str) or not _SAFE_KID.fullmatch(kid):
        raise InvalidCredentialsError
    token_type = header.get("typ")
    if token_type is not None and token_type not in {"JWT", "at+jwt"}:
        raise InvalidCredentialsError
    return header, payload


class OidcAccessTokenValidator:
    """Validate the strict AeroDocs access-token contract."""

    def __init__(
        self,
        settings: Settings,
        resolver: JwksResolver,
        *,
        clock: Callable[[], float],
    ) -> None:
        self.settings = settings
        self.resolver = resolver
        self.clock = clock

    async def validate(self, token: str) -> Principal:
        """Return a principal for one verified RS256 access token."""
        header, unverified = preparse_access_token(token)
        self._validate_claim_structure(unverified)
        kid = cast(str, header["kid"])
        key = await self.resolver.key_for(kid)
        try:
            payload = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                audience=self.settings.oidc_audience,
                issuer=self.settings.oidc_issuer,
                leeway=self.settings.oidc_clock_skew_seconds,
                options={
                    "require": ["iss", "aud", "sub", "iat", "nbf", "exp", "groups"],
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iat": True,
                    "verify_aud": True,
                    "verify_iss": True,
                },
            )
        except jwt.PyJWTError as error:
            raise InvalidCredentialsError from error
        if payload != unverified:
            raise InvalidCredentialsError
        self._validate_claim_structure(payload)
        roles: set[Role] = set()
        groups = cast(list[str], payload["groups"])
        if self.settings.oidc_viewer_group in groups:
            roles.add(Role.EVIDENCE_VIEWER)
        if self.settings.oidc_operator_group in groups:
            roles.add(Role.DOCUMENT_OPERATOR)
        return Principal(
            subject=cast(str, payload["sub"]),
            roles=frozenset(roles),
            credential_source=CredentialSource.DIRECT_BEARER,
        )

    def _validate_claim_structure(self, payload: JsonObject) -> None:
        required = {"iss", "aud", "sub", "iat", "nbf", "exp", "groups"}
        if not required <= payload.keys():
            raise InvalidCredentialsError
        if payload["iss"] != self.settings.oidc_issuer:
            raise InvalidCredentialsError
        if payload["aud"] != self.settings.oidc_audience:
            raise InvalidCredentialsError
        subject = payload["sub"]
        if not isinstance(subject, str) or not _safe_text(subject):
            raise InvalidCredentialsError
        times: list[int] = []
        for name in ("iat", "nbf", "exp"):
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise InvalidCredentialsError
            times.append(value)
        issued, not_before, expires = times
        if not (
            0 < expires - issued <= self.settings.oidc_maximum_token_lifetime_seconds
        ):
            raise InvalidCredentialsError
        if not_before > expires:
            raise InvalidCredentialsError
        if issued > math.floor(self.clock()) + self.settings.oidc_clock_skew_seconds:
            raise InvalidCredentialsError
        groups = payload["groups"]
        if not isinstance(groups, list) or len(groups) > 32:
            raise InvalidCredentialsError
        if any(not isinstance(group, str) or not _safe_text(group) for group in groups):
            raise InvalidCredentialsError
        if len(set(cast(list[str], groups))) != len(groups):
            raise InvalidCredentialsError


def _safe_text(value: str) -> bool:
    if not 1 <= len(value) <= 128 or len(value.encode("utf-8")) > 512:
        return False
    if value != value.strip():
        return False
    return all(
        unicodedata.category(character) not in {"Cc", "Cf", "Cs", "Cn"}
        for character in value
    )
