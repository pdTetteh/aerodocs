"""Strict raw-ASGI direct bearer extraction."""

import re
from typing import cast

from starlette.types import Scope

from app.security.models import InvalidCredentialsError

MAXIMUM_DIRECT_BEARER_BYTES = 16_384
_COMPACT_JWT = re.compile(rb"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")


def direct_bearer_token(scope: Scope) -> str | None:
    """Return one strictly formed direct bearer token from raw ASGI headers."""
    values = cast(
        list[bytes],
        [
            value
            for name, value in scope.get("headers", [])
            if name.lower() == b"authorization"
        ],
    )
    if not values:
        return None
    if len(values) != 1:
        raise InvalidCredentialsError
    value = values[0]
    if not value or b"," in value or any(byte < 0x20 or byte == 0x7F for byte in value):
        raise InvalidCredentialsError
    try:
        scheme, token = value.split(b" ")
    except ValueError as error:
        raise InvalidCredentialsError from error
    if scheme.lower() != b"bearer" or not token:
        raise InvalidCredentialsError
    if len(token) > MAXIMUM_DIRECT_BEARER_BYTES or not _COMPACT_JWT.fullmatch(token):
        raise InvalidCredentialsError
    try:
        return token.decode("ascii")
    except UnicodeDecodeError as error:
        raise InvalidCredentialsError from error
