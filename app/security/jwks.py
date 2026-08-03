"""Bounded HTTPS JWKS retrieval and instance-local key caching."""

from __future__ import annotations

import asyncio
import base64
import binascii
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import httpx
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

from app.security.models import (
    IdentityValidationUnavailableError,
    InvalidCredentialsError,
)

MAXIMUM_JWKS_BYTES = 256 * 1024
MAXIMUM_JWKS_KEYS = 32
MAXIMUM_NEGATIVE_KEYS = 256
NEGATIVE_KEY_TTL_SECONDS = 30.0
MINIMUM_REFRESH_INTERVAL_SECONDS = 30.0
MAXIMUM_REFRESH_WAITERS = 128
DEFAULT_POSITIVE_TTL_SECONDS = 300.0
MAXIMUM_POSITIVE_TTL_SECONDS = 900.0
_SAFE_KID = re.compile(r"[A-Za-z0-9._~-]{1,128}")
_PRIVATE_RSA_FIELDS = frozenset({"d", "p", "q", "dp", "dq", "qi", "oth"})
_ALLOWED_MEDIA_TYPES = frozenset({"application/jwk-set+json", "application/json"})
_TOKEN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")


@dataclass(frozen=True)
class JwksHttpResponse:
    """Bounded transport response consumed by the resolver."""

    status_code: int
    content_type_values: tuple[str, ...]
    cache_control_values: tuple[str, ...]
    body: bytes


class JwksTransport(Protocol):
    """Injectable bounded JWKS transport."""

    async def fetch(self, url: str) -> JwksHttpResponse: ...

    async def close(self) -> None: ...


class HttpxJwksTransport:
    """HTTPX transport with explicit category and total deadlines."""

    def __init__(self) -> None:
        timeout = httpx.Timeout(connect=2.0, read=2.0, write=2.0, pool=1.0)
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)

    async def fetch(self, url: str) -> JwksHttpResponse:
        """Fetch one response without redirects or an oversized body."""
        try:
            async with asyncio.timeout(5.0):
                async with self._client.stream("GET", url) as response:
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(body) + len(chunk) > MAXIMUM_JWKS_BYTES:
                            raise IdentityValidationUnavailableError
                        body.extend(chunk)
                    return JwksHttpResponse(
                        status_code=response.status_code,
                        content_type_values=tuple(
                            response.headers.get_list("content-type")
                        ),
                        cache_control_values=tuple(
                            response.headers.get_list("cache-control")
                        ),
                        body=bytes(body),
                    )
        except IdentityValidationUnavailableError:
            raise
        except (TimeoutError, httpx.HTTPError) as error:
            raise IdentityValidationUnavailableError from error

    async def close(self) -> None:
        """Close pooled HTTP resources."""
        await self._client.aclose()


@dataclass(frozen=True)
class _PositiveCache:
    keys: Mapping[str, RSAPublicKey]
    expires_at: float


class JwksResolver:
    """Resolve verification keys with bounded refresh and negative state."""

    def __init__(
        self,
        url: str,
        transport: JwksTransport,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = url
        self.transport = transport
        self.clock = clock
        self._lock = threading.Lock()
        self._positive: _PositiveCache | None = None
        self._negative: OrderedDict[str, tuple[float, int]] = OrderedDict()
        self._negative_sequence = 0
        self._last_refresh_completed: float | None = None
        self._refresh_task: asyncio.Task[None] | None = None
        self._refresh_waiters = 0

    async def preflight(self) -> None:
        """Attempt one initial bounded refresh without preventing app startup."""
        try:
            await self._refresh_if_permitted(initial=True)
        except (IdentityValidationUnavailableError, InvalidCredentialsError):
            return

    async def key_for(self, kid: str) -> RSAPublicKey:
        """Return a current key or a fixed typed validation failure."""
        if not _SAFE_KID.fullmatch(kid):
            raise InvalidCredentialsError
        now = self.clock()
        with self._lock:
            self._purge_negative(now)
            positive = self._positive
            if positive is not None and positive.expires_at > now:
                known = positive.keys.get(kid)
                if known is not None:
                    return known
                if kid in self._negative or not self._refresh_permitted(now):
                    raise InvalidCredentialsError
                missing_from_current = True
            else:
                missing_from_current = False
                if not self._refresh_permitted(now):
                    raise IdentityValidationUnavailableError

        await self._refresh_if_permitted(initial=False)
        now = self.clock()
        with self._lock:
            positive = self._positive
            if positive is None or positive.expires_at <= now:
                raise IdentityValidationUnavailableError
            key = positive.keys.get(kid)
            if key is not None:
                return key
            if missing_from_current or kid not in self._negative:
                self._insert_negative(kid, now)
        raise InvalidCredentialsError

    async def close(self) -> None:
        """Close the resolver transport."""
        await self.transport.close()

    def status_available(self) -> bool:
        """Return whether an unexpired keyset is cached without refreshing."""
        with self._lock:
            return (
                self._positive is not None and self._positive.expires_at > self.clock()
            )

    async def _refresh_if_permitted(self, *, initial: bool) -> None:
        task: asyncio.Task[None]
        waiter = False
        with self._lock:
            now = self.clock()
            current = self._refresh_task
            if current is not None and not current.done():
                if self._refresh_waiters >= MAXIMUM_REFRESH_WAITERS:
                    raise IdentityValidationUnavailableError
                self._refresh_waiters += 1
                task = current
                waiter = True
            else:
                initial_refresh = initial and self._last_refresh_completed is None
                if not initial_refresh and not self._refresh_permitted(now):
                    raise IdentityValidationUnavailableError
                task = asyncio.create_task(self._perform_refresh())
                self._refresh_task = task
        try:
            await asyncio.shield(task)
        except (IdentityValidationUnavailableError, InvalidCredentialsError):
            raise
        except Exception as error:
            raise IdentityValidationUnavailableError from error
        finally:
            with self._lock:
                if waiter:
                    self._refresh_waiters -= 1
                if self._refresh_task is task and task.done():
                    self._refresh_task = None

    async def _perform_refresh(self) -> None:
        try:
            response = await self.transport.fetch(self.url)
            if response.status_code != 200:
                raise IdentityValidationUnavailableError
            validate_jwks_content_type(response.content_type_values)
            if len(response.body) > MAXIMUM_JWKS_BYTES:
                raise IdentityValidationUnavailableError
            keys = parse_jwks(response.body)
            ttl = _positive_ttl(response.cache_control_values)
            completed = self.clock()
            with self._lock:
                self._positive = _PositiveCache(
                    keys=keys,
                    expires_at=completed + ttl,
                )
                for kid in keys:
                    self._negative.pop(kid, None)
        finally:
            with self._lock:
                self._last_refresh_completed = self.clock()

    def _refresh_permitted(self, now: float) -> bool:
        return (
            self._last_refresh_completed is None
            or now - self._last_refresh_completed >= MINIMUM_REFRESH_INTERVAL_SECONDS
        )

    def _purge_negative(self, now: float) -> None:
        expired = sorted(
            (
                (kid, expiry)
                for kid, (expiry, _sequence) in self._negative.items()
                if expiry <= now
            ),
            key=lambda item: (item[1], item[0]),
        )
        for kid, _expiry in expired:
            self._negative.pop(kid, None)

    def _insert_negative(self, kid: str, now: float) -> None:
        self._purge_negative(now)
        self._negative.pop(kid, None)
        while len(self._negative) >= MAXIMUM_NEGATIVE_KEYS:
            oldest = min(
                self._negative,
                key=lambda item: (self._negative[item][1], item),
            )
            del self._negative[oldest]
        self._negative_sequence += 1
        self._negative[kid] = (
            now + NEGATIVE_KEY_TTL_SECONDS,
            self._negative_sequence,
        )


def validate_jwks_content_type(values: tuple[str, ...]) -> None:
    """Require one unambiguous approved JSON media type."""
    if len(values) != 1 or not values[0].strip():
        raise IdentityValidationUnavailableError
    raw = values[0]
    if "\r" in raw or "\n" in raw or "," in raw:
        raise IdentityValidationUnavailableError
    parts = _split_content_type(raw)
    media = parts[0].strip()
    if "/" not in media:
        raise IdentityValidationUnavailableError
    major, minor = media.split("/", 1)
    if not _TOKEN.fullmatch(major) or not _TOKEN.fullmatch(minor):
        raise IdentityValidationUnavailableError
    if f"{major.lower()}/{minor.lower()}" not in _ALLOWED_MEDIA_TYPES:
        raise IdentityValidationUnavailableError
    seen: set[str] = set()
    for parameter in parts[1:]:
        if "=" not in parameter:
            raise IdentityValidationUnavailableError
        name, raw_value = parameter.split("=", 1)
        name = name.strip()
        raw_value = raw_value.strip()
        if not _TOKEN.fullmatch(name):
            raise IdentityValidationUnavailableError
        value = _parameter_value(raw_value)
        lowered = name.lower()
        if lowered in seen:
            raise IdentityValidationUnavailableError
        seen.add(lowered)
        if lowered == "charset" and value.lower() != "utf-8":
            raise IdentityValidationUnavailableError


def _split_content_type(raw: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quoted = False
    escaped = False
    for character in raw:
        if escaped:
            if ord(character) < 0x20 or ord(character) == 0x7F:
                raise IdentityValidationUnavailableError
            current.append(character)
            escaped = False
        elif quoted and character == "\\":
            current.append(character)
            escaped = True
        elif character == '"':
            quoted = not quoted
            current.append(character)
        elif character == ";" and not quoted:
            part = "".join(current).strip()
            if not part:
                raise IdentityValidationUnavailableError
            parts.append(part)
            current = []
        else:
            if ord(character) < 0x20 or ord(character) == 0x7F:
                raise IdentityValidationUnavailableError
            current.append(character)
    if quoted or escaped:
        raise IdentityValidationUnavailableError
    final = "".join(current).strip()
    if not final:
        raise IdentityValidationUnavailableError
    parts.append(final)
    return parts


def _parameter_value(raw: str) -> str:
    if _TOKEN.fullmatch(raw):
        return raw
    if len(raw) < 2 or raw[0] != '"' or raw[-1] != '"':
        raise IdentityValidationUnavailableError
    result: list[str] = []
    escaped = False
    for character in raw[1:-1]:
        if escaped:
            result.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"' or ord(character) < 0x20 or ord(character) == 0x7F:
            raise IdentityValidationUnavailableError
        else:
            result.append(character)
    if escaped:
        raise IdentityValidationUnavailableError
    return "".join(result)


def parse_jwks(body: bytes) -> Mapping[str, RSAPublicKey]:
    """Parse one duplicate-free bounded RSA verification keyset."""
    from app.security.jwt import duplicate_safe_json

    try:
        parsed = duplicate_safe_json(
            body,
            maximum_bytes=MAXIMUM_JWKS_BYTES,
            maximum_object_members=1 + MAXIMUM_JWKS_KEYS * 16,
        )
    except InvalidCredentialsError as error:
        raise IdentityValidationUnavailableError from error
    if not isinstance(parsed, dict) or set(parsed) != {"keys"}:
        raise IdentityValidationUnavailableError
    raw_keys = parsed["keys"]
    if not isinstance(raw_keys, list) or not 1 <= len(raw_keys) <= MAXIMUM_JWKS_KEYS:
        raise IdentityValidationUnavailableError
    result: dict[str, RSAPublicKey] = {}
    for raw in raw_keys:
        if not isinstance(raw, dict):
            raise IdentityValidationUnavailableError
        key = cast(dict[str, object], raw)
        if _PRIVATE_RSA_FIELDS & key.keys():
            raise IdentityValidationUnavailableError
        kid = key.get("kid")
        if not isinstance(kid, str) or not _SAFE_KID.fullmatch(kid) or kid in result:
            raise IdentityValidationUnavailableError
        if key.get("kty") != "RSA" or key.get("alg", "RS256") != "RS256":
            raise IdentityValidationUnavailableError
        if key.get("use", "sig") != "sig":
            raise IdentityValidationUnavailableError
        operations = key.get("key_ops", ["verify"])
        if operations != ["verify"]:
            raise IdentityValidationUnavailableError
        modulus = _unsigned_integer(key.get("n"))
        exponent = _unsigned_integer(key.get("e"))
        if not 2_048 <= modulus.bit_length() <= 8_192:
            raise IdentityValidationUnavailableError
        if exponent < 65_537 or exponent > 0xFFFFFFFF or exponent % 2 == 0:
            raise IdentityValidationUnavailableError
        try:
            result[kid] = rsa.RSAPublicNumbers(exponent, modulus).public_key()
        except ValueError as error:
            raise IdentityValidationUnavailableError from error
    return result


def _unsigned_integer(value: object) -> int:
    if not isinstance(value, str) or not value or "=" in value:
        raise IdentityValidationUnavailableError
    try:
        padding = "=" * ((4 - len(value) % 4) % 4)
        decoded = base64.b64decode(
            (value + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise IdentityValidationUnavailableError from error
    if not decoded or decoded[0] == 0:
        raise IdentityValidationUnavailableError
    return int.from_bytes(decoded, "big")


def _positive_ttl(values: tuple[str, ...]) -> float:
    if len(values) != 1:
        return DEFAULT_POSITIVE_TTL_SECONDS
    matches = re.findall(r"(?:^|,)\s*max-age\s*=\s*(\d+)\s*(?:,|$)", values[0], re.I)
    if len(matches) != 1:
        return DEFAULT_POSITIVE_TTL_SECONDS
    try:
        value = int(matches[0])
    except ValueError:
        return DEFAULT_POSITIVE_TTL_SECONDS
    if value <= 0:
        return DEFAULT_POSITIVE_TTL_SECONDS
    return float(min(value, int(MAXIMUM_POSITIVE_TTL_SECONDS)))
