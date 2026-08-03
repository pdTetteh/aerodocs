"""Deterministic offline tests for Milestone 8 Phase 8A security."""

from __future__ import annotations

import base64
import json
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Never, cast

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.routing import BaseRoute, Mount, Route

import app.security.routes as security_routes
from app.core.config import Settings
from app.factory import create_app, create_phase_8a_secure_app
from app.security.jwks import (
    MAXIMUM_JWKS_BYTES,
    MAXIMUM_NEGATIVE_KEYS,
    HttpxJwksTransport,
    JwksHttpResponse,
    JwksResolver,
    parse_jwks,
    validate_jwks_content_type,
)
from app.security.jwt import (
    MAXIMUM_JSON_ARRAY_ELEMENTS,
    MAXIMUM_JSON_CONTAINERS,
    MAXIMUM_JSON_NESTING_DEPTH,
    MAXIMUM_JSON_OBJECT_MEMBERS,
    duplicate_safe_json,
    preparse_access_token,
)
from app.security.models import (
    CredentialSource,
    IdentityValidationUnavailableError,
    InvalidCredentialsError,
    Principal,
    Role,
)
from app.security.routes import (
    Permission,
    RouteInventoryError,
    RouteKind,
    _flatten,
    audit_phase_8a_routes,
)
from app.ui.errors import UIConfigurationError

VALID_OIDC_AUDIENCE = "urn:aerodocs:deployment:alpha:api"
VALID_VIEWER_GROUP = "aerodocs-deployment-alpha-evidence-readers"
VALID_OPERATOR_GROUP = "aerodocs-deployment-alpha-document-operators"


@dataclass
class FakeClock:
    value: float

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeTransport:
    def __init__(self, outcomes: list[JwksHttpResponse | Exception]) -> None:
        self.outcomes = deque(outcomes)
        self.calls = 0
        self.closed = False

    async def fetch(self, _url: str) -> JwksHttpResponse:
        self.calls += 1
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def close(self) -> None:
        self.closed = True


class StaticValidator:
    def __init__(self, principal: Principal | None = None) -> None:
        self.principal = principal
        self.tokens: list[str] = []

    async def validate(self, token: str) -> Principal:
        self.tokens.append(token)
        if self.principal is None:
            raise InvalidCredentialsError
        return self.principal


class ParsingValidator:
    async def validate(self, token: str) -> Principal:
        preparse_access_token(token)
        raise AssertionError("structurally invalid token was accepted")


def oidc_settings(default_settings: Settings, **updates: object) -> Settings:
    return default_settings.model_copy(
        update={
            "security_mode": "oidc",
            "oidc_issuer": "https://identity.example.test/",
            "oidc_audience": VALID_OIDC_AUDIENCE,
            "oidc_jwks_url": "https://identity.example.test/jwks.json",
            "oidc_viewer_group": VALID_VIEWER_GROUP,
            "oidc_operator_group": VALID_OPERATOR_GROUP,
            **updates,
        }
    )


def test_phase_8a_has_exactly_one_credential_source() -> None:
    assert list(CredentialSource) == [CredentialSource.DIRECT_BEARER]


def test_test_mode_requires_injected_validator(default_settings: Settings) -> None:
    settings = default_settings.model_copy(update={"security_mode": "test"})
    with pytest.raises(ValueError, match="injected validator"):
        create_app(settings)


def test_secure_constructor_requires_oidc(default_settings: Settings) -> None:
    with pytest.raises(ValueError, match="requires oidc"):
        create_phase_8a_secure_app(default_settings)


def test_oidc_settings_require_complete_same_origin_configuration(
    default_settings: Settings,
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, security_mode="oidc")  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="origins must match"):
        Settings(
            _env_file=None,
            security_mode="oidc",
            oidc_issuer="https://identity.example.test/",
            oidc_audience=VALID_OIDC_AUDIENCE,
            oidc_jwks_url="https://keys.example.test/jwks",
            oidc_viewer_group=VALID_VIEWER_GROUP,
            oidc_operator_group=VALID_OPERATOR_GROUP,
        )  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "audience",
    [
        "*",
        "api-*",
        "api?",
        "api[production]",
        "aerodocs-api",
        "wrong-api",
        "test",
        "FIXTURE",
        "sample",
        "example",
        "placeholder",
        "TODO",
        "demo",
        " ",
        " leading",
        "trailing ",
        "unsafe\x00audience",
    ],
)
def test_oidc_rejects_wildcard_unsafe_and_shipped_audiences(
    default_settings: Settings,
    audience: str,
) -> None:
    with pytest.raises(ValidationError, match="oidc_audience"):
        Settings(
            _env_file=None,
            security_mode="oidc",
            oidc_issuer="https://identity.deployment.invalid/",
            oidc_audience=audience,
            oidc_jwks_url="https://identity.deployment.invalid/jwks",
            oidc_viewer_group=VALID_VIEWER_GROUP,
            oidc_operator_group=VALID_OPERATOR_GROUP,
        )  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "group",
    [
        "*",
        "group-*",
        "fixture-viewers",
        "fixture-operators",
        "viewer",
        "operator",
        "organisation-viewer-group",
        "organisation-operator-group",
        "",
        " ",
        " leading",
        "trailing ",
        "unsafe\u2060group",
    ],
)
def test_oidc_rejects_wildcard_unsafe_and_shipped_group_mappings(
    default_settings: Settings,
    group: str,
) -> None:
    for update in (
        {"oidc_viewer_group": group},
        {"oidc_operator_group": group},
    ):
        configuration: dict[str, object] = {
            "security_mode": "oidc",
            "oidc_issuer": "https://identity.deployment.invalid/",
            "oidc_audience": VALID_OIDC_AUDIENCE,
            "oidc_jwks_url": "https://identity.deployment.invalid/jwks",
            "oidc_viewer_group": VALID_VIEWER_GROUP,
            "oidc_operator_group": VALID_OPERATOR_GROUP,
            **update,
        }
        with pytest.raises(ValidationError, match="group mappings"):
            Settings(_env_file=None, **configuration)  # type: ignore[call-arg,arg-type]


def test_oidc_rejects_identical_group_mappings() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        Settings(
            _env_file=None,
            security_mode="oidc",
            oidc_issuer="https://identity.deployment.invalid/",
            oidc_audience=VALID_OIDC_AUDIENCE,
            oidc_jwks_url="https://identity.deployment.invalid/jwks",
            oidc_viewer_group=VALID_VIEWER_GROUP,
            oidc_operator_group=VALID_VIEWER_GROUP,
        )  # type: ignore[call-arg]


@pytest.mark.parametrize("missing", ["oidc_viewer_group", "oidc_operator_group"])
def test_oidc_rejects_each_missing_group_mapping(missing: str) -> None:
    configuration: dict[str, object] = {
        "security_mode": "oidc",
        "oidc_issuer": "https://identity.deployment.invalid/",
        "oidc_audience": VALID_OIDC_AUDIENCE,
        "oidc_jwks_url": "https://identity.deployment.invalid/jwks",
        "oidc_viewer_group": VALID_VIEWER_GROUP,
        "oidc_operator_group": VALID_OPERATOR_GROUP,
    }
    configuration[missing] = None
    with pytest.raises(ValidationError, match="complete OIDC settings"):
        Settings(_env_file=None, **configuration)  # type: ignore[call-arg,arg-type]


def test_oidc_accepts_exact_deployment_identifiers_with_ordinary_sequences() -> None:
    settings = Settings(
        _env_file=None,
        security_mode="oidc",
        oidc_issuer="https://identity.deployment.invalid/",
        oidc_audience="urn:contest:testing:api",
        oidc_jwks_url="https://identity.deployment.invalid/jwks",
        oidc_viewer_group="contest-evidence-readers",
        oidc_operator_group="testing-document-operators",
    )  # type: ignore[call-arg]
    assert settings.oidc_audience == "urn:contest:testing:api"


def test_secure_constructor_rejects_invalid_identifiers_without_network(
    default_settings: Settings,
) -> None:
    invalid = oidc_settings(default_settings).model_copy(update={"oidc_audience": "*"})
    transport = FakeTransport([])
    with pytest.raises(ValueError, match="oidc_audience"):
        create_phase_8a_secure_app(invalid, jwks_transport=transport)
    assert transport.calls == 0


def test_disabled_mode_ignores_auth_headers_and_preserves_documentation(
    default_settings: Settings,
) -> None:
    with TestClient(create_app(default_settings)) as client:
        response = client.get("/health", headers={"authorization": "broken"})
        assert response.status_code == 200
        assert client.get("/docs").status_code == 200


def test_direct_bearer_route_permissions_and_fixed_errors(
    default_settings: Settings,
) -> None:
    settings = default_settings.model_copy(update={"security_mode": "test"})
    viewer = StaticValidator(
        Principal(subject="viewer", roles=frozenset({Role.EVIDENCE_VIEWER}))
    )
    with TestClient(create_app(settings, access_token_validator=viewer)) as client:
        missing = client.get("/documents")
        malformed = client.get("/documents", headers={"authorization": "Bearer  a.b.c"})
        allowed = client.get("/ui/", headers={"authorization": "bEaReR a.b.c"})
        denied = client.post(
            "/indexes/rebuild",
            headers={"authorization": "Bearer a.b.c"},
        )
        docs = client.get("/docs", headers={"authorization": "Bearer a.b.c"})

    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "authentication_required"
    assert malformed.status_code == 401
    assert malformed.json()["error"]["code"] == "invalid_credentials"
    assert allowed.status_code == 200
    assert denied.status_code == 403
    assert denied.json()["error"]["code"] == "permission_denied"
    assert docs.status_code == 404
    assert viewer.tokens == ["a.b.c", "a.b.c"]


def test_duplicate_and_combined_authorization_are_rejected(
    default_settings: Settings,
) -> None:
    settings = default_settings.model_copy(update={"security_mode": "test"})
    validator = StaticValidator(
        Principal(subject="viewer", roles=frozenset({Role.EVIDENCE_VIEWER}))
    )
    with TestClient(create_app(settings, access_token_validator=validator)) as client:
        duplicate = client.get(
            "/documents",
            headers=[
                ("authorization", "Bearer a.b.c"),
                ("authorization", "Bearer d.e.f"),
            ],
        )
        combined = client.get(
            "/documents",
            headers={"authorization": "Bearer a.b.c, Bearer d.e.f"},
        )
    assert duplicate.status_code == 401
    assert combined.status_code == 401
    assert validator.tokens == []


def test_oversized_direct_bearer_is_rejected_before_validation(
    default_settings: Settings,
) -> None:
    settings = default_settings.model_copy(update={"security_mode": "test"})
    validator = StaticValidator(
        Principal(subject="viewer", roles=frozenset({Role.EVIDENCE_VIEWER}))
    )
    oversized = "a" * 16_381 + ".b.c"
    with TestClient(create_app(settings, access_token_validator=validator)) as client:
        response = client.get(
            "/documents",
            headers={"authorization": f"Bearer {oversized}"},
        )
    assert response.status_code == 401
    assert validator.tokens == []


def test_gateway_header_is_not_a_phase_8a_credential(
    default_settings: Settings,
) -> None:
    settings = default_settings.model_copy(update={"security_mode": "test"})
    validator = StaticValidator(
        Principal(subject="viewer", roles=frozenset({Role.EVIDENCE_VIEWER}))
    )
    with TestClient(create_app(settings, access_token_validator=validator)) as client:
        response = client.get(
            "/documents",
            headers={"x-aerodocs-gateway-access-token": "a.b.c"},
        )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_required"
    assert validator.tokens == []


def test_denial_does_not_lookup_resource_or_log_token(
    default_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = default_settings.model_copy(update={"security_mode": "test"})
    validator = StaticValidator()
    looked_up = False

    def unexpected_lookup(*_args: object, **_kwargs: object) -> object:
        nonlocal looked_up
        looked_up = True
        raise AssertionError("resource lookup occurred before authentication")

    monkeypatch.setattr(
        "app.ingestion.service.DocumentService.metadata",
        unexpected_lookup,
    )
    token = "sensitive.header.value"
    with (
        caplog.at_level(logging.DEBUG),
        TestClient(create_app(settings, access_token_validator=validator)) as client,
    ):
        response = client.get(
            "/documents/doc_" + "0" * 64,
            headers={"authorization": f"Bearer {token}"},
        )
    assert response.status_code == 401
    assert looked_up is False
    assert token not in caplog.text


def test_route_inventory_fails_on_unclassified_route(
    default_settings: Settings,
) -> None:
    application = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    @application.get("/unexpected")
    async def unexpected() -> dict[str, bool]:
        return {"unexpected": True}

    with pytest.raises(RouteInventoryError, match="unclassified"):
        audit_phase_8a_routes(application.routes, ui_enabled=False)


def test_route_inventory_records_direct_bearer_as_only_protected_source(
    default_settings: Settings,
) -> None:
    settings = default_settings.model_copy(update={"security_mode": "test"})
    validator = StaticValidator(
        Principal(subject="viewer", roles=frozenset({Role.EVIDENCE_VIEWER}))
    )
    application = create_app(settings, access_token_validator=validator)
    policies = audit_phase_8a_routes(
        application.routes,
        ui_enabled=True,
    )
    for policy in policies:
        expected = (
            frozenset()
            if policy.path_template == "/health"
            else frozenset({CredentialSource.DIRECT_BEARER})
        )
        assert policy.credential_sources == expected


def test_route_inventory_rejects_api_route_replaced_by_plain_route(
    default_settings: Settings,
) -> None:
    routes = phase_8a_flat_routes(default_settings)
    index, original = route_at(routes, "/documents", {"GET"})
    assert type(original) is APIRoute
    replacement = Route(
        "/documents",
        endpoint=original.endpoint,
        methods=["GET"],
        name=original.name,
    )
    replacement.methods = {"GET"}
    routes[index] = replacement
    with pytest.raises(RouteInventoryError, match="route kind mismatch"):
        audit_phase_8a_routes(routes, ui_enabled=True)


def test_route_inventory_rejects_expected_plain_route_replaced_by_api_route(
    default_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = phase_8a_flat_routes(default_settings)
    key = ("GET", "/documents")
    monkeypatch.setitem(
        security_routes._POLICY,
        key,
        (RouteKind.STARLETTE_ROUTE, Permission.VIEW_EVIDENCE),
    )
    with pytest.raises(RouteInventoryError, match="route kind mismatch"):
        audit_phase_8a_routes(routes, ui_enabled=True)


def test_route_inventory_rejects_mount_replaced_by_route(
    default_settings: Settings,
) -> None:
    routes = phase_8a_flat_routes(default_settings)
    index, original = route_at(routes, "/ui/assets", None)
    assert type(original) is Mount

    async def endpoint(_request: object) -> object:
        raise AssertionError

    replacement = Route(
        "/ui/assets/{path:path}",
        endpoint=endpoint,
        methods=["GET"],
        name="ui-assets",
    )
    replacement.methods = {"GET", "HEAD"}
    routes[index] = replacement
    with pytest.raises(RouteInventoryError, match="route kind mismatch"):
        audit_phase_8a_routes(routes, ui_enabled=True)


def test_route_inventory_rejects_unexpected_static_mount_application(
    default_settings: Settings,
) -> None:
    routes = phase_8a_flat_routes(default_settings)
    index, original = route_at(routes, "/ui/assets", None)
    assert type(original) is Mount

    async def unexpected_app(_scope: object, _receive: object, _send: object) -> None:
        raise AssertionError

    routes[index] = Mount(
        "/ui/assets",
        app=unexpected_app,
        name="ui-assets",
    )
    with pytest.raises(RouteInventoryError, match="mounted application type"):
        audit_phase_8a_routes(routes, ui_enabled=True)


def test_route_inventory_rejects_unsupported_route_subclass(
    default_settings: Settings,
) -> None:
    class UnsupportedApiRoute(APIRoute):
        pass

    routes = phase_8a_flat_routes(default_settings)
    index, original = route_at(routes, "/documents", {"GET"})
    api_route = cast(APIRoute, original)
    routes[index] = UnsupportedApiRoute(
        "/documents",
        endpoint=api_route.endpoint,
        methods=["GET"],
        name=api_route.name,
    )
    with pytest.raises(RouteInventoryError, match="unsupported route object"):
        audit_phase_8a_routes(routes, ui_enabled=True)


def test_route_inventory_records_actual_route_kinds(
    default_settings: Settings,
) -> None:
    policies = audit_phase_8a_routes(
        phase_8a_flat_routes(default_settings),
        ui_enabled=True,
    )
    assert {
        policy.route_kind
        for policy in policies
        if policy.path_template != "/ui/assets/{path:path}"
    } == {RouteKind.FASTAPI_API_ROUTE}
    assert {
        policy.route_kind
        for policy in policies
        if policy.path_template == "/ui/assets/{path:path}"
    } == {RouteKind.MOUNT}


@pytest.mark.parametrize("ui_enabled", [True, False])
def test_valid_test_mode_route_inventories_construct(
    default_settings: Settings,
    ui_enabled: bool,
) -> None:
    settings = default_settings.model_copy(
        update={"security_mode": "test", "ui_enabled": ui_enabled}
    )
    validator = StaticValidator(
        Principal(subject="viewer", roles=frozenset({Role.EVIDENCE_VIEWER}))
    )
    application = create_app(settings, access_token_validator=validator)
    assert application


def test_valid_ui_unavailable_route_inventory_constructs(
    default_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_settings: Settings) -> Never:
        raise UIConfigurationError

    monkeypatch.setattr("app.factory.create_ui_router", unavailable)
    settings = default_settings.model_copy(update={"security_mode": "test"})
    validator = StaticValidator(
        Principal(subject="viewer", roles=frozenset({Role.EVIDENCE_VIEWER}))
    )
    application = create_app(settings, access_token_validator=validator)
    assert application


def phase_8a_flat_routes(default_settings: Settings) -> list[BaseRoute]:
    settings = default_settings.model_copy(update={"security_mode": "test"})
    validator = StaticValidator(
        Principal(subject="viewer", roles=frozenset({Role.EVIDENCE_VIEWER}))
    )
    return _flatten(create_app(settings, access_token_validator=validator).routes)


def route_at(
    routes: list[BaseRoute],
    path: str,
    methods: set[str] | None,
) -> tuple[int, BaseRoute]:
    for index, route in enumerate(routes):
        if (
            getattr(route, "path", None) == path
            and getattr(route, "methods", None) == methods
        ):
            return index, route
    raise AssertionError("expected fixture route was not found")


def test_secure_construction_is_network_free_and_lifespan_failure_keeps_health(
    default_settings: Settings,
    rsa_material: tuple[RSAPrivateKey, dict[str, object]],
) -> None:
    private, key = rsa_material
    transport = FakeTransport([IdentityValidationUnavailableError()])
    application = create_phase_8a_secure_app(
        oidc_settings(default_settings),
        jwks_transport=transport,
    )
    assert transport.calls == 0
    assert all(getattr(route, "path", None) != "/docs" for route in application.routes)
    token = access_token(private, kid=cast(str, key["kid"]))

    with TestClient(application) as client:
        assert transport.calls == 1
        assert client.get("/health").status_code == 200
        protected = client.get(
            "/documents",
            headers={"authorization": f"Bearer {token}"},
        )

    assert protected.status_code == 503
    assert protected.json()["error"]["code"] == "identity_validation_unavailable"
    assert transport.closed is True


@pytest.mark.parametrize(
    "value",
    [
        "application/json",
        "APPLICATION/JWK-SET+JSON",
        "application/json; charset=UTF-8",
        'application/jwk-set+json; charset="utf-8"; version=1',
    ],
)
def test_jwks_media_types_accept_only_approved_json(value: str) -> None:
    validate_jwks_content_type((value,))


@pytest.mark.parametrize(
    "values",
    [
        (),
        ("text/json",),
        ("text/plain",),
        ("text/html",),
        ("application/problem+json",),
        ("application/json; charset=latin-1",),
        ("application/json; charset=utf-8; charset=utf-8",),
        ('application/json; charset="utf-8',),
        ("application/json, text/plain",),
        ("application/json", "application/json"),
    ],
)
def test_jwks_media_types_reject_missing_ambiguous_or_malformed(
    values: tuple[str, ...],
) -> None:
    with pytest.raises(IdentityValidationUnavailableError):
        validate_jwks_content_type(values)


@pytest.mark.asyncio
async def test_httpx_transport_has_final_timeout_and_redirect_policy() -> None:
    transport = HttpxJwksTransport()
    try:
        assert transport._client.timeout.connect == 2.0
        assert transport._client.timeout.read == 2.0
        assert transport._client.timeout.write == 2.0
        assert transport._client.timeout.pool == 1.0
        assert transport._client.follow_redirects is False
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_unknown_kids_obey_global_cooldown_and_do_not_amplify_network(
    rsa_material: tuple[RSAPrivateKey, dict[str, object]],
) -> None:
    _private, key = rsa_material
    response = jwks_response([key])
    transport = FakeTransport([response])
    clock = FakeClock(0.0)
    resolver = JwksResolver(
        "https://identity.example.test/jwks", transport, clock=clock
    )
    await resolver.preflight()

    for number in range(500):
        with pytest.raises(InvalidCredentialsError):
            await resolver.key_for(f"attacker-{number}")

    assert transport.calls == 1
    assert len(resolver._negative) <= MAXIMUM_NEGATIVE_KEYS


@pytest.mark.asyncio
async def test_negative_cache_capacity_and_oldest_eviction(
    rsa_material: tuple[RSAPrivateKey, dict[str, object]],
) -> None:
    _private, key = rsa_material
    response = jwks_response([key])
    transport = FakeTransport([response])
    clock = FakeClock(0.0)
    resolver = JwksResolver(
        "https://identity.example.test/jwks", transport, clock=clock
    )
    await resolver.preflight()
    for number in range(MAXIMUM_NEGATIVE_KEYS + 1):
        resolver._insert_negative(f"unknown-{number:03d}", clock())

    assert len(resolver._negative) == MAXIMUM_NEGATIVE_KEYS
    assert "unknown-000" not in resolver._negative
    assert "unknown-256" in resolver._negative

    clock.advance(30.0)
    resolver._purge_negative(clock())
    assert resolver._negative == {}


@pytest.mark.asyncio
async def test_failed_refresh_starts_cooldown(
    rsa_material: tuple[RSAPrivateKey, dict[str, object]],
) -> None:
    _private, key = rsa_material
    transport = FakeTransport(
        [IdentityValidationUnavailableError(), jwks_response([key])]
    )
    clock = FakeClock(0.0)
    resolver = JwksResolver(
        "https://identity.example.test/jwks", transport, clock=clock
    )
    await resolver.preflight()
    with pytest.raises(IdentityValidationUnavailableError):
        await resolver.key_for(cast(str, key["kid"]))
    assert transport.calls == 1
    clock.advance(30.0)
    assert await resolver.key_for(cast(str, key["kid"]))
    assert transport.calls == 2


@pytest.mark.asyncio
async def test_expired_cache_and_failed_refresh_fail_closed(
    rsa_material: tuple[RSAPrivateKey, dict[str, object]],
) -> None:
    _private, key = rsa_material
    first = jwks_response([key], cache_control="max-age=1")
    transport = FakeTransport([first, IdentityValidationUnavailableError()])
    clock = FakeClock(0.0)
    resolver = JwksResolver(
        "https://identity.example.test/jwks", transport, clock=clock
    )
    await resolver.preflight()
    clock.advance(30.0)
    with pytest.raises(IdentityValidationUnavailableError):
        await resolver.key_for(cast(str, key["kid"]))
    assert transport.calls == 2


@pytest.mark.asyncio
async def test_redirect_and_oversized_jwks_fail_closed() -> None:
    redirect = JwksHttpResponse(
        status_code=302,
        content_type_values=("application/json",),
        cache_control_values=(),
        body=b"{}",
    )
    oversized = JwksHttpResponse(
        status_code=200,
        content_type_values=("application/json",),
        cache_control_values=(),
        body=b"x" * (MAXIMUM_JWKS_BYTES + 1),
    )
    clock = FakeClock(0.0)
    transport = FakeTransport([redirect, oversized])
    resolver = JwksResolver(
        "https://identity.example.test/jwks", transport, clock=clock
    )
    await resolver.preflight()
    clock.advance(30.0)
    with pytest.raises(IdentityValidationUnavailableError):
        await resolver.key_for("key-1")
    assert transport.calls == 2


@pytest.mark.asyncio
async def test_concurrent_requests_share_one_refresh(
    rsa_material: tuple[RSAPrivateKey, dict[str, object]],
) -> None:
    _private, key = rsa_material

    class BlockingTransport(FakeTransport):
        def __init__(self) -> None:
            super().__init__([jwks_response([key])])
            self.started = __import__("asyncio").Event()
            self.release = __import__("asyncio").Event()

        async def fetch(self, url: str) -> JwksHttpResponse:
            self.started.set()
            await self.release.wait()
            return await super().fetch(url)

    import asyncio

    transport = BlockingTransport()
    resolver = JwksResolver("https://identity.example.test/jwks", transport)
    calls = [
        asyncio.create_task(resolver.key_for(cast(str, key["kid"]))) for _ in range(20)
    ]
    await transport.started.wait()
    transport.release.set()
    results = await asyncio.gather(*calls)
    assert len(results) == 20
    assert transport.calls == 1
    assert resolver._refresh_waiters == 0


@pytest.mark.asyncio
async def test_legitimate_rotation_after_cooldown(
    rsa_material: tuple[RSAPrivateKey, dict[str, object]],
) -> None:
    _private, first = rsa_material
    _second_private, second = make_rsa_material("rotated")
    transport = FakeTransport([jwks_response([first]), jwks_response([first, second])])
    clock = FakeClock(0.0)
    resolver = JwksResolver(
        "https://identity.example.test/jwks", transport, clock=clock
    )
    await resolver.preflight()
    with pytest.raises(InvalidCredentialsError):
        await resolver.key_for("rotated")
    clock.advance(30.0)
    assert await resolver.key_for("rotated")
    assert transport.calls == 2


def test_duplicate_protected_header_and_payload_members_are_rejected() -> None:
    duplicate_header = compact_token(
        b'{"alg":"RS256","alg":"RS256","kid":"key-1"}',
        b'{"iss":"issuer"}',
    )
    duplicate_payload = compact_token(
        b'{"alg":"RS256","kid":"key-1"}',
        b'{"sub":"one","sub":"two"}',
    )
    with pytest.raises(InvalidCredentialsError):
        preparse_access_token(duplicate_header)
    with pytest.raises(InvalidCredentialsError):
        preparse_access_token(duplicate_payload)


def test_bounded_json_accepts_exact_nesting_depth() -> None:
    nested = (
        '{"value":'
        + "[" * (MAXIMUM_JSON_NESTING_DEPTH - 1)
        + "0"
        + "]" * (MAXIMUM_JSON_NESTING_DEPTH - 1)
        + "}"
    ).encode()
    assert duplicate_safe_json(nested)


@pytest.mark.parametrize(
    "nested",
    [
        lambda: (
            "[" * (MAXIMUM_JSON_NESTING_DEPTH + 1)
            + "0"
            + "]" * (MAXIMUM_JSON_NESTING_DEPTH + 1)
        ),
        lambda: (
            '{"value":' * (MAXIMUM_JSON_NESTING_DEPTH + 1)
            + "0"
            + "}" * (MAXIMUM_JSON_NESTING_DEPTH + 1)
        ),
        lambda: (
            ('[{"value":' * (MAXIMUM_JSON_NESTING_DEPTH // 2 + 1))
            + "0"
            + ("}]" * (MAXIMUM_JSON_NESTING_DEPTH // 2 + 1))
        ),
    ],
)
def test_bounded_json_rejects_deep_array_object_and_mixed_nesting(
    nested: object,
) -> None:
    assert callable(nested)
    with pytest.raises(InvalidCredentialsError):
        duplicate_safe_json(cast(Callable[[], str], nested)().encode())


def test_bounded_json_ignores_structure_and_escapes_inside_strings() -> None:
    parsed = duplicate_safe_json(
        b'{"text":"[{]} \\"quoted\\" \\\\ slash","items":["}","["]}'
    )
    assert isinstance(parsed, dict)
    assert parsed["items"] == ["}", "["]


def test_bounded_json_object_member_boundary() -> None:
    exact = (
        "{"
        + ",".join(f'"k{number}":0' for number in range(MAXIMUM_JSON_OBJECT_MEMBERS))
        + "}"
    )
    assert (
        len(cast(dict[str, object], duplicate_safe_json(exact.encode())))
        == MAXIMUM_JSON_OBJECT_MEMBERS
    )
    excessive = exact[:-1] + ',"extra":0}'
    with pytest.raises(InvalidCredentialsError):
        duplicate_safe_json(excessive.encode())


def test_bounded_json_array_element_boundary() -> None:
    exact = (
        '{"items":[' + ",".join("0" for _ in range(MAXIMUM_JSON_ARRAY_ELEMENTS)) + "]}"
    )
    assert (
        len(cast(dict[str, list[object]], duplicate_safe_json(exact.encode()))["items"])
        == MAXIMUM_JSON_ARRAY_ELEMENTS
    )
    excessive = exact.replace("]}", ",0]}")
    with pytest.raises(InvalidCredentialsError):
        duplicate_safe_json(excessive.encode())


def test_bounded_json_container_count_boundary() -> None:
    child_containers = MAXIMUM_JSON_CONTAINERS - 2
    exact = '{"items":[' + ",".join("{}" for _ in range(child_containers)) + "]}"
    assert duplicate_safe_json(exact.encode())
    excessive = exact.replace("]}", ",{}]}")
    with pytest.raises(InvalidCredentialsError):
        duplicate_safe_json(excessive.encode())


@pytest.mark.parametrize(
    "malformed",
    [
        b'{"value":' + b"[" * 15 + b"0" + b"]" * 14 + b"}",
        b"{"
        + b",".join(
            f'"k{number}":0'.encode() for number in range(MAXIMUM_JSON_OBJECT_MEMBERS)
        ),
        b'{"items":['
        + b",".join(b"0" for _ in range(MAXIMUM_JSON_ARRAY_ELEMENTS))
        + b",]}",
        b'{"items":['
        + b",".join(b"{}" for _ in range(MAXIMUM_JSON_CONTAINERS - 2))
        + b"]",
    ],
)
def test_bounded_json_malformed_near_each_boundary_is_typed(
    malformed: bytes,
) -> None:
    with pytest.raises(InvalidCredentialsError):
        duplicate_safe_json(malformed)


def test_header_and_payload_use_the_same_bounded_parser() -> None:
    valid_header = b'{"alg":"RS256","kid":"key-1"}'
    valid_payload = b'{"sub":"subject-1"}'
    too_deep = (b"[" * 17) + b"0" + (b"]" * 17)
    for header, payload in ((too_deep, valid_payload), (valid_header, too_deep)):
        with pytest.raises(InvalidCredentialsError):
            preparse_access_token(compact_token(header, payload))


def test_deep_json_request_returns_fixed_401_without_parser_disclosure(
    default_settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = default_settings.model_copy(update={"security_mode": "test"})
    payload = (b"[" * 1_100) + b"0" + (b"]" * 1_100)
    token = compact_token(b'{"alg":"RS256","kid":"key-1"}', payload)
    with (
        caplog.at_level(logging.DEBUG),
        TestClient(
            create_app(settings, access_token_validator=ParsingValidator())
        ) as client,
    ):
        response = client.get(
            "/documents",
            headers={"authorization": f"Bearer {token}"},
        )
    assert response.status_code == 401
    assert response.json()["error"] == {
        "code": "invalid_credentials",
        "message": "The supplied credentials are invalid.",
    }
    combined = response.text + caplog.text
    assert "RecursionError" not in combined
    assert "recursion" not in combined.lower()
    assert "json" not in combined.lower()
    assert token not in combined


def test_valid_viewer_and_operator_tokens_and_known_bad_signature_do_not_refresh(
    default_settings: Settings,
    rsa_material: tuple[RSAPrivateKey, dict[str, object]],
) -> None:
    private, key = rsa_material
    impostor, _impostor_key = make_rsa_material("key-1")
    transport = FakeTransport([jwks_response([key])])
    settings = oidc_settings(default_settings)
    application = create_phase_8a_secure_app(settings, jwks_transport=transport)
    viewer_token = access_token(private, kid="key-1")
    operator_token = access_token(
        private,
        kid="key-1",
        groups=[VALID_OPERATOR_GROUP],
    )
    invalid_signature = access_token(impostor, kid="key-1")

    with TestClient(application) as client:
        viewer = client.get(
            "/documents",
            headers={"authorization": f"Bearer {viewer_token}"},
        )
        operator = client.get(
            "/documents",
            headers={"authorization": f"Bearer {operator_token}"},
        )
        denied = client.get(
            "/documents",
            headers={"authorization": f"Bearer {invalid_signature}"},
        )
        mutation = client.post(
            "/indexes/rebuild",
            headers={"authorization": f"Bearer {operator_token}"},
        )

    assert viewer.status_code == 200
    assert operator.status_code == 200
    assert denied.status_code == 401
    assert denied.json()["error"]["code"] == "invalid_credentials"
    assert mutation.status_code not in {401, 403}
    assert transport.calls == 1


@pytest.mark.parametrize(
    ("claim_updates", "header_updates"),
    [
        ({"iss": "https://wrong.example.test/"}, {}),
        ({"aud": "wrong-api"}, {}),
        ({"exp": 1}, {}),
        ({"iat": 2_000, "nbf": 2_000, "exp": 3_000}, {}),
        ({"groups": ["duplicate", "duplicate"]}, {}),
        ({"groups": VALID_VIEWER_GROUP}, {}),
        ({"sub": "\x00unsafe"}, {}),
        ({}, {"typ": "ID"}),
    ],
)
def test_wrong_issuer_audience_lifetime_and_claim_shapes_are_rejected(
    default_settings: Settings,
    rsa_material: tuple[RSAPrivateKey, dict[str, object]],
    claim_updates: dict[str, object],
    header_updates: dict[str, object],
) -> None:
    private, key = rsa_material
    now = int(time.time())
    claims: dict[str, object] = {
        "iss": "https://identity.example.test/",
        "aud": VALID_OIDC_AUDIENCE,
        "sub": "subject-1",
        "iat": now,
        "nbf": now,
        "exp": now + 900,
        "groups": [VALID_VIEWER_GROUP],
    }
    claims.update(claim_updates)
    headers: dict[str, object] = {"kid": key["kid"], "typ": "at+jwt"}
    headers.update(header_updates)
    token = jwt.encode(
        claims,
        private,
        algorithm="RS256",
        headers=headers,
    )
    transport = FakeTransport([jwks_response([key])])
    application = create_phase_8a_secure_app(
        oidc_settings(default_settings),
        jwks_transport=transport,
    )
    with TestClient(application) as client:
        response = client.get(
            "/documents",
            headers={"authorization": f"Bearer {token}"},
        )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_credentials"


def test_non_rs256_algorithm_is_rejected_before_key_lookup(
    default_settings: Settings,
    rsa_material: tuple[RSAPrivateKey, dict[str, object]],
) -> None:
    _private, key = rsa_material
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": "https://identity.example.test/",
            "aud": VALID_OIDC_AUDIENCE,
            "sub": "subject-1",
            "iat": now,
            "nbf": now,
            "exp": now + 900,
            "groups": [VALID_VIEWER_GROUP],
        },
        "not-a-production-secret-at-least-32-bytes",
        algorithm="HS256",
        headers={"kid": "key-1", "typ": "at+jwt"},
    )
    transport = FakeTransport([jwks_response([key])])
    application = create_phase_8a_secure_app(
        oidc_settings(default_settings),
        jwks_transport=transport,
    )
    with TestClient(application) as client:
        response = client.get(
            "/documents",
            headers={"authorization": f"Bearer {token}"},
        )
    assert response.status_code == 401
    assert transport.calls == 1


def test_future_issued_token_outside_skew_is_rejected(
    default_settings: Settings,
    rsa_material: tuple[RSAPrivateKey, dict[str, object]],
) -> None:
    private, key = rsa_material
    now = int(time.time())
    token = access_token(private, kid="key-1", now=now + 61)
    transport = FakeTransport([jwks_response([key])])
    application = create_phase_8a_secure_app(
        oidc_settings(default_settings),
        jwks_transport=transport,
        wall_clock=lambda: float(now),
    )
    with TestClient(application) as client:
        response = client.get(
            "/documents",
            headers={"authorization": f"Bearer {token}"},
        )
    assert response.status_code == 401


def test_duplicate_jwks_members_are_rejected(
    rsa_material: tuple[RSAPrivateKey, dict[str, object]],
) -> None:
    _private, key = rsa_material
    duplicated = (
        '{"keys":[],"keys":' + json.dumps([key], separators=(",", ":")) + "}"
    ).encode()
    with pytest.raises(IdentityValidationUnavailableError):
        parse_jwks(duplicated)


@pytest.mark.parametrize(
    "update",
    [
        {"kty": "EC"},
        {"use": "enc"},
        {"key_ops": ["sign"]},
        {"alg": "RS512"},
        {"d": "private"},
        {"e": "Aw"},
    ],
)
def test_jwks_rejects_incompatible_or_private_keys(
    rsa_material: tuple[RSAPrivateKey, dict[str, object]],
    update: dict[str, object],
) -> None:
    _private, key = rsa_material
    invalid = {**key, **update}
    with pytest.raises(IdentityValidationUnavailableError):
        parse_jwks(json.dumps({"keys": [invalid]}).encode())


@pytest.fixture
def rsa_material() -> tuple[RSAPrivateKey, dict[str, object]]:
    return make_rsa_material("key-1")


def make_rsa_material(kid: str) -> tuple[RSAPrivateKey, dict[str, object]]:
    private = rsa.generate_private_key(public_exponent=65_537, key_size=2_048)
    numbers = private.public_key().public_numbers()
    return private, {
        "kty": "RSA",
        "kid": kid,
        "use": "sig",
        "key_ops": ["verify"],
        "alg": "RS256",
        "n": integer_segment(numbers.n),
        "e": integer_segment(numbers.e),
    }


def integer_segment(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def compact_token(header: bytes, payload: bytes) -> str:
    encoded = [
        base64.urlsafe_b64encode(part).rstrip(b"=").decode("ascii")
        for part in (header, payload, b"signature")
    ]
    return ".".join(encoded)


def jwks_response(
    keys: list[dict[str, object]],
    *,
    cache_control: str = "max-age=300",
) -> JwksHttpResponse:
    return JwksHttpResponse(
        status_code=200,
        content_type_values=("application/jwk-set+json; charset=utf-8",),
        cache_control_values=(cache_control,),
        body=json.dumps({"keys": keys}, separators=(",", ":")).encode(),
    )


def access_token(
    private: RSAPrivateKey,
    *,
    kid: str,
    issuer: str = "https://identity.example.test/",
    audience: str = VALID_OIDC_AUDIENCE,
    subject: str = "subject-1",
    groups: list[str] | None = None,
    now: int | None = None,
) -> str:
    issued = int(time.time()) if now is None else now
    return jwt.encode(
        {
            "iss": issuer,
            "aud": audience,
            "sub": subject,
            "iat": issued,
            "nbf": issued,
            "exp": issued + 900,
            "groups": groups or [VALID_VIEWER_GROUP],
        },
        private,
        algorithm="RS256",
        headers={"kid": kid, "typ": "at+jwt"},
    )
