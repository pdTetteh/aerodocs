"""Exact Phase 8A route inventory and direct-bearer policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from fastapi.routing import APIRoute, _IncludedRouter
from fastapi.staticfiles import StaticFiles
from starlette.routing import BaseRoute, Mount, Route

from app.security.models import CredentialSource


class RouteKind(StrEnum):
    """Closed route kinds accepted by the security audit."""

    FASTAPI_API_ROUTE = "fastapi_api_route"
    STARLETTE_ROUTE = "starlette_route"
    MOUNT = "mount"


class Permission(StrEnum):
    """Closed Phase 8A permissions."""

    PUBLIC = "public"
    VIEW_EVIDENCE = "view_evidence"
    OPERATE_DOCUMENTS = "operate_documents"


@dataclass(frozen=True)
class RoutePolicy:
    """One exact audited method/path policy."""

    method: str
    path_template: str
    route_kind: RouteKind
    permission: Permission
    credential_sources: frozenset[CredentialSource]
    path_pattern: re.Pattern[str]


class RouteInventoryError(ValueError):
    """Signal that actual application routes differ from the closed policy."""


_POLICY: dict[tuple[str, str], tuple[RouteKind, Permission]] = {
    ("GET", "/health"): (RouteKind.FASTAPI_API_ROUTE, Permission.PUBLIC),
    ("GET", "/documents"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.VIEW_EVIDENCE,
    ),
    ("POST", "/documents"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.OPERATE_DOCUMENTS,
    ),
    ("GET", "/documents/{document_id}"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.VIEW_EVIDENCE,
    ),
    ("GET", "/documents/{document_id}/pages"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.VIEW_EVIDENCE,
    ),
    ("GET", "/documents/{document_id}/chunks"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.VIEW_EVIDENCE,
    ),
    ("GET", "/indexes/active"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.VIEW_EVIDENCE,
    ),
    ("POST", "/indexes/rebuild"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.OPERATE_DOCUMENTS,
    ),
    ("POST", "/search"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.VIEW_EVIDENCE,
    ),
    ("POST", "/answers"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.VIEW_EVIDENCE,
    ),
    ("POST", "/work-orders"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.OPERATE_DOCUMENTS,
    ),
    ("GET", "/work-orders"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.VIEW_EVIDENCE,
    ),
    ("GET", "/work-orders/{work_order_id}"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.VIEW_EVIDENCE,
    ),
    ("POST", "/work-orders/{work_order_id}/review"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.OPERATE_DOCUMENTS,
    ),
    ("GET", "/ui/api/index-status"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.VIEW_EVIDENCE,
    ),
    ("GET", "/ui/"): (RouteKind.FASTAPI_API_ROUTE, Permission.VIEW_EVIDENCE),
    ("GET", "/ui/documents"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.VIEW_EVIDENCE,
    ),
    ("GET", "/ui/search"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.VIEW_EVIDENCE,
    ),
    ("GET", "/ui/ask"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.VIEW_EVIDENCE,
    ),
    ("GET", "/ui/{unknown_path:path}"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.VIEW_EVIDENCE,
    ),
    ("GET", "/ui/{path:path}"): (
        RouteKind.FASTAPI_API_ROUTE,
        Permission.VIEW_EVIDENCE,
    ),
    ("GET", "/ui/assets/{path:path}"): (
        RouteKind.MOUNT,
        Permission.VIEW_EVIDENCE,
    ),
    ("HEAD", "/ui/assets/{path:path}"): (
        RouteKind.MOUNT,
        Permission.VIEW_EVIDENCE,
    ),
}


def audit_phase_8a_routes(
    routes: list[BaseRoute],
    *,
    ui_enabled: bool,
) -> list[RoutePolicy]:
    """Audit actual route objects and return the executable closed policy."""
    discovered: list[RoutePolicy] = []
    actual_keys: set[tuple[str, str]] = set()
    for route in _flatten(routes):
        actual_kind = _actual_route_kind(route)
        if actual_kind is RouteKind.MOUNT:
            assert type(route) is Mount
            if route.path != "/ui/assets" or route.name != "ui-assets":
                raise RouteInventoryError("unexpected mounted application")
            if type(route.app) is not StaticFiles:
                raise RouteInventoryError("unexpected mounted application type")
            pattern = re.compile(r"^/ui/assets(?:/.*)?$")
            for method in ("GET", "HEAD"):
                key = (method, "/ui/assets/{path:path}")
                _add_policy(discovered, actual_keys, key, pattern, actual_kind)
            continue
        assert type(route) in {APIRoute, Route}
        concrete_route = cast(APIRoute | Route, route)
        methods = concrete_route.methods or set()
        for method in sorted(methods):
            key = (method, concrete_route.path)
            _add_policy(
                discovered,
                actual_keys,
                key,
                concrete_route.path_regex,
                actual_kind,
            )

    expected = {key for key in _POLICY if ui_enabled or not key[1].startswith("/ui")}
    # UI configuration fallback replaces all normal UI entries with one catch-all.
    normal_ui = {
        key
        for key in expected
        if key[1].startswith("/ui") and key[1] != "/ui/{path:path}"
    }
    fallback = {("GET", "/ui/{path:path}")}
    accepted = (
        actual_keys == expected - fallback or actual_keys == (expected - normal_ui)
        if ui_enabled
        else actual_keys == expected
    )
    if not accepted:
        missing = sorted(expected - actual_keys)
        unexpected = sorted(actual_keys - expected)
        raise RouteInventoryError(f"route inventory mismatch: {missing=} {unexpected=}")
    return discovered


def _add_policy(
    discovered: list[RoutePolicy],
    actual_keys: set[tuple[str, str]],
    key: tuple[str, str],
    pattern: re.Pattern[str],
    actual_kind: RouteKind,
) -> None:
    if key in actual_keys or key not in _POLICY:
        raise RouteInventoryError(f"unclassified or duplicate route: {key}")
    actual_keys.add(key)
    expected_kind, permission = _POLICY[key]
    if actual_kind is not expected_kind:
        raise RouteInventoryError(f"route kind mismatch: {key}")
    credential_sources = (
        frozenset()
        if permission is Permission.PUBLIC
        else frozenset({CredentialSource.DIRECT_BEARER})
    )
    discovered.append(
        RoutePolicy(
            method=key[0],
            path_template=key[1],
            route_kind=actual_kind,
            permission=permission,
            credential_sources=credential_sources,
            path_pattern=pattern,
        )
    )


def _actual_route_kind(route: BaseRoute) -> RouteKind:
    if type(route) is APIRoute:
        return RouteKind.FASTAPI_API_ROUTE
    if type(route) is Mount:
        return RouteKind.MOUNT
    if type(route) is Route:
        return RouteKind.STARLETTE_ROUTE
    raise RouteInventoryError("unsupported route object")


def _flatten(routes: list[BaseRoute]) -> list[BaseRoute]:
    flattened: list[BaseRoute] = []
    for route in routes:
        if type(route) is _IncludedRouter:
            included = route
            original = included.original_router
            context = included.include_context
            if original is None or context is None or context.prefix:
                raise RouteInventoryError("unsupported included router")
            flattened.extend(_flatten(original.routes))
        else:
            flattened.append(route)
    return flattened
