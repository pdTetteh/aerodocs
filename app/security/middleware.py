"""Direct-bearer authentication and route authorisation middleware."""

import json
from uuid import uuid4

from starlette.types import ASGIApp, Receive, Scope, Send

from app.security.bearer import direct_bearer_token
from app.security.models import (
    AccessTokenValidator,
    IdentityValidationUnavailableError,
    InvalidCredentialsError,
    Principal,
)
from app.security.routes import Permission, RoutePolicy


class DirectBearerSecurityMiddleware:
    """Enforce the exact Phase 8A direct-bearer route policy."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        validator: AccessTokenValidator,
        policies: list[RoutePolicy],
    ) -> None:
        self.app = app
        self.validator = validator
        self.policies = tuple(policies)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        policy = self._policy(scope)
        if policy is None:
            await _error(
                send,
                404,
                "not_found",
                "The requested resource was not found.",
            )
            return
        if policy.permission is Permission.PUBLIC:
            await self.app(scope, receive, send)
            return
        try:
            token = direct_bearer_token(scope)
            if token is None:
                await _error(
                    send,
                    401,
                    "authentication_required",
                    "Authentication is required.",
                    authenticate=True,
                )
                return
            principal = await self.validator.validate(token)
        except InvalidCredentialsError:
            await _error(
                send,
                401,
                "invalid_credentials",
                "The supplied credentials are invalid.",
                authenticate=True,
            )
            return
        except IdentityValidationUnavailableError:
            await _error(
                send,
                503,
                "identity_validation_unavailable",
                "Identity validation is temporarily unavailable.",
            )
            return
        if not _permitted(principal, policy.permission):
            await _error(
                send,
                403,
                "permission_denied",
                "Permission is denied.",
            )
            return
        scope.setdefault("state", {})["aerodocs_principal"] = principal
        await self.app(scope, receive, send)

    def _policy(self, scope: Scope) -> RoutePolicy | None:
        method = str(scope.get("method", "")).upper()
        path = str(scope.get("path", ""))
        for policy in self.policies:
            if policy.method == method and policy.path_pattern.fullmatch(path):
                return policy
        return None


def _permitted(principal: Principal, permission: Permission) -> bool:
    if permission is Permission.VIEW_EVIDENCE:
        return principal.can_view_evidence()
    if permission is Permission.OPERATE_DOCUMENTS:
        return principal.can_operate_documents()
    return permission is Permission.PUBLIC


async def _error(
    send: Send,
    status: int,
    code: str,
    message: str,
    *,
    authenticate: bool = False,
) -> None:
    payload = json.dumps(
        {
            "error": {"code": code, "message": message},
            "correlation_id": uuid4().hex,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode("ascii")),
        (b"cache-control", b"no-store"),
    ]
    if authenticate:
        headers.append((b"www-authenticate", b"Bearer"))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": payload})
