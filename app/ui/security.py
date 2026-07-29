"""Path-scoped browser security headers for the read-only UI."""

from starlette.types import ASGIApp, Message, Receive, Scope, Send

UI_CONTENT_SECURITY_POLICY = (
    "default-src 'none'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "connect-src 'self'; "
    "img-src 'self'; "
    "font-src 'self'; "
    "form-action 'self'; "
    "base-uri 'none'; "
    "object-src 'none'; "
    "frame-ancestors 'none'"
)


class UiSecurityHeadersMiddleware:
    """Add restrictive headers only to the isolated ``/ui`` namespace."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if (
            scope["type"] != "http"
            or not path.startswith("/ui")
            or path.startswith("/ui/api/")
        ):
            await self.app(scope, receive, send)
            return

        async def secured_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (
                            b"content-security-policy",
                            UI_CONTENT_SECURITY_POLICY.encode(),
                        ),
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"no-referrer"),
                        (
                            b"permissions-policy",
                            b"camera=(), microphone=(), geolocation=()",
                        ),
                        (
                            b"cache-control",
                            (
                                b"public, max-age=3600, must-revalidate"
                                if path.startswith("/ui/assets/")
                                else b"no-store"
                            ),
                        ),
                    ]
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, secured_send)
