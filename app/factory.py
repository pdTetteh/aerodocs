"""FastAPI application factory."""

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.documents import create_documents_router
from app.api.errors import (
    document_error_handler,
    generation_error_handler,
    retrieval_error_handler,
)
from app.api.generation import create_generation_router
from app.api.health import create_health_router
from app.api.retrieval import create_retrieval_router
from app.core.config import Settings
from app.core.logging import configure_logging
from app.core.request_limits import RequestBodyLimitMiddleware
from app.generation.errors import GenerationError
from app.generation.provider import LlmProvider
from app.ingestion.errors import DocumentError
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.errors import RetrievalError
from app.security.jwks import HttpxJwksTransport, JwksResolver, JwksTransport
from app.security.jwt import OidcAccessTokenValidator
from app.security.middleware import DirectBearerSecurityMiddleware
from app.security.models import AccessTokenValidator
from app.security.routes import audit_phase_8a_routes
from app.services.runtime import create_services
from app.ui.api import create_ui_api_router
from app.ui.errors import UIConfigurationError
from app.ui.router import create_ui_router, create_ui_unavailable_router
from app.ui.security import UiSecurityHeadersMiddleware

LOGGER = logging.getLogger("aerodocs")


def create_app(
    settings: Settings | None = None,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    llm_provider: LlmProvider | None = None,
    access_token_validator: AccessTokenValidator | None = None,
    jwks_transport: JwksTransport | None = None,
    wall_clock: Callable[[], float] = time.time,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """Create and configure an AeroDocs FastAPI application."""
    return _create_app(
        settings,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        access_token_validator=access_token_validator,
        jwks_transport=jwks_transport,
        wall_clock=wall_clock,
        monotonic_clock=monotonic_clock,
        documentation_enabled=True,
    )


def create_phase_8a_secure_app(
    settings: Settings,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    llm_provider: LlmProvider | None = None,
    jwks_transport: JwksTransport | None = None,
    wall_clock: Callable[[], float] = time.time,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """Create the OIDC direct-bearer Phase 8A review application.

    This construction path is intentionally incomplete for pilot deployment.
    Browser-session authentication is introduced only in Phase 8B.
    """
    if settings.security_mode != "oidc":
        raise ValueError("the Phase 8A secure application requires oidc mode")
    return _create_app(
        settings,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        access_token_validator=None,
        jwks_transport=jwks_transport,
        wall_clock=wall_clock,
        monotonic_clock=monotonic_clock,
        documentation_enabled=False,
    )


def _create_app(
    settings: Settings | None,
    *,
    embedding_provider: EmbeddingProvider | None,
    llm_provider: LlmProvider | None,
    access_token_validator: AccessTokenValidator | None,
    jwks_transport: JwksTransport | None,
    wall_clock: Callable[[], float],
    monotonic_clock: Callable[[], float],
    documentation_enabled: bool,
) -> FastAPI:
    resolved_settings = settings or Settings()
    resolved_settings.validate_http_security()
    configure_logging(resolved_settings.log_level)
    resolved_documentation_enabled = (
        documentation_enabled and resolved_settings.security_mode == "disabled"
    )

    resolver: JwksResolver | None = None
    validator = access_token_validator
    if resolved_settings.security_mode == "test":
        if validator is None:
            raise ValueError("test security mode requires an injected validator")
        if jwks_transport is not None:
            raise ValueError("test security mode cannot configure a JWKS transport")
    elif resolved_settings.security_mode == "oidc":
        if validator is not None:
            raise ValueError("oidc mode does not accept an injected validator")
        if resolved_settings.oidc_jwks_url is None:
            raise ValueError("oidc mode requires a JWKS URL")
        resolver = JwksResolver(
            resolved_settings.oidc_jwks_url,
            jwks_transport or HttpxJwksTransport(),
            clock=monotonic_clock,
        )
        validator = OidcAccessTokenValidator(
            resolved_settings,
            resolver,
            clock=wall_clock,
        )
    elif validator is not None or jwks_transport is not None:
        raise ValueError("disabled mode does not accept identity dependencies")

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        if resolver is not None:
            await resolver.preflight()
        try:
            yield
        finally:
            if resolver is not None:
                await resolver.close()

    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        openapi_url="/openapi.json" if resolved_documentation_enabled else None,
        docs_url="/docs" if resolved_documentation_enabled else None,
        redoc_url="/redoc" if resolved_documentation_enabled else None,
        lifespan=lifespan,
    )
    application.add_middleware(
        RequestBodyLimitMiddleware,
        maximum_bytes=resolved_settings.max_request_body_bytes,
    )
    if resolved_settings.ui_enabled:
        application.add_middleware(UiSecurityHeadersMiddleware)
    application.include_router(create_health_router(resolved_settings))
    services = create_services(
        resolved_settings,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
    )
    application.include_router(create_documents_router(services.documents))
    application.include_router(create_retrieval_router(services.retrieval))
    application.include_router(create_generation_router(services.answers))
    if resolved_settings.ui_enabled:
        try:
            ui_router = create_ui_router(resolved_settings)
        except UIConfigurationError:
            LOGGER.warning("AeroDocs Evidence Review UI configuration is unavailable")
            application.include_router(create_ui_unavailable_router())
        else:
            static_root = Path(__file__).parent / "ui" / "static"
            application.include_router(create_ui_api_router(services.retrieval))
            application.mount(
                "/ui/assets",
                StaticFiles(directory=static_root),
                name="ui-assets",
            )
            application.include_router(ui_router)
    application.add_exception_handler(DocumentError, document_error_handler)
    application.add_exception_handler(RetrievalError, retrieval_error_handler)
    application.add_exception_handler(GenerationError, generation_error_handler)

    if resolved_settings.security_mode != "disabled":
        if validator is None:
            raise ValueError("enabled security requires a validator")
        policies = audit_phase_8a_routes(
            application.routes,
            ui_enabled=resolved_settings.ui_enabled,
        )
        application.add_middleware(
            DirectBearerSecurityMiddleware,
            validator=validator,
            policies=policies,
        )

    LOGGER.info("AeroDocs application created")
    return application
