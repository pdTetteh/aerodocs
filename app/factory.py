"""FastAPI application factory."""

import logging
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
) -> FastAPI:
    """Create and configure an AeroDocs FastAPI application."""
    resolved_settings = settings or Settings()
    configure_logging(resolved_settings.log_level)

    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
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

    LOGGER.info("AeroDocs application created")
    return application
