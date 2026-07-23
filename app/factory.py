"""FastAPI application factory."""

import logging

from fastapi import FastAPI

from app.api.documents import create_documents_router
from app.api.errors import document_error_handler
from app.api.health import create_health_router
from app.core.config import Settings
from app.core.logging import configure_logging
from app.core.request_limits import RequestBodyLimitMiddleware
from app.ingestion.errors import DocumentError
from app.ingestion.service import DocumentService

LOGGER = logging.getLogger("aerodocs")


def create_app(settings: Settings | None = None) -> FastAPI:
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
    application.include_router(create_health_router(resolved_settings))
    document_service = DocumentService(resolved_settings)
    application.include_router(create_documents_router(document_service))
    application.add_exception_handler(DocumentError, document_error_handler)

    LOGGER.info("AeroDocs application created")
    return application
