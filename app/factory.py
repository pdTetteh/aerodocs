"""FastAPI application factory."""

import logging

from fastapi import FastAPI

from app.api.health import create_health_router
from app.core.config import Settings
from app.core.logging import configure_logging

LOGGER = logging.getLogger("aerodocs")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure an AeroDocs FastAPI application."""
    resolved_settings = settings or Settings()
    configure_logging(resolved_settings.log_level)

    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
    )
    application.include_router(create_health_router(resolved_settings))

    LOGGER.info("AeroDocs application created")
    return application
