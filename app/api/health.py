"""Application health endpoint."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import Settings


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    name: str
    version: str
    status: Literal["healthy"]


def create_health_router(settings: Settings) -> APIRouter:
    """Create the health router using the supplied application settings."""
    router = APIRouter(tags=["system"])

    @router.get("/health", response_model=HealthResponse)
    async def get_health() -> HealthResponse:
        return HealthResponse(
            name=settings.app_name,
            version=settings.app_version,
            status="healthy",
        )

    return router
