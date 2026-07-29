"""Optional read-only JSON projections used only by the browser UI."""

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app.api.retrieval import RETRIEVAL_ERRORS
from app.retrieval.models import ActiveIndexStatus
from app.retrieval.service import RetrievalService


def create_ui_api_router(service: RetrievalService) -> APIRouter:
    """Bind compact UI API routes to application-local production services."""
    router = APIRouter(prefix="/ui/api", include_in_schema=False)

    @router.get(
        "/index-status",
        response_model=ActiveIndexStatus,
        responses=RETRIEVAL_ERRORS,
    )
    async def active_index_status() -> ActiveIndexStatus:
        return await run_in_threadpool(service.active_index_status)

    return router
