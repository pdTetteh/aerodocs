"""Local index management and evidence-search endpoints."""

from fastapi import APIRouter, Response
from starlette.concurrency import run_in_threadpool

from app.api.errors import ErrorResponse
from app.retrieval.models import (
    IndexBuildResponse,
    IndexMetadata,
    SearchRequest,
    SearchResponse,
)
from app.retrieval.service import RetrievalService

RETRIEVAL_ERRORS: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorResponse, "description": "Index or document not found"},
    409: {"model": ErrorResponse, "description": "Index conflict or stale index"},
    413: {"model": ErrorResponse, "description": "Index resource limit exceeded"},
    422: {"model": ErrorResponse, "description": "Invalid retrieval request"},
    500: {"model": ErrorResponse, "description": "Index integrity failure"},
    503: {"model": ErrorResponse, "description": "Embedding model unavailable"},
}


def create_retrieval_router(service: RetrievalService) -> APIRouter:
    router = APIRouter(tags=["retrieval"])

    @router.post(
        "/indexes/rebuild",
        response_model=IndexBuildResponse,
        status_code=201,
        responses={**RETRIEVAL_ERRORS, 200: {"model": IndexBuildResponse}},
    )
    async def rebuild_index(response: Response) -> IndexBuildResponse:
        result = await run_in_threadpool(service.rebuild)
        response.status_code = 201 if result.rebuilt else 200
        return result

    @router.get(
        "/indexes/active",
        response_model=IndexMetadata,
        responses=RETRIEVAL_ERRORS,
    )
    async def active_index() -> IndexMetadata:
        return await run_in_threadpool(service.active_metadata_without_models)

    @router.post(
        "/search",
        response_model=SearchResponse,
        responses=RETRIEVAL_ERRORS,
    )
    async def search(request: SearchRequest) -> SearchResponse:
        return await run_in_threadpool(service.search, request)

    return router
