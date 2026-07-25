"""Grounded answer-generation endpoint."""

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app.api.errors import ErrorResponse
from app.generation.models import AnswerRequest, AnswerResponse
from app.generation.service import AnswerService

GENERATION_ERRORS: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorResponse, "description": "Index or document not found"},
    409: {"model": ErrorResponse, "description": "Index is stale or incompatible"},
    413: {"model": ErrorResponse, "description": "Answer resource limit exceeded"},
    422: {"model": ErrorResponse, "description": "Invalid answer request"},
    500: {"model": ErrorResponse, "description": "Integrity validation failed"},
    502: {"model": ErrorResponse, "description": "Generation output invalid"},
    503: {"model": ErrorResponse, "description": "Model or audit unavailable"},
    504: {"model": ErrorResponse, "description": "Generation timed out"},
}


def create_generation_router(service: AnswerService) -> APIRouter:
    """Create the grounded-answer router for one application instance."""
    router = APIRouter(tags=["generation"])

    @router.post(
        "/answers",
        response_model=AnswerResponse,
        responses=GENERATION_ERRORS,
    )
    async def answer(request: AnswerRequest) -> AnswerResponse:
        return await run_in_threadpool(service.answer, request)

    return router
