"""Human-reviewed work-order drafting endpoints."""

from typing import Annotated

from fastapi import APIRouter, Query, Request, status
from starlette.concurrency import run_in_threadpool

from app.api.errors import ErrorResponse
from app.security.models import Principal
from app.work_orders.models import (
    DEFAULT_WORK_ORDER_LIST_LIMIT,
    MAXIMUM_WORK_ORDER_LIST_LIMIT,
    DefectReport,
    ReviewRequest,
    WorkOrderDraft,
    WorkOrderListResponse,
)
from app.work_orders.service import WorkOrderService

WORK_ORDER_ERRORS: dict[int | str, dict[str, object]] = {
    403: {"model": ErrorResponse, "description": "Separate reviewer required"},
    404: {"model": ErrorResponse, "description": "Work-order draft not found"},
    409: {"model": ErrorResponse, "description": "Draft already reviewed"},
    413: {"model": ErrorResponse, "description": "Resource limit exceeded"},
    422: {"model": ErrorResponse, "description": "Invalid request"},
    500: {"model": ErrorResponse, "description": "Integrity validation failed"},
    503: {"model": ErrorResponse, "description": "Storage or model unavailable"},
}


def _verified_subject(request: Request) -> str | None:
    principal = getattr(request.state, "aerodocs_principal", None)
    return principal.subject if isinstance(principal, Principal) else None


def create_work_orders_router(service: WorkOrderService) -> APIRouter:
    """Create the work-order router for one application instance."""
    router = APIRouter(tags=["work-orders"])

    @router.post(
        "/work-orders",
        response_model=WorkOrderDraft,
        status_code=status.HTTP_201_CREATED,
        responses=WORK_ORDER_ERRORS,
    )
    async def create_draft(
        defect: DefectReport,
        request: Request,
    ) -> WorkOrderDraft:
        return await run_in_threadpool(
            service.draft,
            defect,
            principal_subject=_verified_subject(request),
        )

    @router.get(
        "/work-orders",
        response_model=WorkOrderListResponse,
        responses=WORK_ORDER_ERRORS,
    )
    async def list_drafts(
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[
            int, Query(ge=1, le=MAXIMUM_WORK_ORDER_LIST_LIMIT)
        ] = DEFAULT_WORK_ORDER_LIST_LIMIT,
    ) -> WorkOrderListResponse:
        return await run_in_threadpool(service.list, offset=offset, limit=limit)

    @router.get(
        "/work-orders/{work_order_id}",
        response_model=WorkOrderDraft,
        responses=WORK_ORDER_ERRORS,
    )
    async def get_draft(work_order_id: str) -> WorkOrderDraft:
        return await run_in_threadpool(service.get, work_order_id)

    @router.post(
        "/work-orders/{work_order_id}/review",
        response_model=WorkOrderDraft,
        responses=WORK_ORDER_ERRORS,
    )
    async def review_draft(
        work_order_id: str,
        review: ReviewRequest,
        request: Request,
    ) -> WorkOrderDraft:
        return await run_in_threadpool(
            service.review,
            work_order_id,
            review,
            principal_subject=_verified_subject(request),
        )

    return router
