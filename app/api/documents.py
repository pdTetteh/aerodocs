"""Document upload and traceability endpoints."""

from typing import Annotated

from fastapi import APIRouter, File, Query, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.errors import ErrorResponse
from app.ingestion.models import (
    ChunkListResponse,
    DocumentIngestResponse,
    DocumentMetadata,
    PageListResponse,
)
from app.ingestion.service import DocumentService

UPLOAD_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ErrorResponse, "description": "Invalid PDF upload"},
    409: {"model": ErrorResponse, "description": "Processing profile conflict"},
    413: {"model": ErrorResponse, "description": "PDF is too large"},
    415: {"model": ErrorResponse, "description": "Unsupported media type"},
    500: {"model": ErrorResponse, "description": "Document processing failure"},
}
READ_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    404: {"model": ErrorResponse, "description": "Document not found"},
    500: {"model": ErrorResponse, "description": "Document storage failure"},
}


def create_documents_router(service: DocumentService) -> APIRouter:
    """Create document endpoints bound to one application-local service."""
    router = APIRouter(prefix="/documents", tags=["documents"])

    @router.post(
        "",
        response_model=DocumentIngestResponse,
        status_code=201,
        responses={
            **UPLOAD_ERROR_RESPONSES,
            200: {
                "model": DocumentIngestResponse,
                "description": "Existing duplicate document",
            },
        },
    )
    async def upload_document(
        response: Response,
        file: Annotated[UploadFile, File(description="Public or authorized PDF")],
    ) -> DocumentIngestResponse:
        result = await run_in_threadpool(
            service.ingest,
            file.file,
            file.filename,
            file.content_type,
        )
        response.status_code = 200 if result.duplicate else 201
        return result

    @router.get(
        "/{document_id}",
        response_model=DocumentMetadata,
        responses=READ_ERROR_RESPONSES,
    )
    async def document_metadata(document_id: str) -> DocumentMetadata:
        return await run_in_threadpool(service.metadata, document_id)

    @router.get(
        "/{document_id}/pages",
        response_model=PageListResponse,
        responses=READ_ERROR_RESPONSES,
    )
    async def document_pages(
        document_id: str,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> PageListResponse:
        return await run_in_threadpool(
            service.pages,
            document_id,
            offset=offset,
            limit=limit,
        )

    @router.get(
        "/{document_id}/chunks",
        response_model=ChunkListResponse,
        responses=READ_ERROR_RESPONSES,
    )
    async def document_chunks(
        document_id: str,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> ChunkListResponse:
        return await run_in_threadpool(
            service.chunks,
            document_id,
            offset=offset,
            limit=limit,
        )

    return router
