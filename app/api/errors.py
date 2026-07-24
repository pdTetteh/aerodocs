"""Typed HTTP representation of document-domain errors."""

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.ingestion.errors import DocumentError
from app.retrieval.errors import RetrievalError


class ErrorDetail(BaseModel):
    """Stable public error details."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Typed API error envelope."""

    error: ErrorDetail
    details: dict[str, object] | None = None


async def document_error_handler(
    _request: Request,
    exception: Exception,
) -> JSONResponse:
    """Map expected document failures without leaking internal details."""
    if not isinstance(exception, DocumentError):
        raise exception
    response = ErrorResponse(
        error=ErrorDetail(
            code=exception.code,
            message=exception.public_message,
        ),
        details=getattr(exception, "public_details", None),
    )
    return JSONResponse(
        status_code=exception.status_code,
        content=response.model_dump(mode="json", exclude_none=True),
    )


async def retrieval_error_handler(
    _request: Request,
    exception: Exception,
) -> JSONResponse:
    """Map expected retrieval failures without leaking internal details."""
    if not isinstance(exception, RetrievalError):
        raise exception
    response = ErrorResponse(
        error=ErrorDetail(code=exception.code, message=exception.public_message),
        details=getattr(exception, "public_details", None),
    )
    return JSONResponse(
        status_code=exception.status_code,
        content=response.model_dump(mode="json", exclude_none=True),
    )
