"""FastAPI application factory."""

import logging

from fastapi import FastAPI

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
from app.generation.provider import LlmProvider, TransformersLlmProvider
from app.generation.service import AnswerService
from app.ingestion.errors import DocumentError
from app.ingestion.service import DocumentService
from app.retrieval.embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from app.retrieval.errors import RetrievalError
from app.retrieval.service import RetrievalService

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
    application.include_router(create_health_router(resolved_settings))
    document_service = DocumentService(resolved_settings)
    application.include_router(create_documents_router(document_service))
    provider = embedding_provider or SentenceTransformerEmbeddingProvider(
        resolved_settings
    )
    retrieval_service = RetrievalService(resolved_settings, provider)
    application.include_router(create_retrieval_router(retrieval_service))
    generator = llm_provider or TransformersLlmProvider(resolved_settings)
    answer_service = AnswerService(
        resolved_settings,
        retrieval_service,
        generator,
    )
    application.include_router(create_generation_router(answer_service))
    application.add_exception_handler(DocumentError, document_error_handler)
    application.add_exception_handler(RetrievalError, retrieval_error_handler)
    application.add_exception_handler(GenerationError, generation_error_handler)

    LOGGER.info("AeroDocs application created")
    return application
