"""Per-application AeroDocs service composition."""

from dataclasses import dataclass

from app.core.config import Settings
from app.generation.provider import LlmProvider, TransformersLlmProvider
from app.generation.service import AnswerService
from app.ingestion.service import DocumentService
from app.retrieval.embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from app.retrieval.service import RetrievalService


@dataclass(frozen=True)
class AeroDocsServices:
    """Application-local services shared by protocol adapters."""

    settings: Settings
    documents: DocumentService
    retrieval: RetrievalService
    answers: AnswerService


def create_services(
    settings: Settings,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    llm_provider: LlmProvider | None = None,
) -> AeroDocsServices:
    """Create an isolated AeroDocs service graph with lazy providers."""
    documents = DocumentService(settings)
    embeddings = embedding_provider or SentenceTransformerEmbeddingProvider(settings)
    retrieval = RetrievalService(settings, embeddings)
    generator = llm_provider or TransformersLlmProvider(settings)
    answers = AnswerService(settings, retrieval, generator)
    return AeroDocsServices(
        settings=settings,
        documents=documents,
        retrieval=retrieval,
        answers=answers,
    )
