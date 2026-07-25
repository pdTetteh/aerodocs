"""Safe structured models exposed by the AeroDocs MCP adapter."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.retrieval.models import ModelIdentity


class McpStatus(BaseModel):
    """Metadata-only local MCP status."""

    name: str
    version: str
    status: str
    transport: str
    safety_notice: str


class ActiveIndexSummary(BaseModel):
    """Bounded active-index metadata without storage details."""

    index_profile_id: str
    corpus_fingerprint: str
    embedding_model: ModelIdentity
    vector_count: int = Field(ge=1)
    source_document_count: int = Field(ge=1)
    built_at: datetime
