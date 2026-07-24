"""Typed local retrieval-evaluation records."""

from pydantic import BaseModel, Field, model_validator

from app.retrieval.models import SearchResult


class RelevantPageReference(BaseModel):
    document_id: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_interval(self) -> "RelevantPageReference":
        if self.page_end < self.page_start:
            raise ValueError("page_end must be at least page_start")
        return self


class EvaluationCase(BaseModel):
    case_id: str
    query: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevant_references: list[RelevantPageReference] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def require_relevance(self) -> "EvaluationCase":
        if not self.relevant_chunk_ids and not self.relevant_references:
            raise ValueError("at least one relevance target is required")
        return self


class CaseEvaluation(BaseModel):
    case_id: str
    reciprocal_rank: float
    recall_at_1: float = Field(ge=0.0, le=1.0)
    recall_at_3: float = Field(ge=0.0, le=1.0)
    recall_at_5: float = Field(ge=0.0, le=1.0)
    latency_seconds: float
    failure_category: str | None
    results: list[SearchResult]


class EvaluationReport(BaseModel):
    case_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mean_reciprocal_rank: float
    no_result_rate: float
    average_query_latency_seconds: float
    cases: list[CaseEvaluation]
