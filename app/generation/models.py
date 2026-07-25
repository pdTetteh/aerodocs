"""Typed grounded-generation models."""

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.generation.limits import (
    MAX_DRAFT_CLAIM_CHARACTERS,
    MAX_DRAFT_CLAIMS,
    MAX_DRAFT_CONFLICT_EVIDENCE_IDS,
    MAX_DRAFT_QUOTE_CHARACTERS,
    MAX_DRAFT_SUPPORTS_PER_CLAIM,
)


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    top_k: int | None = None
    document_ids: list[str] | None = None


class ProviderIdentity(BaseModel):
    name: str
    revision: str
    device: str


@dataclass(frozen=True)
class PreparedGenerationInput:
    """Exact tokenized prompt and immutable generation limits."""

    input_ids: object
    attention_mask: object
    prompt_token_count: int
    model_context_limit: int
    maximum_new_tokens: int
    provider: ProviderIdentity
    response_prefix: str
    deadline_monotonic: float


class EvidencePassage(BaseModel):
    evidence_id: str
    rank: int
    score: float
    document_id: str
    filename: str
    title: str
    chunk_id: str
    chunk_order: int
    page_start: int
    page_end: int
    section_heading: str | None
    text: str
    character_count: int
    truncated: bool = False


class Citation(BaseModel):
    """Authoritative provenance shared by claim-support links."""

    number: int
    evidence_id: str
    score: float
    document_id: str
    title: str
    filename: str
    page_start: int
    page_end: int
    section_heading: str | None
    chunk_id: str


class ClaimSupport(BaseModel):
    """Claim-specific link to one citation and its validated quotation."""

    claim_id: str
    evidence_id: str
    citation_number: int
    quote: str | None


class AnswerClaim(BaseModel):
    claim_id: str
    text: str
    supports: list[ClaimSupport]


class GroundingOutcome(BaseModel):
    claim_count: int = 0
    supported_claim_count: int = 0
    citation_count: int = 0
    fabricated_evidence_id_count: int = 0
    invalid_quote_count: int = 0
    unsupported_claim_count: int = 0
    semantic_mismatch_count: int = 0


AbstentionReason = Literal[
    "insufficient_evidence",
    "prompt_injection_risk",
    "conflicting_evidence",
    "grounding_validation_failed",
    "safety_boundary",
]


class AnswerResponse(BaseModel):
    question: str
    abstained: bool
    abstention_reason: AbstentionReason | None
    explanation: str | None
    answer: str | None
    claims: list[AnswerClaim]
    citations: list[Citation]
    evidence: list[EvidencePassage]
    index_profile_id: str
    corpus_fingerprint: str
    requested_top_k: int
    generation_format_version: str
    provider: ProviderIdentity
    context_character_count: int
    prompt_token_count: int
    output_token_count: int
    latency_seconds: float
    audit_id: str
    grounding: GroundingOutcome


class DraftSupport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1, max_length=64)
    quote: str | None = Field(max_length=MAX_DRAFT_QUOTE_CHARACTERS)


class DraftClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_DRAFT_CLAIM_CHARACTERS)
    supports: list[DraftSupport] = Field(
        min_length=1,
        max_length=MAX_DRAFT_SUPPORTS_PER_CLAIM,
    )


class DraftAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answer"]
    claims: list[DraftClaim] = Field(min_length=1, max_length=MAX_DRAFT_CLAIMS)


class DraftAbstention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["abstain"]
    reason_code: Literal["insufficient_evidence"]


class DraftConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["conflict"]
    conflicting_evidence_ids: list[str] = Field(
        min_length=2,
        max_length=MAX_DRAFT_CONFLICT_EVIDENCE_IDS,
    )


GenerationDraft = Annotated[
    DraftAnswer | DraftAbstention | DraftConflict,
    Field(discriminator="status"),
]

AuditStatus = Literal[
    "success",
    "policy_abstention",
    "evidence_abstention",
    "conflict_abstention",
    "injection_abstention",
    "retrieval_failure",
    "stale_index_failure",
    "model_load_failure",
    "generation_failure",
    "generation_timeout",
    "prompt_limit_failure",
    "output_limit_failure",
    "invalid_output_failure",
    "grounding_failure",
    "audit_persistence_failure",
    "invalid_request_failure",
]


class GenerationAudit(BaseModel):
    audit_format_version: str
    audit_id: str
    created_at: datetime
    question_sha256: str
    requested_document_ids: list[str] | None
    requested_top_k: int | None
    index_profile_id: str | None
    corpus_fingerprint: str | None
    selected_chunk_ids: list[str]
    selected_scores: list[float]
    provider: ProviderIdentity
    status: AuditStatus
    reason_code: str | None
    claim_count: int
    evidence_ids: list[str]
    grounding: GroundingOutcome
    latency_seconds: float
    prompt_token_count: int
    output_token_count: int
    timeout_stage: Literal["lock_wait", "preparation", "generation"] | None = None
