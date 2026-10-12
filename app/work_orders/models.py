"""Typed work-order drafting models."""

import hashlib
import json
import unicodedata
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.generation.models import AbstentionReason, AnswerClaim, Citation

WORK_ORDER_FORMAT_VERSION = "1"
MAX_DEFECT_TITLE_CHARACTERS = 200
MAX_DEFECT_DESCRIPTION_CHARACTERS = 2_000
MAX_DEFECT_COMPONENT_CHARACTERS = 200
MAX_DEFECT_DOCUMENT_IDS = 20
MAX_REVIEW_NOTE_CHARACTERS = 1_000
MAX_REVIEWER_CHARACTERS = 100
DEFAULT_WORK_ORDER_LIST_LIMIT = 20
MAXIMUM_WORK_ORDER_LIST_LIMIT = 100

WorkOrderStatus = Literal["draft", "human_confirmed", "human_rejected"]
FindingPurpose = Literal[
    "inspection",
    "limits",
    "repair_or_replacement",
    "required_approvals",
]
FindingOutcome = Literal["supported", "abstained", "failed"]
ReviewDecisionKind = Literal["confirm", "reject"]

SAFETY_NOTICES: tuple[str, ...] = (
    "This draft is an evidence-navigation aid. It does not approve maintenance, "
    "determine airworthiness, or authorize return to service.",
    "Every finding requires human verification against the controlling "
    "manufacturer documentation and applicable regulations.",
    "Confirming a draft records that a human reviewed it. It is not maintenance "
    "approval and does not release anything to service.",
)


def normalize_text(value: str) -> str:
    """Normalize untrusted text to bounded single-spaced printable text."""
    normalized = unicodedata.normalize("NFKC", value)
    printable = "".join(
        " " if character.isspace() else character
        for character in normalized
        if character.isspace() or not unicodedata.category(character).startswith("C")
    )
    return " ".join(printable.split())


class DefectReport(BaseModel):
    """A synthetic or public-document defect scenario to research."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(max_length=MAX_DEFECT_TITLE_CHARACTERS)
    description: str = Field(max_length=MAX_DEFECT_DESCRIPTION_CHARACTERS)
    component: str | None = Field(
        default=None, max_length=MAX_DEFECT_COMPONENT_CHARACTERS
    )
    document_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_DEFECT_DOCUMENT_IDS,
    )

    @field_validator("title", "description", mode="after")
    @classmethod
    def _require_text(cls, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("component", mode="after")
    @classmethod
    def _normalize_component(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_text(value) or None


class Finding(BaseModel):
    """One researched question and its grounded outcome."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str
    purpose: FindingPurpose
    question: str
    outcome: FindingOutcome
    abstention_reason: AbstentionReason | None = None
    failure_code: str | None = None
    explanation: str | None = None
    claims: list[AnswerClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    audit_id: str | None = None
    requires_human_verification: Literal[True] = True


class ReviewRequest(BaseModel):
    """A human decision on a draft."""

    model_config = ConfigDict(extra="forbid")

    decision: ReviewDecisionKind
    note: str = Field(max_length=MAX_REVIEW_NOTE_CHARACTERS)
    acknowledge_human_responsibility: bool = False
    reviewer: str | None = Field(default=None, max_length=MAX_REVIEWER_CHARACTERS)

    @field_validator("note", mode="after")
    @classmethod
    def _require_note(cls, value: str) -> str:
        normalized = normalize_text(value)
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("reviewer", mode="after")
    @classmethod
    def _normalize_reviewer(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_text(value) or None


class ReviewDecision(BaseModel):
    """The recorded human decision. It can never represent maintenance approval."""

    model_config = ConfigDict(extra="forbid")

    decision: ReviewDecisionKind
    reviewer: str
    reviewer_identity_verified: bool
    note: str
    reviewed_at: datetime
    approves_maintenance: Literal[False] = False


class WorkOrderDraft(BaseModel):
    """A cited, human-reviewable work-order draft."""

    model_config = ConfigDict(extra="forbid")

    work_order_format_version: str
    work_order_id: str
    status: WorkOrderStatus
    created_at: datetime
    created_by: str | None
    created_by_verified: bool
    defect: DefectReport
    findings: list[Finding]
    index_profile_id: str | None
    corpus_fingerprint: str | None
    notices: list[str]
    review: ReviewDecision | None = None
    content_sha256: str


class WorkOrderSummary(BaseModel):
    work_order_id: str
    status: WorkOrderStatus
    created_at: datetime
    title: str
    finding_count: int
    supported_finding_count: int


class WorkOrderListResponse(BaseModel):
    offset: int
    limit: int
    total: int
    items: list[WorkOrderSummary]


def compute_content_sha256(draft: WorkOrderDraft) -> str:
    """Hash the immutable draft content; status and review may change later."""
    content = draft.model_dump(
        mode="json",
        exclude={"status", "review", "content_sha256"},
    )
    encoded = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
