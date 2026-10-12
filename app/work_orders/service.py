"""Work-order drafting and human review.

The service researches a defect through the grounded answer pipeline and stores
the result as a draft. It never approves maintenance: the only terminal states
are recorded human decisions about the *draft*, and each requires a person.
"""

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from app.core.config import Settings
from app.generation.errors import GenerationError, InvalidAnswerRequestError
from app.generation.models import AnswerRequest, AnswerResponse
from app.retrieval.errors import RetrievalError
from app.work_orders.errors import (
    AcknowledgementRequiredError,
    InvalidDefectReportError,
    ReviewerIdentityError,
    SeparateReviewerRequiredError,
    WorkOrderAlreadyReviewedError,
    WorkOrderError,
)
from app.work_orders.models import (
    DEFAULT_WORK_ORDER_LIST_LIMIT,
    MAXIMUM_WORK_ORDER_LIST_LIMIT,
    SAFETY_NOTICES,
    WORK_ORDER_FORMAT_VERSION,
    DefectReport,
    Finding,
    ReviewDecision,
    ReviewRequest,
    WorkOrderDraft,
    WorkOrderListResponse,
    WorkOrderSummary,
    compute_content_sha256,
)
from app.work_orders.planner import PlannedQuestion, plan_questions
from app.work_orders.store import WorkOrderStore, new_work_order_id


class AnswerPort(Protocol):
    def answer(self, request: AnswerRequest) -> AnswerResponse: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


class WorkOrderService:
    """Create cited drafts and record explicit human review decisions."""

    def __init__(
        self,
        settings: Settings,
        answers: AnswerPort,
        *,
        store: WorkOrderStore | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.settings = settings
        self.answers = answers
        self.store = store or WorkOrderStore(settings.work_order_root)
        self.clock = clock
        # Review is read-modify-write on one file; AeroDocs runs one worker.
        self._lock = threading.Lock()

    def draft(
        self,
        defect: DefectReport,
        *,
        principal_subject: str | None = None,
    ) -> WorkOrderDraft:
        """Research a defect and store a cited draft awaiting human review."""
        findings: list[Finding] = []
        first_error: WorkOrderError | GenerationError | RetrievalError | None = None
        index_profile_id: str | None = None
        corpus_fingerprint: str | None = None
        for planned in plan_questions(defect):
            try:
                response = self.answers.answer(
                    AnswerRequest(
                        question=planned.question,
                        document_ids=defect.document_ids,
                    )
                )
            except InvalidAnswerRequestError as error:
                raise InvalidDefectReportError from error
            except (GenerationError, RetrievalError) as error:
                first_error = first_error or error
                findings.append(_failed_finding(planned, error))
                continue
            index_profile_id = index_profile_id or response.index_profile_id
            corpus_fingerprint = corpus_fingerprint or response.corpus_fingerprint
            findings.append(_finding_from_response(planned, response))
        if all(finding.outcome == "failed" for finding in findings):
            assert first_error is not None
            raise first_error

        draft = WorkOrderDraft(
            work_order_format_version=WORK_ORDER_FORMAT_VERSION,
            work_order_id=new_work_order_id(),
            status="draft",
            created_at=self.clock(),
            created_by=principal_subject,
            created_by_verified=principal_subject is not None,
            defect=defect,
            findings=findings,
            index_profile_id=index_profile_id,
            corpus_fingerprint=corpus_fingerprint,
            notices=list(SAFETY_NOTICES),
            content_sha256="",
        )
        draft.content_sha256 = compute_content_sha256(draft)
        self.store.create(draft)
        return draft

    def review(
        self,
        work_order_id: str,
        request: ReviewRequest,
        *,
        principal_subject: str | None = None,
    ) -> WorkOrderDraft:
        """Record one human decision on a draft. Decisions are final."""
        with self._lock:
            draft = self.store.read(work_order_id)
            if draft.status != "draft":
                raise WorkOrderAlreadyReviewedError
            if request.decision == "confirm" and (
                not request.acknowledge_human_responsibility
            ):
                raise AcknowledgementRequiredError

            if principal_subject is not None:
                if request.reviewer is not None:
                    raise ReviewerIdentityError
                reviewer = principal_subject
                verified = True
            else:
                if request.reviewer is None:
                    raise ReviewerIdentityError
                reviewer = request.reviewer
                verified = False

            if (
                request.decision == "confirm"
                and self.settings.work_order_require_separate_reviewer
                and verified
                and draft.created_by_verified
                and draft.created_by == reviewer
            ):
                raise SeparateReviewerRequiredError

            reviewed = draft.model_copy(
                update={
                    "status": (
                        "human_confirmed"
                        if request.decision == "confirm"
                        else "human_rejected"
                    ),
                    "review": ReviewDecision(
                        decision=request.decision,
                        reviewer=reviewer,
                        reviewer_identity_verified=verified,
                        note=request.note,
                        reviewed_at=self.clock(),
                    ),
                }
            )
            self.store.replace(reviewed)
            return reviewed

    def get(self, work_order_id: str) -> WorkOrderDraft:
        return self.store.read(work_order_id)

    def list(
        self,
        *,
        offset: int = 0,
        limit: int = DEFAULT_WORK_ORDER_LIST_LIMIT,
    ) -> WorkOrderListResponse:
        limit = max(1, min(limit, MAXIMUM_WORK_ORDER_LIST_LIMIT))
        offset = max(0, offset)
        drafts = self.store.list_all()
        return WorkOrderListResponse(
            offset=offset,
            limit=limit,
            total=len(drafts),
            items=[
                WorkOrderSummary(
                    work_order_id=item.work_order_id,
                    status=item.status,
                    created_at=item.created_at,
                    title=item.defect.title,
                    finding_count=len(item.findings),
                    supported_finding_count=sum(
                        finding.outcome == "supported" for finding in item.findings
                    ),
                )
                for item in drafts[offset : offset + limit]
            ],
        )


def _failed_finding(
    planned: PlannedQuestion,
    error: GenerationError | RetrievalError,
) -> Finding:
    return Finding(
        finding_id=planned.finding_id,
        purpose=planned.purpose,
        question=planned.question,
        outcome="failed",
        failure_code=error.code,
        explanation=error.public_message,
    )


def _finding_from_response(
    planned: PlannedQuestion,
    response: AnswerResponse,
) -> Finding:
    if response.abstained:
        return Finding(
            finding_id=planned.finding_id,
            purpose=planned.purpose,
            question=planned.question,
            outcome="abstained",
            abstention_reason=response.abstention_reason,
            explanation=response.explanation,
            audit_id=response.audit_id,
        )
    return Finding(
        finding_id=planned.finding_id,
        purpose=planned.purpose,
        question=planned.question,
        outcome="supported",
        claims=response.claims,
        citations=response.citations,
        audit_id=response.audit_id,
    )
