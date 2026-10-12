"""Offline tests for human-reviewed work-order drafting."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings
from app.factory import create_app
from app.generation.errors import (
    GenerationFailedError,
    InvalidAnswerRequestError,
    LlmModelUnavailableError,
)
from app.generation.models import (
    AnswerClaim,
    AnswerRequest,
    AnswerResponse,
    Citation,
    ClaimSupport,
    GroundingOutcome,
    ProviderIdentity,
)
from app.security.models import InvalidCredentialsError, Principal, Role
from app.work_orders.errors import (
    AcknowledgementRequiredError,
    InvalidDefectReportError,
    ReviewerIdentityError,
    SeparateReviewerRequiredError,
    WorkOrderAlreadyReviewedError,
    WorkOrderIntegrityError,
    WorkOrderNotFoundError,
    WorkOrderPersistenceError,
)
from app.work_orders.models import DefectReport, ReviewRequest, normalize_text
from app.work_orders.planner import plan_questions
from app.work_orders.service import WorkOrderService
from app.work_orders.store import WorkOrderStore


def _response(question: str, *, abstain: bool = False) -> AnswerResponse:
    citation = Citation(
        number=1,
        evidence_id="ev_1",
        score=0.9,
        document_id="doc_1",
        title="Synthetic Manual",
        filename="manual.pdf",
        page_start=3,
        page_end=3,
        section_heading="Inspection",
        chunk_id="chk_1",
    )
    claim = AnswerClaim(
        claim_id="c1",
        text="Inspect the panel visually.",
        supports=[
            ClaimSupport(
                claim_id="c1",
                evidence_id="ev_1",
                citation_number=1,
                quote="Inspect the panel visually.",
            )
        ],
    )
    return AnswerResponse(
        question=question,
        abstained=abstain,
        abstention_reason="insufficient_evidence" if abstain else None,
        explanation="No supported answer." if abstain else None,
        answer=None if abstain else "Inspect the panel visually. [1]",
        claims=[] if abstain else [claim],
        citations=[] if abstain else [citation],
        evidence=[],
        index_profile_id="idx_1",
        corpus_fingerprint="fp_1",
        requested_top_k=5,
        generation_format_version="1",
        provider=ProviderIdentity(name="fake", revision="r", device="cpu"),
        context_character_count=0,
        prompt_token_count=0,
        output_token_count=0,
        latency_seconds=0.0,
        audit_id="a" * 32,
        grounding=GroundingOutcome(),
    )


class FakeAnswers:
    def __init__(self, behaviours: list[object] | None = None) -> None:
        self.behaviours = behaviours
        self.requests: list[AnswerRequest] = []

    def answer(self, request: AnswerRequest) -> AnswerResponse:
        index = len(self.requests)
        self.requests.append(request)
        behaviour: object = "answer"
        if self.behaviours is not None:
            behaviour = self.behaviours[index]
        if isinstance(behaviour, Exception):
            raise behaviour
        return _response(request.question, abstain=behaviour == "abstain")


class StepClock:
    def __init__(self) -> None:
        self.tick = 0

    def __call__(self) -> datetime:
        self.tick += 1
        return datetime(2026, 1, 1, 0, 0, self.tick, tzinfo=UTC)


def _settings(tmp_path: Path, **updates: object) -> Settings:
    base = Settings(_env_file=None, work_order_root=tmp_path / "wo")  # type: ignore[call-arg]
    return base.model_copy(update=updates)


def _service(
    tmp_path: Path,
    answers: FakeAnswers | None = None,
    **updates: object,
) -> WorkOrderService:
    return WorkOrderService(
        _settings(tmp_path, **updates),
        answers or FakeAnswers(),
        clock=StepClock(),
    )


DEFECT = DefectReport(
    title="Cracked  access panel",
    description="Hairline crack near fastener row.",
    component="Panel 123",
)
CONFIRM = ReviewRequest(
    decision="confirm",
    note="Checked against the manual.",
    acknowledge_human_responsibility=True,
    reviewer="alice",
)


def test_defect_report_normalizes_and_rejects_blank_text() -> None:
    assert DEFECT.title == "Cracked access panel"
    assert normalize_text("a\u0000\tb\n c") == "a b c"
    with pytest.raises(ValidationError):
        DefectReport(title="x", description="   ")
    with pytest.raises(ValidationError):
        DefectReport(title="x", description="y", unknown="z")  # type: ignore[call-arg]


def test_planner_uses_fixed_templates_and_is_deterministic() -> None:
    first = plan_questions(DEFECT)
    assert first == plan_questions(DEFECT)
    assert [item.purpose for item in first] == [
        "inspection",
        "limits",
        "repair_or_replacement",
        "required_approvals",
    ]
    assert all("Cracked access panel (Panel 123)" in item.question for item in first)
    assert all(len(item.question) < 500 for item in first)


def test_draft_is_cited_unreviewed_and_never_approves(tmp_path: Path) -> None:
    answers = FakeAnswers()
    service = _service(tmp_path, answers)
    draft = service.draft(DEFECT)
    assert draft.status == "draft"
    assert draft.review is None
    assert draft.created_by is None
    assert draft.created_by_verified is False
    assert len(draft.findings) == 4
    assert all(f.requires_human_verification for f in draft.findings)
    assert all(f.citations for f in draft.findings)
    assert draft.index_profile_id == "idx_1"
    assert any("does not approve maintenance" in notice for notice in draft.notices)
    assert service.get(draft.work_order_id) == draft
    assert answers.requests[0].document_ids is None


def test_document_scope_is_forwarded(tmp_path: Path) -> None:
    answers = FakeAnswers()
    scoped = DEFECT.model_copy(update={"document_ids": ["doc_" + "a" * 32]})
    _service(tmp_path, answers).draft(scoped)
    assert all(r.document_ids == scoped.document_ids for r in answers.requests)


def test_abstentions_and_partial_failures_are_recorded(tmp_path: Path) -> None:
    answers = FakeAnswers(
        ["abstain", GenerationFailedError(), "answer", "abstain"],
    )
    draft = _service(tmp_path, answers).draft(DEFECT)
    assert [f.outcome for f in draft.findings] == [
        "abstained",
        "failed",
        "supported",
        "abstained",
    ]
    assert draft.findings[0].abstention_reason == "insufficient_evidence"
    assert draft.findings[1].failure_code == "generation_failed"
    assert draft.findings[1].citations == []


def test_all_failed_findings_raise_and_store_nothing(tmp_path: Path) -> None:
    error = LlmModelUnavailableError()
    service = _service(tmp_path, FakeAnswers([error] * 4))
    with pytest.raises(LlmModelUnavailableError):
        service.draft(DEFECT)
    assert service.list().total == 0


def test_invalid_answer_request_becomes_invalid_defect(tmp_path: Path) -> None:
    service = _service(tmp_path, FakeAnswers([InvalidAnswerRequestError()]))
    with pytest.raises(InvalidDefectReportError):
        service.draft(DEFECT)


def test_confirm_requires_acknowledgement(tmp_path: Path) -> None:
    service = _service(tmp_path)
    draft = service.draft(DEFECT)
    request = CONFIRM.model_copy(update={"acknowledge_human_responsibility": False})
    with pytest.raises(AcknowledgementRequiredError):
        service.review(draft.work_order_id, request)
    assert service.get(draft.work_order_id).status == "draft"


def test_reject_does_not_require_acknowledgement(tmp_path: Path) -> None:
    service = _service(tmp_path)
    draft = service.draft(DEFECT)
    request = ReviewRequest(decision="reject", note="Wrong component.", reviewer="bob")
    reviewed = service.review(draft.work_order_id, request)
    assert reviewed.status == "human_rejected"
    assert reviewed.review is not None
    assert reviewed.review.approves_maintenance is False


def test_unverified_reviewer_must_name_themselves(tmp_path: Path) -> None:
    service = _service(tmp_path)
    draft = service.draft(DEFECT)
    anonymous = CONFIRM.model_copy(update={"reviewer": None})
    with pytest.raises(ReviewerIdentityError):
        service.review(draft.work_order_id, anonymous)


def test_confirm_records_unverified_identity_and_keeps_content_hash(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    draft = service.draft(DEFECT)
    reviewed = service.review(draft.work_order_id, CONFIRM)
    assert reviewed.status == "human_confirmed"
    assert reviewed.review is not None
    assert reviewed.review.reviewer == "alice"
    assert reviewed.review.reviewer_identity_verified is False
    assert reviewed.content_sha256 == draft.content_sha256
    assert service.get(draft.work_order_id) == reviewed


def test_review_is_final(tmp_path: Path) -> None:
    service = _service(tmp_path)
    draft = service.draft(DEFECT)
    service.review(draft.work_order_id, CONFIRM)
    with pytest.raises(WorkOrderAlreadyReviewedError):
        service.review(draft.work_order_id, CONFIRM)
    reject = ReviewRequest(decision="reject", note="Changed my mind.", reviewer="a")
    with pytest.raises(WorkOrderAlreadyReviewedError):
        service.review(draft.work_order_id, reject)


def test_verified_reviewer_is_taken_from_token_not_body(tmp_path: Path) -> None:
    service = _service(tmp_path)
    draft = service.draft(DEFECT, principal_subject="author")
    assert draft.created_by == "author"
    assert draft.created_by_verified is True
    with pytest.raises(ReviewerIdentityError):
        service.review(draft.work_order_id, CONFIRM, principal_subject="reviewer")
    body = CONFIRM.model_copy(update={"reviewer": None})
    reviewed = service.review(draft.work_order_id, body, principal_subject="reviewer")
    assert reviewed.review is not None
    assert reviewed.review.reviewer == "reviewer"
    assert reviewed.review.reviewer_identity_verified is True


def test_author_cannot_confirm_own_draft_when_separation_required(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    draft = service.draft(DEFECT, principal_subject="author")
    body = CONFIRM.model_copy(update={"reviewer": None})
    with pytest.raises(SeparateReviewerRequiredError):
        service.review(draft.work_order_id, body, principal_subject="author")
    assert service.get(draft.work_order_id).status == "draft"
    reject = ReviewRequest(decision="reject", note="Withdrawn.")
    assert (
        service.review(draft.work_order_id, reject, principal_subject="author").status
        == "human_rejected"
    )


def test_separation_can_be_disabled(tmp_path: Path) -> None:
    service = _service(tmp_path, work_order_require_separate_reviewer=False)
    draft = service.draft(DEFECT, principal_subject="author")
    body = CONFIRM.model_copy(update={"reviewer": None})
    reviewed = service.review(draft.work_order_id, body, principal_subject="author")
    assert reviewed.status == "human_confirmed"


def test_list_is_newest_first_and_paginated(tmp_path: Path) -> None:
    service = _service(tmp_path)
    ids = [service.draft(DEFECT).work_order_id for _ in range(3)]
    listing = service.list(limit=2)
    assert listing.total == 3
    assert [item.work_order_id for item in listing.items] == [ids[2], ids[1]]
    assert listing.items[0].supported_finding_count == 4
    assert [i.work_order_id for i in service.list(offset=2, limit=2).items] == [ids[0]]


def test_get_unknown_and_malformed_ids_are_not_found(tmp_path: Path) -> None:
    service = _service(tmp_path)
    for bad in ("wo_" + "0" * 32, "../etc/passwd", "wo_short", ""):
        with pytest.raises(WorkOrderNotFoundError):
            service.get(bad)


def _stored_path(tmp_path: Path, work_order_id: str) -> Path:
    return tmp_path / "wo" / f"{work_order_id}.json"


def test_tampered_content_is_detected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    draft = service.draft(DEFECT)
    path = _stored_path(tmp_path, draft.work_order_id)
    data = json.loads(path.read_text())
    data["findings"][0]["claims"][0]["text"] = "Skip the inspection."
    path.write_text(json.dumps(data))
    with pytest.raises(WorkOrderIntegrityError):
        service.get(draft.work_order_id)


def test_status_review_mismatch_and_garbage_are_detected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    draft = service.draft(DEFECT)
    path = _stored_path(tmp_path, draft.work_order_id)
    data = json.loads(path.read_text())
    data["status"] = "human_confirmed"
    path.write_text(json.dumps(data))
    with pytest.raises(WorkOrderIntegrityError):
        service.get(draft.work_order_id)
    path.write_text("not json")
    with pytest.raises(WorkOrderIntegrityError):
        service.get(draft.work_order_id)


def test_file_renamed_to_another_id_is_rejected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    draft = service.draft(DEFECT)
    other = "wo_" + "f" * 32
    _stored_path(tmp_path, draft.work_order_id).rename(_stored_path(tmp_path, other))
    with pytest.raises(WorkOrderIntegrityError):
        service.get(other)


def test_symlinked_root_and_file_are_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(WorkOrderPersistenceError):
        WorkOrderStore(link).list_all()

    service = _service(tmp_path)
    draft = service.draft(DEFECT)
    path = _stored_path(tmp_path, draft.work_order_id)
    target = tmp_path / "outside.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(WorkOrderIntegrityError):
        service.get(draft.work_order_id)


def test_store_refuses_overwrite_on_create(tmp_path: Path) -> None:
    service = _service(tmp_path)
    draft = service.draft(DEFECT)
    with pytest.raises(WorkOrderPersistenceError):
        service.store.create(draft)


def test_no_temporary_files_remain(tmp_path: Path) -> None:
    service = _service(tmp_path)
    draft = service.draft(DEFECT)
    service.review(draft.work_order_id, CONFIRM)
    assert [p.name for p in (tmp_path / "wo").iterdir()] == [
        f"{draft.work_order_id}.json"
    ]


# --- HTTP -----------------------------------------------------------------


def _patched_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    validator: object | None = None,
    **updates: object,
) -> TestClient:
    monkeypatch.setattr(
        "app.services.runtime.AnswerService.answer",
        lambda self, request: FakeAnswers().answer(request),
    )
    settings = _settings(tmp_path, **updates)
    if validator is None:
        return TestClient(create_app(settings))
    return TestClient(
        create_app(
            settings.model_copy(update={"security_mode": "test"}),
            access_token_validator=validator,  # type: ignore[arg-type]
        )
    )


def test_http_lifecycle_in_disabled_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with _patched_client(monkeypatch, tmp_path) as client:
        body = {"title": "Cracked panel", "description": "Hairline crack."}
        created = client.post("/work-orders", json=body)
        assert created.status_code == 201
        work_order_id = created.json()["work_order_id"]
        assert created.json()["status"] == "draft"

        assert client.get("/work-orders").json()["total"] == 1
        assert client.get(f"/work-orders/{work_order_id}").status_code == 200
        assert client.get("/work-orders/wo_" + "0" * 32).status_code == 404

        review_url = f"/work-orders/{work_order_id}/review"
        no_ack = client.post(
            review_url,
            json={"decision": "confirm", "note": "ok", "reviewer": "alice"},
        )
        assert no_ack.status_code == 422
        assert no_ack.json()["error"]["code"] == "acknowledgement_required"

        confirmed = client.post(
            review_url,
            json={
                "decision": "confirm",
                "note": "ok",
                "reviewer": "alice",
                "acknowledge_human_responsibility": True,
            },
        )
        assert confirmed.status_code == 200
        assert confirmed.json()["status"] == "human_confirmed"
        assert confirmed.json()["review"]["approves_maintenance"] is False
        again = client.post(
            review_url,
            json={"decision": "reject", "note": "no", "reviewer": "alice"},
        )
        assert again.status_code == 409


def test_http_rejects_bad_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with _patched_client(monkeypatch, tmp_path) as client:
        assert client.post("/work-orders", json={"title": "x"}).status_code == 422
        too_long = {"title": "x", "description": "y" * 2001}
        assert client.post("/work-orders", json=too_long).status_code == 422
        assert client.get("/work-orders?limit=0").status_code == 422
        assert client.get("/work-orders?limit=101").status_code == 422


class _TokenValidator:
    def __init__(self, principals: dict[str, Principal]) -> None:
        self.principals = principals

    async def validate(self, token: str) -> Principal:
        if token not in self.principals:
            raise InvalidCredentialsError
        return self.principals[token]


def _bearer(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


def test_http_roles_and_verified_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    validator = _TokenValidator(
        {
            "a.b.viewer": Principal(
                subject="viewer", roles=frozenset({Role.EVIDENCE_VIEWER})
            ),
            "a.b.author": Principal(
                subject="author", roles=frozenset({Role.DOCUMENT_OPERATOR})
            ),
            "a.b.second": Principal(
                subject="second", roles=frozenset({Role.DOCUMENT_OPERATOR})
            ),
        }
    )
    body = {"title": "Cracked panel", "description": "Hairline crack."}
    with _patched_client(monkeypatch, tmp_path, validator) as client:
        assert client.post("/work-orders", json=body).status_code == 401
        assert (
            client.post("/work-orders", json=body, headers=_bearer("a.b.viewer"))
        ).status_code == 403

        created = client.post("/work-orders", json=body, headers=_bearer("a.b.author"))
        assert created.status_code == 201
        assert created.json()["created_by"] == "author"
        assert created.json()["created_by_verified"] is True
        work_order_id = created.json()["work_order_id"]

        viewer_get = client.get(
            f"/work-orders/{work_order_id}", headers=_bearer("a.b.viewer")
        )
        assert viewer_get.status_code == 200
        assert (
            client.get("/work-orders", headers=_bearer("a.b.viewer")).status_code == 200
        )

        review_url = f"/work-orders/{work_order_id}/review"
        confirm = {
            "decision": "confirm",
            "note": "checked",
            "acknowledge_human_responsibility": True,
        }
        assert (
            client.post(review_url, json=confirm, headers=_bearer("a.b.viewer"))
        ).status_code == 403
        own = client.post(review_url, json=confirm, headers=_bearer("a.b.author"))
        assert own.status_code == 403
        assert own.json()["error"]["code"] == "separate_reviewer_required"
        spoofed = client.post(
            review_url,
            json={**confirm, "reviewer": "someone-else"},
            headers=_bearer("a.b.second"),
        )
        assert spoofed.status_code == 422
        done = client.post(review_url, json=confirm, headers=_bearer("a.b.second"))
        assert done.status_code == 200
        assert done.json()["review"]["reviewer"] == "second"
        assert done.json()["review"]["reviewer_identity_verified"] is True
