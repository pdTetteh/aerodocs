"""Grounded answer orchestration over validated retrieval evidence."""

import hashlib
import re
import time
from datetime import UTC, datetime
from uuid import uuid4

from app.core.config import Settings
from app.generation.audit import GenerationAuditStore
from app.generation.context import (
    ContextAssembly,
    assemble_context,
    requests_unsafe_decision,
    system_prompt,
    user_prompt,
)
from app.generation.errors import (
    AnswerResourceLimitError,
    GenerationFailedError,
    GenerationOutputInvalidError,
    GenerationTimeoutError,
    InvalidAnswerRequestError,
    LlmModelUnavailableError,
)
from app.generation.grounding import semantic_compatible
from app.generation.models import (
    AnswerClaim,
    AnswerRequest,
    AnswerResponse,
    AuditStatus,
    Citation,
    ClaimSupport,
    DraftAnswer,
    DraftConflict,
    EvidencePassage,
    GenerationAudit,
    GroundingOutcome,
)
from app.generation.parsing import parse_generation_draft
from app.generation.provider import LlmProvider
from app.ingestion.storage import DOCUMENT_ID_PATTERN
from app.retrieval.errors import RetrievalError, StaleIndexError
from app.retrieval.models import SearchRequest, SearchResponse
from app.retrieval.service import RetrievalService

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-/.][a-z0-9]+)*", re.I)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "with",
}


class AnswerService:
    """Retrieve evidence, generate claims, validate grounding, and audit."""

    def __init__(
        self,
        settings: Settings,
        retrieval: RetrievalService,
        provider: LlmProvider,
    ) -> None:
        self.settings = settings
        self.retrieval = retrieval
        self.provider = provider
        self.audits = GenerationAuditStore(settings.generation_audit_root)

    def answer(self, request: AnswerRequest) -> AnswerResponse:
        """Return a supported citation-linked answer or explicit abstention."""
        started = time.perf_counter()
        audit_id = uuid4().hex
        question = request.question.strip()
        try:
            question, top_k, document_ids = self._validate_request(request)
        except InvalidAnswerRequestError:
            self._write_audit(
                audit_id=audit_id,
                question=question,
                document_ids=None,
                requested_top_k=request.top_k,
                status="invalid_request_failure",
                reason_code="invalid_answer_request",
                started=started,
            )
            raise

        if requests_unsafe_decision(question):
            return self._policy_abstain(
                question,
                audit_id,
                started,
                document_ids,
                top_k,
            )

        try:
            search = self.retrieval.search(
                SearchRequest(
                    query=question,
                    top_k=top_k,
                    document_ids=document_ids,
                    minimum_score=self.settings.answer_minimum_similarity_score,
                )
            )
        except RetrievalError as error:
            self._write_audit(
                audit_id=audit_id,
                question=question,
                document_ids=document_ids,
                requested_top_k=top_k,
                status=(
                    "stale_index_failure"
                    if isinstance(error, StaleIndexError)
                    else "retrieval_failure"
                ),
                reason_code=error.code,
                started=started,
            )
            raise

        context = assemble_context(search, self.settings)
        if context.injection_detected:
            return self._abstain(
                question,
                search,
                context,
                audit_id,
                started,
                document_ids,
                "prompt_injection_risk",
                "injection_abstention",
                "Retrieved text contains instruction-like content and was quarantined.",
            )
        if context.conflict_evidence_ids:
            return self._abstain(
                question,
                search,
                context,
                audit_id,
                started,
                document_ids,
                "conflicting_evidence",
                "conflict_abstention",
                "Retrieved passages conflict. Consult the controlling documents "
                "and authorized engineering personnel.",
                cited_evidence_ids=context.conflict_evidence_ids,
            )
        if not context.evidence:
            return self._abstain(
                question,
                search,
                context,
                audit_id,
                started,
                document_ids,
                "insufficient_evidence",
                "evidence_abstention",
                "No retrieved passage met the configured evidence threshold.",
            )

        system = system_prompt()
        user = user_prompt(question, context.context_json)
        try:
            prepared = self.provider.prepare(system, user)
        except LlmModelUnavailableError as error:
            self._operational_failure(
                audit_id,
                question,
                document_ids,
                top_k,
                search,
                context,
                "model_load_failure",
                error.code,
                started,
            )
            raise
        except GenerationFailedError as error:
            self._operational_failure(
                audit_id,
                question,
                document_ids,
                top_k,
                search,
                context,
                "generation_failure",
                error.code,
                started,
            )
            raise
        prompt_tokens = prepared.prompt_token_count
        if (
            prompt_tokens > self.settings.maximum_generation_prompt_tokens
            or prompt_tokens + prepared.maximum_new_tokens
            > prepared.model_context_limit
        ):
            self._operational_failure(
                audit_id,
                question,
                document_ids,
                top_k,
                search,
                context,
                "prompt_limit_failure",
                "answer_resource_limit",
                started,
                prompt_tokens=prompt_tokens,
            )
            raise AnswerResourceLimitError
        try:
            raw = self.provider.generate(prepared)
        except GenerationTimeoutError as error:
            self._operational_failure(
                audit_id,
                question,
                document_ids,
                top_k,
                search,
                context,
                "generation_timeout",
                error.code,
                started,
                prompt_tokens=prompt_tokens,
                timeout_stage=error.stage,
            )
            raise
        except LlmModelUnavailableError as error:
            self._operational_failure(
                audit_id,
                question,
                document_ids,
                top_k,
                search,
                context,
                "model_load_failure",
                error.code,
                started,
                prompt_tokens=prompt_tokens,
            )
            raise
        except GenerationFailedError as error:
            self._operational_failure(
                audit_id,
                question,
                document_ids,
                top_k,
                search,
                context,
                "generation_failure",
                error.code,
                started,
                prompt_tokens=prompt_tokens,
            )
            raise
        try:
            output_tokens = self.provider.count_tokens(raw)
        except GenerationFailedError as error:
            self._operational_failure(
                audit_id,
                question,
                document_ids,
                top_k,
                search,
                context,
                "generation_failure",
                error.code,
                started,
                prompt_tokens=prompt_tokens,
            )
            raise
        if output_tokens > self.settings.maximum_generation_output_tokens:
            self._operational_failure(
                audit_id,
                question,
                document_ids,
                top_k,
                search,
                context,
                "output_limit_failure",
                "answer_resource_limit",
                started,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
            raise AnswerResourceLimitError
        try:
            draft = parse_generation_draft(raw)
            return self._ground(
                question,
                search,
                context,
                draft,
                audit_id,
                started,
                document_ids,
                prompt_tokens,
                output_tokens,
            )
        except GenerationOutputInvalidError as error:
            self._operational_failure(
                audit_id,
                question,
                document_ids,
                top_k,
                search,
                context,
                "invalid_output_failure",
                error.code,
                started,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
            raise

    def _ground(
        self,
        question: str,
        search: SearchResponse,
        context: ContextAssembly,
        draft: object,
        audit_id: str,
        started: float,
        document_ids: list[str] | None,
        prompt_tokens: int,
        output_tokens: int,
    ) -> AnswerResponse:
        if not isinstance(draft, DraftAnswer):
            if isinstance(draft, DraftConflict):
                identifiers = draft.conflicting_evidence_ids
                known = {item.evidence_id for item in context.evidence}
                if (
                    len(identifiers) < 2
                    or len(identifiers)
                    > self.settings.maximum_generation_conflict_evidence_ids
                    or len(set(identifiers)) != len(identifiers)
                    or not set(identifiers) <= known
                ):
                    raise GenerationOutputInvalidError
                return self._abstain(
                    question,
                    search,
                    context,
                    audit_id,
                    started,
                    document_ids,
                    "conflicting_evidence",
                    "conflict_abstention",
                    "The supplied passages appear to conflict. Consult the "
                    "controlling documents and authorized engineering personnel.",
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    cited_evidence_ids=identifiers,
                )
            return self._abstain(
                question,
                search,
                context,
                audit_id,
                started,
                document_ids,
                "insufficient_evidence",
                "evidence_abstention",
                "The local model found the supplied evidence insufficient.",
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        if not 1 <= len(draft.claims) <= self.settings.maximum_generation_claims:
            raise GenerationOutputInvalidError
        evidence = {item.evidence_id: item for item in context.evidence}
        claims: list[AnswerClaim] = []
        citations: list[Citation] = []
        citation_by_evidence: dict[str, int] = {}
        for claim_number, generated_claim in enumerate(draft.claims, start=1):
            claim_id = f"claim_{claim_number}"
            text = generated_claim.text.strip()
            if (
                not text
                or len(text) > self.settings.maximum_generation_claim_characters
                or not 1
                <= len(generated_claim.supports)
                <= self.settings.maximum_generation_supports_per_claim
            ):
                raise GenerationOutputInvalidError
            identifiers = [support.evidence_id for support in generated_claim.supports]
            if len(set(identifiers)) != len(identifiers):
                raise GenerationOutputInvalidError
            unknown = set(identifiers) - evidence.keys()
            if unknown:
                raise GenerationOutputInvalidError
            relevant_text: list[str] = []
            supports: list[ClaimSupport] = []
            for generated_support in generated_claim.supports:
                passage = evidence[generated_support.evidence_id]
                quote = generated_support.quote
                if quote is not None:
                    quote = quote.strip()
                    if (
                        not quote
                        or len(quote)
                        > self.settings.maximum_generation_quote_characters
                        or _normalize_space(quote) not in _normalize_space(passage.text)
                    ):
                        raise GenerationOutputInvalidError
                    relevant_text.append(quote)
                else:
                    relevant_text.append(passage.text)
                number = citation_by_evidence.get(passage.evidence_id)
                if number is None:
                    number = len(citations) + 1
                    citation_by_evidence[passage.evidence_id] = number
                    citations.append(_citation(number, passage))
                supports.append(
                    ClaimSupport(
                        claim_id=claim_id,
                        evidence_id=passage.evidence_id,
                        citation_number=number,
                        quote=quote,
                    )
                )
            support_text = " ".join(relevant_text)
            if not semantic_compatible(text, support_text):
                return self._grounding_abstention(
                    question,
                    search,
                    context,
                    audit_id,
                    started,
                    document_ids,
                    prompt_tokens,
                    output_tokens,
                    len(draft.claims),
                    len(claims),
                    len(citations),
                    semantic_mismatch=True,
                )
            if (
                _lexical_support(text, support_text)
                < self.settings.minimum_claim_lexical_support
            ):
                return self._grounding_abstention(
                    question,
                    search,
                    context,
                    audit_id,
                    started,
                    document_ids,
                    prompt_tokens,
                    output_tokens,
                    len(draft.claims),
                    len(claims),
                    len(citations),
                    semantic_mismatch=False,
                )
            claims.append(AnswerClaim(claim_id=claim_id, text=text, supports=supports))

        outcome = GroundingOutcome(
            claim_count=len(claims),
            supported_claim_count=len(claims),
            citation_count=len(citations),
        )
        elapsed = time.perf_counter() - started
        answer = " ".join(
            f"{claim.text} "
            f"{' '.join(f'[{support.citation_number}]' for support in claim.supports)}"
            for claim in claims
        )
        response = AnswerResponse(
            question=question,
            abstained=False,
            abstention_reason=None,
            explanation=None,
            answer=answer,
            claims=claims,
            citations=citations,
            evidence=context.evidence,
            index_profile_id=search.index_profile_id,
            corpus_fingerprint=search.corpus_fingerprint,
            requested_top_k=search.requested_top_k,
            generation_format_version=self.settings.generation_format_version,
            provider=self.provider.identity(),
            context_character_count=context.character_count,
            prompt_token_count=prompt_tokens,
            output_token_count=output_tokens,
            latency_seconds=elapsed,
            audit_id=audit_id,
            grounding=outcome,
        )
        self._write_audit(
            audit_id=audit_id,
            question=question,
            document_ids=document_ids,
            requested_top_k=search.requested_top_k,
            search=search,
            context=context,
            status="success",
            reason_code=None,
            outcome=outcome,
            started=started,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            claim_count=len(claims),
            evidence_ids=[
                support.evidence_id for claim in claims for support in claim.supports
            ],
        )
        return response

    def _grounding_abstention(
        self,
        question: str,
        search: SearchResponse,
        context: ContextAssembly,
        audit_id: str,
        started: float,
        document_ids: list[str] | None,
        prompt_tokens: int,
        output_tokens: int,
        claim_count: int,
        supported_claim_count: int,
        citation_count: int,
        *,
        semantic_mismatch: bool,
    ) -> AnswerResponse:
        outcome = GroundingOutcome(
            claim_count=claim_count,
            supported_claim_count=supported_claim_count,
            citation_count=citation_count,
            unsupported_claim_count=0 if semantic_mismatch else 1,
            semantic_mismatch_count=1 if semantic_mismatch else 0,
        )
        return self._abstain(
            question,
            search,
            context,
            audit_id,
            started,
            document_ids,
            "grounding_validation_failed",
            "grounding_failure",
            "At least one generated claim was not adequately grounded in the "
            "cited passages.",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            outcome=outcome,
        )

    def _policy_abstain(
        self,
        question: str,
        audit_id: str,
        started: float,
        document_ids: list[str] | None,
        requested_top_k: int,
    ) -> AnswerResponse:
        """Abstain before retrieval or model loading on autonomous decisions."""
        outcome = GroundingOutcome()
        self._write_audit(
            audit_id=audit_id,
            question=question,
            document_ids=document_ids,
            requested_top_k=requested_top_k,
            status="policy_abstention",
            reason_code="safety_boundary",
            started=started,
            outcome=outcome,
        )
        return AnswerResponse(
            question=question,
            abstained=True,
            abstention_reason="safety_boundary",
            explanation=(
                "AeroDocs cannot approve maintenance, determine airworthiness, "
                "or authorize release to service."
            ),
            answer=None,
            claims=[],
            citations=[],
            evidence=[],
            index_profile_id="",
            corpus_fingerprint="",
            requested_top_k=requested_top_k,
            generation_format_version=self.settings.generation_format_version,
            provider=self.provider.identity(),
            context_character_count=0,
            prompt_token_count=0,
            output_token_count=0,
            latency_seconds=time.perf_counter() - started,
            audit_id=audit_id,
            grounding=outcome,
        )

    def _abstain(
        self,
        question: str,
        search: SearchResponse,
        context: ContextAssembly,
        audit_id: str,
        started: float,
        document_ids: list[str] | None,
        reason: str,
        status: AuditStatus,
        explanation: str,
        *,
        prompt_tokens: int = 0,
        output_tokens: int = 0,
        outcome: GroundingOutcome | None = None,
        cited_evidence_ids: list[str] | None = None,
    ) -> AnswerResponse:
        grounding = outcome or GroundingOutcome()
        identifiers = cited_evidence_ids or []
        evidence_by_id = {item.evidence_id: item for item in context.evidence}
        citations = [
            _citation(number, evidence_by_id[evidence_id])
            for number, evidence_id in enumerate(identifiers, start=1)
        ]
        elapsed = time.perf_counter() - started
        self._write_audit(
            audit_id=audit_id,
            question=question,
            document_ids=document_ids,
            requested_top_k=search.requested_top_k,
            search=search,
            context=context,
            status=status,
            reason_code=reason,
            outcome=grounding,
            started=started,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            evidence_ids=identifiers,
        )
        return AnswerResponse(
            question=question,
            abstained=True,
            abstention_reason=reason,  # type: ignore[arg-type]
            explanation=explanation,
            answer=None,
            claims=[],
            citations=citations,
            evidence=context.evidence,
            index_profile_id=search.index_profile_id,
            corpus_fingerprint=search.corpus_fingerprint,
            requested_top_k=search.requested_top_k,
            generation_format_version=self.settings.generation_format_version,
            provider=self.provider.identity(),
            context_character_count=context.character_count,
            prompt_token_count=prompt_tokens,
            output_token_count=output_tokens,
            latency_seconds=elapsed,
            audit_id=audit_id,
            grounding=grounding,
        )

    def _operational_failure(
        self,
        audit_id: str,
        question: str,
        document_ids: list[str] | None,
        requested_top_k: int,
        search: SearchResponse,
        context: ContextAssembly,
        status: AuditStatus,
        reason_code: str,
        started: float,
        *,
        prompt_tokens: int = 0,
        output_tokens: int = 0,
        timeout_stage: str | None = None,
    ) -> None:
        self._write_audit(
            audit_id=audit_id,
            question=question,
            document_ids=document_ids,
            requested_top_k=requested_top_k,
            search=search,
            context=context,
            status=status,
            reason_code=reason_code,
            started=started,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            timeout_stage=timeout_stage,
        )

    def _write_audit(
        self,
        *,
        audit_id: str,
        question: str,
        document_ids: list[str] | None,
        requested_top_k: int | None,
        status: AuditStatus,
        reason_code: str | None,
        started: float,
        search: SearchResponse | None = None,
        context: ContextAssembly | None = None,
        outcome: GroundingOutcome | None = None,
        prompt_tokens: int = 0,
        output_tokens: int = 0,
        claim_count: int = 0,
        evidence_ids: list[str] | None = None,
        timeout_stage: str | None = None,
    ) -> None:
        evidence = [] if context is None else context.evidence
        self.audits.write(
            GenerationAudit(
                audit_format_version=self.settings.generation_audit_format_version,
                audit_id=audit_id,
                created_at=datetime.now(UTC),
                question_sha256=hashlib.sha256(question.encode()).hexdigest(),
                requested_document_ids=document_ids,
                requested_top_k=requested_top_k,
                index_profile_id=(None if search is None else search.index_profile_id),
                corpus_fingerprint=(
                    None if search is None else search.corpus_fingerprint
                ),
                selected_chunk_ids=[item.chunk_id for item in evidence],
                selected_scores=[item.score for item in evidence],
                provider=self.provider.identity(),
                status=status,
                reason_code=reason_code,
                claim_count=claim_count,
                evidence_ids=evidence_ids or [],
                grounding=outcome or GroundingOutcome(),
                latency_seconds=time.perf_counter() - started,
                prompt_token_count=prompt_tokens,
                output_token_count=output_tokens,
                timeout_stage=timeout_stage,  # type: ignore[arg-type]
            )
        )

    def _validate_request(
        self,
        request: AnswerRequest,
    ) -> tuple[str, int, list[str] | None]:
        question = request.question.strip()
        top_k = (
            self.settings.answer_default_top_k
            if request.top_k is None
            else request.top_k
        )
        document_ids = request.document_ids
        invalid_document_ids = document_ids is not None and (
            not document_ids
            or any(not isinstance(document_id, str) for document_id in document_ids)
            or len(set(document_ids)) != len(document_ids)
            or any(
                not DOCUMENT_ID_PATTERN.fullmatch(document_id)
                for document_id in document_ids
            )
        )
        if (
            not question
            or len(question) > self.settings.maximum_answer_question_characters
            or not 1 <= top_k <= self.settings.answer_maximum_top_k
            or invalid_document_ids
        ):
            raise InvalidAnswerRequestError
        return question, top_k, document_ids


def _citation(number: int, passage: EvidencePassage) -> Citation:
    return Citation(
        number=number,
        evidence_id=passage.evidence_id,
        score=passage.score,
        document_id=passage.document_id,
        title=passage.title,
        filename=passage.filename,
        page_start=passage.page_start,
        page_end=passage.page_end,
        section_heading=passage.section_heading,
        chunk_id=passage.chunk_id,
    )


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def _lexical_support(claim: str, evidence: str) -> float:
    claim_tokens = {
        token.casefold()
        for token in TOKEN_PATTERN.findall(claim)
        if token.casefold() not in STOP_WORDS
    }
    if not claim_tokens:
        return 0.0
    evidence_tokens = {token.casefold() for token in TOKEN_PATTERN.findall(evidence)}
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)
