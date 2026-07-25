"""Deterministic grounding and abstention evaluation for generated answers."""

import argparse
import json
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.generation.errors import GenerationError, GenerationOutputInvalidError
from app.generation.models import AnswerRequest, AnswerResponse
from app.generation.provider import TransformersLlmProvider
from app.generation.service import AnswerService, _lexical_support
from app.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from app.retrieval.errors import RetrievalError
from app.retrieval.service import RetrievalService

ExpectedOutcome = Literal[
    "answer",
    "policy_abstention",
    "evidence_abstention",
    "conflict_abstention",
]
ObservedOutcome = Literal[
    "answer",
    "policy_abstention",
    "evidence_abstention",
    "conflict_abstention",
    "malformed_generation",
    "operational_failure",
]


class GenerationEvaluationCase(BaseModel):
    case_id: str
    question: str
    expected_outcome: ExpectedOutcome
    document_ids: list[str] | None = None
    notes: str | None = None


class GenerationCaseResult(BaseModel):
    case_id: str
    expected_outcome: ExpectedOutcome
    observed_outcome: ObservedOutcome
    execution_succeeded: bool
    claim_count: int | None
    cited_claim_count: int | None
    citation_count: int | None
    valid_citation_count: int | None
    support_count: int | None
    valid_evidence_id_count: int | None
    quote_count: int | None
    valid_quote_count: int | None
    lexically_supported_claim_count: int | None
    failure_category: str | None


class GenerationEvaluationReport(BaseModel):
    case_count: int
    execution_success_rate: float
    operational_failure_rate: float
    malformed_output_rate: float
    policy_abstention_precision: float
    policy_abstention_recall: float
    policy_abstention_accuracy: float
    evidence_abstention_precision: float
    evidence_abstention_recall: float
    evidence_abstention_accuracy: float
    conflict_precision: float
    conflict_recall: float
    valid_citation_rate: float
    claim_citation_coverage: float
    evidence_id_integrity_rate: float
    exact_quote_validity_rate: float
    lexical_support_proxy_rate: float
    unsupported_claim_proxy_rate: float
    cases: list[GenerationCaseResult]


def evaluate_answers(
    service: AnswerService,
    cases: list[GenerationEvaluationCase],
) -> GenerationEvaluationReport:
    """Evaluate structural grounding without an LLM or embedding judge."""
    outcomes: list[GenerationCaseResult] = []
    for case in sorted(cases, key=lambda item: item.case_id):
        try:
            response = service.answer(
                AnswerRequest(
                    question=case.question,
                    document_ids=case.document_ids,
                )
            )
        except GenerationOutputInvalidError as error:
            outcomes.append(_failure(case, "malformed_generation", error.code))
            continue
        except (GenerationError, RetrievalError) as error:
            outcomes.append(_failure(case, "operational_failure", error.code))
            continue
        outcomes.append(_evaluate_response(case, response))
    return _summarize(outcomes)


def _evaluate_response(
    case: GenerationEvaluationCase,
    response: AnswerResponse,
) -> GenerationCaseResult:
    evidence = {item.evidence_id: item for item in response.evidence}
    citations = {item.number: item for item in response.citations}
    supports = [support for claim in response.claims for support in claim.supports]
    valid_citations = sum(
        citation.evidence_id in evidence for citation in response.citations
    )
    valid_ids = sum(support.evidence_id in evidence for support in supports)
    quoted = [support for support in supports if support.quote is not None]
    valid_quotes = sum(
        support.evidence_id in evidence
        and support.quote is not None
        and " ".join(support.quote.split())
        in " ".join(evidence[support.evidence_id].text.split())
        for support in quoted
    )
    supported = sum(
        _lexical_support(
            claim.text,
            " ".join(
                evidence[support.evidence_id].text
                for support in claim.supports
                if support.evidence_id in evidence
            ),
        )
        >= 0.60
        for claim in response.claims
    )
    cited_claims = sum(
        bool(claim.supports)
        and all(support.citation_number in citations for support in claim.supports)
        for claim in response.claims
    )
    return GenerationCaseResult(
        case_id=case.case_id,
        expected_outcome=case.expected_outcome,
        observed_outcome=_observed(response),
        execution_succeeded=True,
        claim_count=len(response.claims),
        cited_claim_count=cited_claims,
        citation_count=len(response.citations),
        valid_citation_count=valid_citations,
        support_count=len(supports),
        valid_evidence_id_count=valid_ids,
        quote_count=len(quoted),
        valid_quote_count=valid_quotes,
        lexically_supported_claim_count=supported,
        failure_category=None,
    )


def _failure(
    case: GenerationEvaluationCase,
    outcome: Literal["malformed_generation", "operational_failure"],
    category: str,
) -> GenerationCaseResult:
    return GenerationCaseResult(
        case_id=case.case_id,
        expected_outcome=case.expected_outcome,
        observed_outcome=outcome,
        execution_succeeded=False,
        claim_count=None,
        cited_claim_count=None,
        citation_count=None,
        valid_citation_count=None,
        support_count=None,
        valid_evidence_id_count=None,
        quote_count=None,
        valid_quote_count=None,
        lexically_supported_claim_count=None,
        failure_category=category,
    )


def _summarize(cases: list[GenerationCaseResult]) -> GenerationEvaluationReport:
    count = len(cases)
    if count == 0:
        return GenerationEvaluationReport(
            case_count=0,
            execution_success_rate=0.0,
            operational_failure_rate=0.0,
            malformed_output_rate=0.0,
            policy_abstention_precision=0.0,
            policy_abstention_recall=0.0,
            policy_abstention_accuracy=0.0,
            evidence_abstention_precision=0.0,
            evidence_abstention_recall=0.0,
            evidence_abstention_accuracy=0.0,
            conflict_precision=0.0,
            conflict_recall=0.0,
            valid_citation_rate=0.0,
            claim_citation_coverage=0.0,
            evidence_id_integrity_rate=0.0,
            exact_quote_validity_rate=0.0,
            lexical_support_proxy_rate=0.0,
            unsupported_claim_proxy_rate=0.0,
            cases=[],
        )
    claims = sum(item.claim_count or 0 for item in cases)
    citations = sum(item.citation_count or 0 for item in cases)
    supports = sum(item.support_count or 0 for item in cases)
    quotes = sum(item.quote_count or 0 for item in cases)
    supported = sum(item.lexically_supported_claim_count or 0 for item in cases)
    return GenerationEvaluationReport(
        case_count=count,
        execution_success_rate=sum(item.execution_succeeded for item in cases) / count,
        operational_failure_rate=(
            sum(item.observed_outcome == "operational_failure" for item in cases)
            / count
        ),
        malformed_output_rate=(
            sum(item.observed_outcome == "malformed_generation" for item in cases)
            / count
        ),
        policy_abstention_precision=_precision(cases, "policy_abstention"),
        policy_abstention_recall=_recall(cases, "policy_abstention"),
        policy_abstention_accuracy=_accuracy(cases, "policy_abstention"),
        evidence_abstention_precision=_precision(cases, "evidence_abstention"),
        evidence_abstention_recall=_recall(cases, "evidence_abstention"),
        evidence_abstention_accuracy=_accuracy(cases, "evidence_abstention"),
        conflict_precision=_precision(cases, "conflict_abstention"),
        conflict_recall=_recall(cases, "conflict_abstention"),
        valid_citation_rate=(
            0.0
            if citations == 0
            else sum(item.valid_citation_count or 0 for item in cases) / citations
        ),
        claim_citation_coverage=(
            0.0
            if claims == 0
            else sum(item.cited_claim_count or 0 for item in cases) / claims
        ),
        evidence_id_integrity_rate=(
            0.0
            if supports == 0
            else sum(item.valid_evidence_id_count or 0 for item in cases) / supports
        ),
        exact_quote_validity_rate=(
            0.0
            if quotes == 0
            else sum(item.valid_quote_count or 0 for item in cases) / quotes
        ),
        lexical_support_proxy_rate=(0.0 if claims == 0 else supported / claims),
        unsupported_claim_proxy_rate=(
            0.0 if claims == 0 else (claims - supported) / claims
        ),
        cases=cases,
    )


def _observed(response: AnswerResponse) -> ExpectedOutcome:
    if not response.abstained:
        return "answer"
    if response.abstention_reason == "safety_boundary":
        return "policy_abstention"
    if response.abstention_reason == "conflicting_evidence":
        return "conflict_abstention"
    return "evidence_abstention"


def _precision(cases: list[GenerationCaseResult], outcome: ExpectedOutcome) -> float:
    predicted = sum(item.observed_outcome == outcome for item in cases)
    true_positive = sum(
        item.observed_outcome == outcome and item.expected_outcome == outcome
        for item in cases
    )
    return 0.0 if predicted == 0 else true_positive / predicted


def _recall(cases: list[GenerationCaseResult], outcome: ExpectedOutcome) -> float:
    expected = sum(item.expected_outcome == outcome for item in cases)
    true_positive = sum(
        item.observed_outcome == outcome and item.expected_outcome == outcome
        for item in cases
    )
    return 0.0 if expected == 0 else true_positive / expected


def _accuracy(cases: list[GenerationCaseResult], outcome: ExpectedOutcome) -> float:
    correct = sum(
        (item.observed_outcome == outcome) == (item.expected_outcome == outcome)
        for item in cases
    )
    return correct / len(cases)


def load_cases(path: Path) -> list[GenerationEvaluationCase]:
    """Load manually inspectable JSONL evaluation cases."""
    try:
        with path.open("r", encoding="utf-8") as source:
            return [
                GenerationEvaluationCase.model_validate_json(line)
                for line in source
                if line.strip()
            ]
    except (OSError, UnicodeError, ValidationError, ValueError) as error:
        raise SystemExit("Generation evaluation cases are invalid.") from error


def write_report(path: Path, report: GenerationEvaluationReport) -> None:
    """Atomically write an ignored local evaluation report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(
                report.model_dump(mode="json"),
                output,
                ensure_ascii=False,
                sort_keys=True,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    """Run local generation evaluation using configured local providers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/runtime/generation/evaluations/latest.json"),
    )
    arguments = parser.parse_args()
    settings = Settings()
    retrieval = RetrievalService(
        settings,
        SentenceTransformerEmbeddingProvider(settings),
    )
    service = AnswerService(
        settings,
        retrieval,
        TransformersLlmProvider(settings),
    )
    report = evaluate_answers(service, load_cases(arguments.cases))
    write_report(arguments.output, report)


if __name__ == "__main__":
    main()
