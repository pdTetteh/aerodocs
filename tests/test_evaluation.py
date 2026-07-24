"""Tests for genuine deterministic retrieval Recall@k and MRR."""

from typing import cast

import pytest

from app.evaluation.metrics import summarize
from app.evaluation.models import (
    CaseEvaluation,
    EvaluationCase,
    RelevantPageReference,
)
from app.evaluation.runner import evaluate
from app.retrieval.errors import IndexNotReadyError
from app.retrieval.models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.retrieval.service import RetrievalService


class FakeSearchService:
    def __init__(
        self,
        responses: dict[str, list[SearchResult] | Exception],
    ) -> None:
        self.responses = responses

    def search(self, request: SearchRequest) -> SearchResponse:
        value = self.responses[request.query]
        if isinstance(value, Exception):
            raise value
        return SearchResponse(
            query=request.query,
            index_profile_id="ipr_test",
            corpus_fingerprint="cor_test",
            similarity_metric="cosine",
            requested_top_k=5,
            returned_count=len(value),
            results=value,
        )


def result(
    rank: int,
    chunk_id: str,
    *,
    document_id: str = "doc_" + "a" * 64,
    page_start: int = 1,
    page_end: int = 1,
) -> SearchResult:
    return SearchResult(
        rank=rank,
        score=1.0 / rank,
        document_id=document_id,
        filename="manual.pdf",
        title="Manual",
        chunk_id=chunk_id,
        chunk_order=rank,
        page_start=page_start,
        page_end=page_end,
        section_heading=None,
        text=f"passage {rank}",
        character_count=len(f"passage {rank}"),
    )


def run_cases(
    cases: list[EvaluationCase],
    responses: dict[str, list[SearchResult] | Exception],
) -> list[CaseEvaluation]:
    service = cast(RetrievalService, FakeSearchService(responses))
    return evaluate(service, cases).cases


def test_single_relevant_chunk_has_full_recall_and_mrr() -> None:
    case = EvaluationCase(
        case_id="single",
        query="single",
        relevant_chunk_ids=["chunk-a"],
    )

    outcome = run_cases([case], {"single": [result(1, "chunk-a")]})[0]

    assert outcome.recall_at_1 == 1.0
    assert outcome.recall_at_3 == 1.0
    assert outcome.recall_at_5 == 1.0
    assert outcome.reciprocal_rank == 1.0


def test_fractional_recall_uses_distinct_targets_and_cutoffs() -> None:
    case = EvaluationCase(
        case_id="fractional",
        query="fractional",
        relevant_chunk_ids=["a", "b", "c", "d", "a"],
    )
    ranked = [
        result(1, "a"),
        result(2, "irrelevant"),
        result(3, "b"),
        result(4, "a"),
    ]

    outcome = run_cases([case], {"fractional": ranked})[0]

    assert outcome.recall_at_1 == 0.25
    assert outcome.recall_at_3 == 0.5
    assert outcome.recall_at_5 == 0.5
    assert outcome.reciprocal_rank == 1.0


def test_page_ranges_use_overlap_and_deduplicate_identical_references() -> None:
    document_id = "doc_" + "b" * 64
    other_document = "doc_" + "c" * 64
    case = EvaluationCase(
        case_id="pages",
        query="pages",
        relevant_references=[
            RelevantPageReference(
                document_id=document_id,
                page_start=2,
                page_end=2,
            ),
            RelevantPageReference(
                document_id=document_id,
                page_start=4,
                page_end=6,
            ),
            RelevantPageReference(
                document_id=document_id,
                page_start=8,
                page_end=8,
            ),
            RelevantPageReference(
                document_id=other_document,
                page_start=2,
                page_end=2,
            ),
            RelevantPageReference(
                document_id=document_id,
                page_start=2,
                page_end=2,
            ),
        ],
    )
    ranked = [
        result(1, "exact", document_id=document_id, page_start=2, page_end=2),
        result(2, "partial", document_id=document_id, page_start=5, page_end=7),
        result(3, "adjacent", document_id=document_id, page_start=7, page_end=7),
        result(4, "wrong", document_id=other_document, page_start=9, page_end=9),
    ]

    outcome = run_cases([case], {"pages": ranked})[0]

    assert outcome.recall_at_1 == 0.25
    assert outcome.recall_at_3 == 0.5
    assert outcome.recall_at_5 == 0.5


def test_mixed_targets_fewer_results_duplicates_and_no_match() -> None:
    document_id = "doc_" + "d" * 64
    mixed = EvaluationCase(
        case_id="mixed",
        query="mixed",
        relevant_chunk_ids=["chunk-a", "chunk-b"],
        relevant_references=[
            RelevantPageReference(
                document_id=document_id,
                page_start=10,
                page_end=12,
            )
        ],
    )
    none = EvaluationCase(
        case_id="none",
        query="none",
        relevant_chunk_ids=["missing"],
    )
    outcomes = run_cases(
        [mixed, none],
        {
            "mixed": [
                result(1, "chunk-a"),
                result(
                    2,
                    "chunk-a",
                    document_id=document_id,
                    page_start=11,
                    page_end=11,
                ),
            ],
            "none": [result(1, "irrelevant")],
        },
    )

    mixed_outcome, no_match = outcomes
    assert mixed_outcome.recall_at_1 == pytest.approx(1 / 3)
    assert mixed_outcome.recall_at_3 == pytest.approx(2 / 3)
    assert mixed_outcome.recall_at_5 == pytest.approx(2 / 3)
    assert no_match.recall_at_1 == 0.0
    assert no_match.recall_at_3 == 0.0
    assert no_match.recall_at_5 == 0.0
    assert no_match.reciprocal_rank == 0.0


def test_search_failure_is_zero_recall_with_failure_category() -> None:
    case = EvaluationCase(
        case_id="failure",
        query="failure",
        relevant_chunk_ids=["chunk-a"],
    )

    outcome = run_cases([case], {"failure": IndexNotReadyError()})[0]

    assert outcome.recall_at_1 == 0.0
    assert outcome.recall_at_3 == 0.0
    assert outcome.recall_at_5 == 0.0
    assert outcome.reciprocal_rank == 0.0
    assert outcome.failure_category == "index_not_ready"


def test_aggregate_metrics_are_means_of_fractional_case_recalls() -> None:
    report = summarize(
        [
            CaseEvaluation(
                case_id="a",
                reciprocal_rank=1.0,
                recall_at_1=0.25,
                recall_at_3=0.5,
                recall_at_5=0.75,
                latency_seconds=0.1,
                failure_category=None,
                results=[result(1, "a")],
            ),
            CaseEvaluation(
                case_id="b",
                reciprocal_rank=0.0,
                recall_at_1=0.0,
                recall_at_3=0.25,
                recall_at_5=0.5,
                latency_seconds=0.3,
                failure_category=None,
                results=[],
            ),
        ]
    )

    assert report.recall_at_1 == 0.125
    assert report.recall_at_3 == 0.375
    assert report.recall_at_5 == 0.625
    assert report.mean_reciprocal_rank == 0.5
    assert report.no_result_rate == 0.5
    assert report.average_query_latency_seconds == 0.2
