"""Command-line deterministic retrieval evaluation."""

import argparse
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from app.core.config import Settings
from app.evaluation.metrics import summarize
from app.evaluation.models import CaseEvaluation, EvaluationCase, EvaluationReport
from app.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from app.retrieval.errors import RetrievalError
from app.retrieval.models import SearchRequest, SearchResult
from app.retrieval.service import RetrievalService


def evaluate(
    service: RetrievalService,
    cases: list[EvaluationCase],
) -> EvaluationReport:
    """Evaluate ranked evidence without LLM judging."""
    outcomes: list[CaseEvaluation] = []
    for case in sorted(cases, key=lambda item: item.case_id):
        started = time.perf_counter()
        try:
            response = service.search(SearchRequest(query=case.query, top_k=5))
        except RetrievalError as error:
            outcomes.append(
                CaseEvaluation(
                    case_id=case.case_id,
                    reciprocal_rank=0.0,
                    recall_at_1=0.0,
                    recall_at_3=0.0,
                    recall_at_5=0.0,
                    latency_seconds=time.perf_counter() - started,
                    failure_category=error.code,
                    results=[],
                )
            )
            continue
        latency = time.perf_counter() - started
        relevant_ranks = [
            result.rank for result in response.results if _is_relevant(case, result)
        ]
        first_rank = min(relevant_ranks, default=0)
        recalls = {
            cutoff: _recall_at(case, response.results, cutoff) for cutoff in (1, 3, 5)
        }
        outcomes.append(
            CaseEvaluation(
                case_id=case.case_id,
                reciprocal_rank=0.0 if first_rank == 0 else 1.0 / first_rank,
                recall_at_1=recalls[1],
                recall_at_3=recalls[3],
                recall_at_5=recalls[5],
                latency_seconds=latency,
                failure_category=None if response.results else "no_results",
                results=response.results,
            )
        )
    return summarize(outcomes)


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load inspectable JSONL cases."""
    cases: list[EvaluationCase] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    cases.append(EvaluationCase.model_validate_json(line))
    except (OSError, UnicodeError, ValidationError, ValueError) as error:
        raise SystemExit("Evaluation cases are invalid.") from error
    return cases


def write_report(path: Path, report: EvaluationReport) -> None:
    """Atomically write a local evaluation report."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/runtime/indexes/evaluations/latest.json"),
    )
    arguments = parser.parse_args()
    settings = Settings()
    provider = SentenceTransformerEmbeddingProvider(settings)
    report = evaluate(
        RetrievalService(settings, provider),
        load_cases(arguments.cases),
    )
    write_report(arguments.output, report)


def _is_relevant(case: EvaluationCase, result: SearchResult) -> bool:
    if result.chunk_id in case.relevant_chunk_ids:
        return True
    return any(
        reference.document_id == result.document_id
        and reference.page_start <= result.page_end
        and reference.page_end >= result.page_start
        for reference in case.relevant_references
    )


def _recall_at(
    case: EvaluationCase,
    results: list[SearchResult],
    cutoff: int,
) -> float:
    chunk_targets = set(case.relevant_chunk_ids)
    reference_targets = {
        (reference.document_id, reference.page_start, reference.page_end)
        for reference in case.relevant_references
    }
    matched_chunks: set[str] = set()
    matched_references: set[tuple[str, int, int]] = set()
    for result in results[:cutoff]:
        if result.chunk_id in chunk_targets:
            matched_chunks.add(result.chunk_id)
        for target in reference_targets:
            document_id, page_start, page_end = target
            if (
                document_id == result.document_id
                and page_start <= result.page_end
                and page_end >= result.page_start
            ):
                matched_references.add(target)
    target_count = len(chunk_targets) + len(reference_targets)
    return (len(matched_chunks) + len(matched_references)) / target_count


if __name__ == "__main__":
    main()
