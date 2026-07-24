"""Deterministic retrieval metric calculations."""

from app.evaluation.models import CaseEvaluation, EvaluationReport


def summarize(cases: list[CaseEvaluation]) -> EvaluationReport:
    """Aggregate inspectable per-case retrieval outcomes."""
    count = len(cases)
    if count == 0:
        return EvaluationReport(
            case_count=0,
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            mean_reciprocal_rank=0.0,
            no_result_rate=0.0,
            average_query_latency_seconds=0.0,
            cases=[],
        )
    return EvaluationReport(
        case_count=count,
        recall_at_1=sum(case.recall_at_1 for case in cases) / count,
        recall_at_3=sum(case.recall_at_3 for case in cases) / count,
        recall_at_5=sum(case.recall_at_5 for case in cases) / count,
        mean_reciprocal_rank=sum(case.reciprocal_rank for case in cases) / count,
        no_result_rate=sum(not case.results for case in cases) / count,
        average_query_latency_seconds=(
            sum(case.latency_seconds for case in cases) / count
        ),
        cases=cases,
    )
