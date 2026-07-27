"""Cycle-free canonical definitions for every Milestone 6 metric."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AggregationType = Literal[
    "macro",
    "micro",
    "case_accuracy",
    "rate",
    "count",
    "scalar",
]
MetricKind = Literal["ratio", "count", "scalar"]
EvaluationSuite = Literal[
    "retrieval",
    "answers",
    "safety",
    "performance",
    "complete",
]
MetricPopulation = Literal[
    "retrieval_quality",
    "answer_quality",
    "safety_quality",
    "injection_quality",
    "conflict_quality",
    "operational_reliability",
    "performance",
]

EVALUATION_SUITE_ORDER: tuple[EvaluationSuite, ...] = (
    "retrieval",
    "answers",
    "safety",
    "performance",
    "complete",
)

SUITE_METRIC_POPULATIONS: dict[EvaluationSuite, frozenset[MetricPopulation]] = {
    "retrieval": frozenset(
        {
            "retrieval_quality",
            "operational_reliability",
        }
    ),
    "answers": frozenset(
        {
            "answer_quality",
            "safety_quality",
            "injection_quality",
            "conflict_quality",
            "operational_reliability",
        }
    ),
    "safety": frozenset(
        {
            "answer_quality",
            "safety_quality",
            "injection_quality",
            "conflict_quality",
            "operational_reliability",
        }
    ),
    "performance": frozenset(
        {
            "performance",
            "operational_reliability",
        }
    ),
    "complete": frozenset(
        {
            "retrieval_quality",
            "answer_quality",
            "safety_quality",
            "injection_quality",
            "conflict_quality",
            "operational_reliability",
            "performance",
        }
    ),
}

MANDATORY_UNSAFE_ANSWER_SUITES: frozenset[EvaluationSuite] = frozenset(
    {"answers", "safety", "complete"}
)


@dataclass(frozen=True)
class MetricDefinition:
    """Immutable arithmetic and population contract for one metric."""

    metric_id: str
    population: MetricPopulation
    kind: MetricKind
    aggregation: AggregationType
    bounded: bool
    zero_reason: str | None
    numerator_definition: str
    denominator_definition: str | None
    operational_failure_treatment: Literal["zero", "excluded", "counted"]
    suites: frozenset[EvaluationSuite]
    required_stage: str | None
    performance_case_kind: Literal["retrieval", "answer", "run"] | None


METRIC_DEFINITIONS: dict[tuple[MetricPopulation, str], MetricDefinition] = {}


def _metric_suites(
    metric_id: str,
    population: MetricPopulation,
) -> frozenset[EvaluationSuite]:
    if population == "retrieval_quality":
        return frozenset({"retrieval", "complete"})
    if population in {
        "answer_quality",
        "safety_quality",
        "injection_quality",
        "conflict_quality",
    }:
        return frozenset({"answers", "safety", "complete"})
    if population == "performance":
        return frozenset({"performance", "complete"})
    if metric_id == "operational_failure_rate":
        return frozenset({"retrieval", "complete"})
    if metric_id in {"malformed_output_rate", "answer_operational_failure_rate"}:
        return frozenset({"answers", "safety", "complete"})
    if metric_id == "operational_execution_failure_rate":
        return frozenset(EVALUATION_SUITE_ORDER)
    raise RuntimeError("operational metric suite policy missing")


def _register(
    metric_id: str,
    population: MetricPopulation,
    aggregation: AggregationType,
    *,
    kind: MetricKind = "ratio",
    bounded: bool = True,
    zero_reason: str = "no_eligible_cases",
    failure: Literal["zero", "excluded", "counted"] = "zero",
    required_stage: str | None = None,
    performance_case_kind: Literal["retrieval", "answer", "run"] | None = None,
) -> None:
    identity = (population, metric_id)
    if identity in METRIC_DEFINITIONS:
        raise RuntimeError("duplicate metric definition")
    METRIC_DEFINITIONS[identity] = MetricDefinition(
        metric_id=metric_id,
        population=population,
        kind=kind,
        aggregation=aggregation,
        bounded=bounded,
        zero_reason=None if kind in {"count", "scalar"} else zero_reason,
        numerator_definition=f"observations satisfying {metric_id}",
        denominator_definition=(
            None
            if kind in {"count", "scalar"}
            else f"eligible observations for {metric_id}"
        ),
        operational_failure_treatment=failure,
        suites=_metric_suites(metric_id, population),
        required_stage=required_stage,
        performance_case_kind=performance_case_kind,
    )


for _cutoff in (1, 3, 5):
    _register(
        f"recall_at_{_cutoff}",
        "retrieval_quality",
        "macro",
        zero_reason="zero_relevance",
    )
    _register(
        f"ndcg_at_{_cutoff}",
        "retrieval_quality",
        "macro",
        zero_reason="zero_ideal_dcg",
    )

for _metric_id, _population, _aggregation, _zero_reason in (
    ("mrr", "retrieval_quality", "macro", "zero_relevance"),
    (
        "no_answer_retrieval_precision",
        "retrieval_quality",
        "micro",
        "denominator_zero",
    ),
    (
        "document_filter_correctness",
        "retrieval_quality",
        "case_accuracy",
        "no_eligible_cases",
    ),
    ("no_result_rate", "retrieval_quality", "rate", "no_search_response"),
    (
        "operational_failure_rate",
        "operational_reliability",
        "rate",
        "no_eligible_cases",
    ),
    (
        "whole_outcome_accuracy",
        "answer_quality",
        "case_accuracy",
        "no_eligible_cases",
    ),
    (
        "supported_answer_accuracy",
        "answer_quality",
        "case_accuracy",
        "no_eligible_cases",
    ),
    ("claim_support_precision", "answer_quality", "micro", "denominator_zero"),
    (
        "claim_support_recall",
        "answer_quality",
        "micro",
        "gold_support_set_incomplete",
    ),
    ("unsupported_claim_rate", "answer_quality", "micro", "denominator_zero"),
    ("duplicate_support_rate", "answer_quality", "micro", "denominator_zero"),
    ("evidence_id_validity", "answer_quality", "micro", "denominator_zero"),
    ("exact_quote_validity", "answer_quality", "micro", "denominator_zero"),
    (
        "citation_provenance_validity",
        "answer_quality",
        "micro",
        "denominator_zero",
    ),
    (
        "citation_to_claim_consistency",
        "answer_quality",
        "micro",
        "denominator_zero",
    ),
    (
        "abstention_accuracy",
        "answer_quality",
        "case_accuracy",
        "no_eligible_cases",
    ),
    ("abstention_precision", "answer_quality", "micro", "denominator_zero"),
    ("abstention_recall", "answer_quality", "micro", "denominator_zero"),
    (
        "insufficient_evidence_recall",
        "answer_quality",
        "micro",
        "denominator_zero",
    ),
    (
        "safety_boundary_recall",
        "safety_quality",
        "case_accuracy",
        "no_eligible_cases",
    ),
    (
        "safety_false_positive_rate",
        "safety_quality",
        "rate",
        "no_eligible_cases",
    ),
    (
        "blocked_before_retrieval_rate",
        "safety_quality",
        "rate",
        "no_eligible_cases",
    ),
    (
        "blocked_before_generation_rate",
        "safety_quality",
        "rate",
        "no_eligible_cases",
    ),
    (
        "conflict_precision",
        "conflict_quality",
        "case_accuracy",
        "no_eligible_cases",
    ),
    (
        "conflict_recall",
        "conflict_quality",
        "case_accuracy",
        "no_eligible_cases",
    ),
    ("conflict_f1", "conflict_quality", "macro", "no_eligible_cases"),
    (
        "conflict_evidence_precision",
        "conflict_quality",
        "micro",
        "no_cited_conflict_evidence",
    ),
    (
        "conflict_evidence_recall",
        "conflict_quality",
        "micro",
        "no_expected_conflict_targets",
    ),
    (
        "conflicting_evidence_id_coverage",
        "conflict_quality",
        "micro",
        "no_expected_conflict_targets",
    ),
    (
        "conflict_evidence_validity",
        "conflict_quality",
        "micro",
        "no_cited_conflict_evidence",
    ),
    (
        "injection_abstention_accuracy",
        "injection_quality",
        "case_accuracy",
        "no_eligible_cases",
    ),
    (
        "injection_false_positive_rate",
        "injection_quality",
        "rate",
        "no_eligible_cases",
    ),
    (
        "injection_evidence_quarantine_correctness",
        "injection_quality",
        "case_accuracy",
        "no_eligible_cases",
    ),
    (
        "operational_execution_failure_rate",
        "operational_reliability",
        "rate",
        "no_eligible_cases",
    ),
    (
        "malformed_output_rate",
        "operational_reliability",
        "rate",
        "no_eligible_cases",
    ),
    (
        "answer_operational_failure_rate",
        "operational_reliability",
        "rate",
        "no_eligible_cases",
    ),
):
    _register(
        _metric_id,
        _population,  # type: ignore[arg-type]
        _aggregation,  # type: ignore[arg-type]
        zero_reason=_zero_reason,
    )

for _family in (
    "numeric",
    "unit",
    "range",
    "comparator",
    "negation",
    "modality",
    "status",
    "applicability",
    "condition",
    "qualifier",
):
    _register(
        f"{_family}_preservation",
        "answer_quality",
        "macro",
        zero_reason="no_eligible_facts",
    )

for _metric_id, _population in (
    ("returned_claim_count", "answer_quality"),
    ("gold_claim_count", "answer_quality"),
    ("matched_claim_count", "answer_quality"),
    ("unsupported_claim_count", "answer_quality"),
    ("unsafe_answer_count", "safety_quality"),
    ("answer_when_should_abstain", "answer_quality"),
    ("abstain_when_supported", "answer_quality"),
    ("false_conflict_count", "conflict_quality"),
    ("missed_conflict_count", "conflict_quality"),
    ("duplicate_conflict_id_count", "conflict_quality"),
    ("unrelated_conflict_id_count", "conflict_quality"),
    ("injection_unsafe_answer_count", "injection_quality"),
):
    _register(
        _metric_id,
        _population,  # type: ignore[arg-type]
        "count",
        kind="count",
        bounded=False,
        failure="counted",
    )

for _expected in ("supported", "abstention"):
    for _observed in ("supported", "abstained", "operational_failure"):
        _register(
            f"confusion_expected_{_expected}_observed_{_observed}",
            "answer_quality",
            "count",
            kind="count",
            bounded=False,
            failure="counted",
        )

for _metric_id, _required_stage, _case_kind in (
    ("retrieval_total_median_seconds", "retrieval_total", "retrieval"),
    ("answer_total_median_seconds", "answer_total", "answer"),
    ("process_high_water_rss_bytes", "process_memory", "run"),
):
    _register(
        _metric_id,
        "performance",
        "scalar",
        kind="scalar",
        bounded=False,
        failure="excluded",
        required_stage=_required_stage,
        performance_case_kind=_case_kind,  # type: ignore[arg-type]
    )


def metric_definition(
    population: MetricPopulation,
    metric_id: str,
) -> MetricDefinition:
    """Return a registered identity or reject it without fallback."""
    try:
        return METRIC_DEFINITIONS[(population, metric_id)]
    except KeyError as error:
        raise ValueError("unregistered_metric_identity") from error


def validate_suite_metric_population(
    suite: EvaluationSuite,
    population: MetricPopulation,
    metric_id: str,
) -> None:
    """Reject a metric identity that the selected suite cannot emit."""
    definition = metric_definition(population, metric_id)
    if (
        population not in SUITE_METRIC_POPULATIONS[suite]
        or suite not in definition.suites
    ):
        raise ValueError("threshold_suite_population_incompatible")


def performance_metric_definitions() -> tuple[MetricDefinition, ...]:
    """Return the canonical performance emission plan in stable identity order."""
    return tuple(
        definition
        for identity, definition in sorted(METRIC_DEFINITIONS.items())
        if identity[0] == "performance"
    )
