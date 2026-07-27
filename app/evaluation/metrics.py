"""Strict metric records plus the legacy retrieval report summarizer."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, model_validator

from app.evaluation.metric_definitions import (
    AggregationType,
    MetricPopulation,
    metric_definition,
)
from app.evaluation.models import CaseEvaluation, EvaluationReport


class MetricRecord(BaseModel):
    """A registry-bound, arithmetically self-consistent metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_id: str
    population: MetricPopulation
    value: float | int | None
    numerator: float | int
    denominator: int | None
    aggregation: AggregationType
    eligible: bool
    exclusion_reason: str | None = None
    eligible_case_count: int
    total_population_case_count: int

    @model_validator(mode="after")
    def validate_consistency(self) -> MetricRecord:
        definition = metric_definition(self.population, self.metric_id)
        if self.aggregation != definition.aggregation:
            raise ValueError("metric_aggregation_mismatch")
        values = (self.numerator, self.value)
        if any(
            value is not None and not math.isfinite(float(value)) for value in values
        ):
            raise ValueError("metric values must be finite")
        if self.numerator < 0:
            raise ValueError("metric numerator cannot be negative")
        if self.denominator is not None and self.denominator < 0:
            raise ValueError("metric denominator cannot be negative")
        if (
            self.eligible_case_count < 0
            or self.total_population_case_count < 0
            or self.eligible_case_count > self.total_population_case_count
        ):
            raise ValueError("metric case counts are inconsistent")
        if definition.kind == "count":
            if (
                self.aggregation != "count"
                or self.denominator is not None
                or not self.eligible
                or type(self.value) is not int
                or type(self.numerator) is not int
                or self.value != self.numerator
                or self.exclusion_reason is not None
            ):
                raise ValueError("count metric is inconsistent")
            return self
        if definition.kind == "scalar":
            if self.aggregation != "scalar" or self.denominator is not None:
                raise ValueError("scalar metric is inconsistent")
            if self.eligible:
                if (
                    self.value is None
                    or float(self.value) < 0
                    or float(self.numerator) != float(self.value)
                    or self.exclusion_reason is not None
                ):
                    raise ValueError("scalar metric is inconsistent")
            elif (
                self.value is not None
                or self.numerator != 0
                or self.exclusion_reason
                not in {
                    "no_eligible_performance_case",
                    "no_successful_performance_measurements",
                    "memory_observation_unavailable",
                }
            ):
                raise ValueError("ineligible scalar metric is inconsistent")
            return self
        if self.aggregation == "count":
            raise ValueError("ratio metric cannot use count aggregation")
        if self.eligible:
            if (
                self.denominator is None
                or self.denominator <= 0
                or self.value is None
                or self.exclusion_reason is not None
            ):
                raise ValueError("eligible ratio metric requires a denominator")
            expected = float(self.numerator) / self.denominator
            if not math.isclose(
                float(self.value), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("metric value contradicts numerator and denominator")
        else:
            if self.value is not None or self.exclusion_reason is None:
                raise ValueError("ineligible metric requires a null value and reason")
            if (
                self.denominator == 0
                and definition.zero_reason is not None
                and self.exclusion_reason
                not in {definition.zero_reason, "denominator_zero"}
            ):
                raise ValueError("metric null reason is not registered")
        if definition.bounded:
            if self.numerator > (self.denominator or 0):
                raise ValueError("bounded metric numerator exceeds denominator")
            if self.value is not None and not 0.0 <= float(self.value) <= 1.0:
                raise ValueError("bounded metric must be between zero and one")
        return self


def metric_record(
    metric_id: str,
    numerator: float,
    denominator: int,
    aggregation: AggregationType,
    *,
    population: MetricPopulation = "answer_quality",
    exclusion_reason: str | None = None,
    eligible_case_count: int | None = None,
    total_population_case_count: int | None = None,
) -> MetricRecord:
    """Create a registered ratio record; zero denominators are null."""
    definition = metric_definition(population, metric_id)
    if definition.kind != "ratio" or definition.aggregation != aggregation:
        raise ValueError("metric definition mismatch")
    if denominator < 0:
        raise ValueError("denominator cannot be negative")
    eligible_count = denominator if eligible_case_count is None else eligible_case_count
    total_count = (
        max(eligible_count, denominator)
        if total_population_case_count is None
        else total_population_case_count
    )
    if denominator == 0:
        reason = exclusion_reason or definition.zero_reason or "denominator_zero"
        return MetricRecord(
            metric_id=metric_id,
            population=population,
            value=None,
            numerator=numerator,
            denominator=0,
            aggregation=aggregation,
            eligible=False,
            exclusion_reason=reason,
            eligible_case_count=eligible_count,
            total_population_case_count=total_count,
        )
    return MetricRecord(
        metric_id=metric_id,
        population=population,
        value=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
        aggregation=aggregation,
        eligible=True,
        eligible_case_count=eligible_count,
        total_population_case_count=total_count,
    )


def count_record(
    metric_id: str,
    count: int,
    *,
    population: MetricPopulation = "answer_quality",
    eligible_case_count: int = 0,
    total_population_case_count: int = 0,
) -> MetricRecord:
    """Represent a registered non-negative exact integer count."""
    definition = metric_definition(population, metric_id)
    if definition.kind != "count":
        raise ValueError("metric definition mismatch")
    if count < 0:
        raise ValueError("count cannot be negative")
    return MetricRecord(
        metric_id=metric_id,
        population=population,
        value=count,
        numerator=count,
        denominator=None,
        aggregation="count",
        eligible=True,
        eligible_case_count=eligible_case_count,
        total_population_case_count=total_population_case_count,
    )


def scalar_record(
    metric_id: str,
    value: float | int,
    *,
    population: MetricPopulation = "performance",
    eligible_case_count: int = 0,
    total_population_case_count: int = 0,
) -> MetricRecord:
    """Represent a registered finite non-negative scalar observation."""
    definition = metric_definition(population, metric_id)
    if definition.kind != "scalar" or not math.isfinite(float(value)):
        raise ValueError("metric definition mismatch")
    if float(value) < 0:
        raise ValueError("scalar cannot be negative")
    return MetricRecord(
        metric_id=metric_id,
        population=population,
        value=value,
        numerator=value,
        denominator=None,
        aggregation="scalar",
        eligible=True,
        eligible_case_count=eligible_case_count,
        total_population_case_count=total_population_case_count,
    )


def ineligible_scalar_record(
    metric_id: str,
    reason: str,
    *,
    population: MetricPopulation = "performance",
    total_population_case_count: int = 0,
) -> MetricRecord:
    """Represent a registered scalar that had no usable observation."""
    definition = metric_definition(population, metric_id)
    if definition.kind != "scalar" or reason not in {
        "no_eligible_performance_case",
        "no_successful_performance_measurements",
        "memory_observation_unavailable",
    }:
        raise ValueError("metric definition mismatch")
    return MetricRecord(
        metric_id=metric_id,
        population=population,
        value=None,
        numerator=0,
        denominator=None,
        aggregation="scalar",
        eligible=False,
        exclusion_reason=reason,
        eligible_case_count=0,
        total_population_case_count=total_population_case_count,
    )


def validate_unique_metrics(metrics: list[MetricRecord]) -> None:
    """Reject duplicate exact metric identities."""
    identities = [(item.population, item.metric_id) for item in metrics]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate_metric_identity")


def summarize(cases: list[CaseEvaluation]) -> EvaluationReport:
    """Aggregate the compatible Milestone 3 retrieval evaluator."""
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
