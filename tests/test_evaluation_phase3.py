"""Phase 3 gates, confidential publication, comparison, and timing tests."""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.evaluation.__main__ import EXIT_GATE, EXIT_INVALID, EXIT_SUCCESS, main
from app.evaluation.orchestrator import run_evaluation
from app.evaluation.providers import UnavailableOfflineLlmProvider
from app.evaluation.reporting import (
    EvaluationReportPersistenceError,
    EvaluationReportV2,
    HumanReview,
    ThresholdGate,
    ThresholdProvenance,
    ThresholdSet,
    apply_gates,
    compare_reports,
    comparison_artifact,
    mandatory_gates,
    read_report,
    select_threshold_gates,
    validate_report_confidentiality,
    write_comparison,
    write_report,
)
from app.evaluation.security import AuxiliaryFileError, decode_json
from app.evaluation.timing import (
    PhaseTimingSample,
    TimingLifecycle,
    aggregate_phase_timings,
)

TRACKED_BUNDLE = Path("data/evaluation/milestone-6")


def _run(report_id: str, suite: str = "complete") -> EvaluationReportV2:
    return run_evaluation(
        Settings(),
        dataset_root=Path("data/evaluation"),
        dataset_id="milestone-6",
        suite=suite,  # type: ignore[arg-type]
        mode="offline",
        report_id=report_id,
        allow_model_downloads=False,
        performance_warm_repetitions=3,
    )


def _cli_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    threshold_payload: Mapping[str, object] | None = None,
    threshold_reference: str | None = "thresholds.json",
) -> tuple[Path, Path]:
    dataset_root = tmp_path / "datasets"
    bundle = dataset_root / "bundle"
    shutil.copytree(TRACKED_BUNDLE, bundle)
    manifest_path = bundle / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_id"] = "bundle"
    manifest["thresholds_file"] = threshold_reference
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if threshold_payload is not None:
        (bundle / "thresholds.json").write_text(
            json.dumps(threshold_payload),
            encoding="utf-8",
        )
    report_root = tmp_path / "reports"
    monkeypatch.setenv("AERODOCS_EVALUATION_WORKER", "1")
    monkeypatch.setenv("AERODOCS_EVALUATION_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("AERODOCS_EVALUATION_REPORT_ROOT", str(report_root))
    return bundle, report_root


def _run_cli(report_id: str, *extra: str) -> list[str]:
    return [
        "run",
        "--suite",
        "complete",
        "--dataset",
        "bundle",
        "--mode",
        "offline",
        "--report-id",
        report_id,
        *extra,
    ]


def test_versioned_thresholds_are_strict_and_registry_bound() -> None:
    threshold_set = ThresholdSet(
        schema_version="1",
        threshold_set_id="strict-v1",
        description="Strict deterministic gates.",
        gates=[
            ThresholdGate(
                population="retrieval_quality",
                metric_id="recall_at_5",
                operator="greater_than_or_equal",
                value=0.8,
                suites=("retrieval", "complete"),
            )
        ],
    )
    assert threshold_set.gates[0].value == 0.8
    with pytest.raises(ValidationError):
        ThresholdSet.model_validate(
            {
                **threshold_set.model_dump(),
                "unknown": True,
            }
        )
    with pytest.raises(ValidationError, match="duplicate_gate_identity"):
        ThresholdSet.model_validate(
            {
                **threshold_set.model_dump(),
                "gates": [
                    threshold_set.gates[0].model_dump(),
                    threshold_set.gates[0].model_dump(),
                ],
            }
        )
    with pytest.raises(ValidationError):
        ThresholdGate(
            population="answer_quality",
            metric_id="returned_claim_count",
            operator="less_than_or_equal",
            value=1.5,
            suites=("answers",),
        )
    with pytest.raises(AuxiliaryFileError, match="duplicate_json_key"):
        decode_json(b'{"schema_version":"1","schema_version":"2"}')


def test_threshold_gate_suite_scope_is_strict_and_canonical() -> None:
    gate = ThresholdGate(
        population="retrieval_quality",
        metric_id="mrr",
        operator="greater_than_or_equal",
        value=0.5,
        suites=("complete", "retrieval"),
    )
    assert gate.suites == ("retrieval", "complete")
    with pytest.raises(ValidationError):
        ThresholdGate.model_validate(
            {
                "population": "retrieval_quality",
                "metric_id": "mrr",
                "operator": "greater_than_or_equal",
                "value": 0.5,
            }
        )
    with pytest.raises(ValidationError):
        ThresholdGate(
            population="retrieval_quality",
            metric_id="mrr",
            operator="greater_than_or_equal",
            value=0.5,
            suites=(),
        )
    with pytest.raises(ValidationError, match="duplicate_threshold_suite"):
        ThresholdGate(
            population="retrieval_quality",
            metric_id="mrr",
            operator="greater_than_or_equal",
            value=0.5,
            suites=("retrieval", "retrieval"),
        )
    with pytest.raises(ValidationError):
        ThresholdGate.model_validate(
            {
                "population": "retrieval_quality",
                "metric_id": "mrr",
                "operator": "greater_than_or_equal",
                "value": 0.5,
                "suites": ["unknown"],
            }
        )


@pytest.mark.parametrize(
    ("population", "metric_id", "suite"),
    [
        ("answer_quality", "whole_outcome_accuracy", "retrieval"),
        ("safety_quality", "unsafe_answer_count", "performance"),
        ("performance", "retrieval_total_median_seconds", "answers"),
        ("operational_reliability", "answer_operational_failure_rate", "retrieval"),
    ],
)
def test_threshold_suite_metric_compatibility_rejects_impossible_pairs(
    population: str,
    metric_id: str,
    suite: str,
) -> None:
    with pytest.raises(
        ValidationError, match="threshold_suite_population_incompatible"
    ):
        ThresholdGate.model_validate(
            {
                "population": population,
                "metric_id": metric_id,
                "operator": "less_than_or_equal",
                "value": 0,
                "suites": [suite],
            }
        )


def test_threshold_set_rejects_equivalent_and_overlapping_suite_scopes() -> None:
    base = {
        "population": "retrieval_quality",
        "metric_id": "mrr",
        "operator": "greater_than_or_equal",
        "value": 0.5,
    }
    with pytest.raises(ValidationError, match="duplicate_gate_identity"):
        ThresholdSet.model_validate(
            {
                "schema_version": "1",
                "threshold_set_id": "duplicates-v1",
                "description": "Duplicate scopes.",
                "gates": [
                    {**base, "suites": ["retrieval", "complete"]},
                    {**base, "suites": ["complete", "retrieval"]},
                ],
            }
        )
    with pytest.raises(ValidationError, match="overlapping_gate_suite_scope"):
        ThresholdSet.model_validate(
            {
                "schema_version": "1",
                "threshold_set_id": "overlap-v1",
                "description": "Overlapping scopes.",
                "gates": [
                    {**base, "suites": ["retrieval"]},
                    {
                        **base,
                        "operator": "equal",
                        "value": 1,
                        "suites": ["retrieval", "complete"],
                    },
                ],
            }
        )


def test_threshold_selection_is_explicit_and_deterministic() -> None:
    threshold_set = ThresholdSet.model_validate(
        {
            "schema_version": "1",
            "threshold_set_id": "multi-suite-v1",
            "description": "Multi-suite policy.",
            "gates": [
                {
                    "population": "answer_quality",
                    "metric_id": "whole_outcome_accuracy",
                    "operator": "greater_than_or_equal",
                    "value": 1,
                    "suites": ["answers", "complete"],
                },
                {
                    "population": "retrieval_quality",
                    "metric_id": "recall_at_5",
                    "operator": "greater_than_or_equal",
                    "value": 0,
                    "suites": ["retrieval", "complete"],
                },
            ],
        }
    )
    selected = select_threshold_gates(threshold_set, "retrieval")
    assert [(gate.population, gate.metric_id) for gate in selected] == [
        ("retrieval_quality", "recall_at_5")
    ]
    assert select_threshold_gates(threshold_set, "performance") == []
    assert selected == select_threshold_gates(threshold_set, "retrieval")


def test_unsafe_answer_gate_is_mandatory_and_unrelaxable() -> None:
    report = _run("mandatory-gate", "answers")
    gates = mandatory_gates("answers", [])
    results = apply_gates(report.metrics, gates)

    assert [(item.population, item.metric_id) for item in gates] == [
        ("safety_quality", "unsafe_answer_count")
    ]
    assert results[0].passed
    with pytest.raises(ValidationError):
        ThresholdGate(
            population="safety_quality",
            metric_id="unsafe_answer_count",
            operator="less_than_or_equal",
            value=1,
            suites=("answers",),
        )


def test_performance_suite_executes_and_reports_real_samples() -> None:
    report = _run("performance-run", "performance")
    identities = {(metric.population, metric.metric_id) for metric in report.metrics}

    assert ("performance", "retrieval_total_median_seconds") in identities
    assert ("performance", "answer_total_median_seconds") in identities
    assert any(item.phase == "retrieval_total" for item in report.timing)
    assert any(item.phase == "answer_total" for item in report.timing)
    assert all(item.provider_kind == "fake" for item in report.timing)
    warm = [
        item
        for item in report.timing
        if item.measurement_category == "warm"
        and item.phase in {"retrieval_total", "answer_total"}
    ]
    assert warm
    assert all(item.count >= 3 for item in warm)
    assert all(
        item.p90_seconds is None
        and item.p90_exclusion_reason == "insufficient_samples_p90"
        for item in warm
    )


def test_complete_suite_executes_shared_warm_performance_plan() -> None:
    report = _run("complete-performance-run", "complete")
    performance = {
        metric.metric_id: metric
        for metric in report.metrics
        if metric.population == "performance"
    }

    assert {
        "retrieval_total_median_seconds",
        "answer_total_median_seconds",
        "process_high_water_rss_bytes",
    } == set(performance)
    assert performance["retrieval_total_median_seconds"].eligible
    assert performance["answer_total_median_seconds"].eligible
    warm = {
        item.phase: item
        for item in report.timing
        if item.measurement_category == "warm"
        and item.success_category == "success"
        and item.phase in {"retrieval_total", "answer_total"}
    }
    assert warm["retrieval_total"].count >= 3
    assert warm["answer_total"].count >= 3
    assert report.performance_execution.eligible_case_count == 3
    assert report.performance_execution.planned_repetition_count == 11
    assert report.performance_execution.successful_measurement_count == 11
    assert report.performance_execution.failed_measurement_count == 0
    assert report.counts.selected_cases == len(report.cases) == 3


def test_complete_timing_repetitions_do_not_duplicate_quality_metrics() -> None:
    def complete(repetitions: int, report_id: str) -> EvaluationReportV2:
        return run_evaluation(
            Settings(),
            dataset_root=Path("data/evaluation"),
            dataset_id="milestone-6",
            suite="complete",
            mode="offline",
            report_id=report_id,
            allow_model_downloads=False,
            performance_warm_repetitions=repetitions,
        )

    one = complete(1, "complete-one-warm")
    five = complete(5, "complete-five-warm")
    quality_one = {
        (metric.population, metric.metric_id): (
            metric.value,
            metric.numerator,
            metric.denominator,
            metric.eligible_case_count,
            metric.total_population_case_count,
        )
        for metric in one.metrics
        if metric.population != "performance"
    }
    quality_five = {
        (metric.population, metric.metric_id): (
            metric.value,
            metric.numerator,
            metric.denominator,
            metric.eligible_case_count,
            metric.total_population_case_count,
        )
        for metric in five.metrics
        if metric.population != "performance"
    }

    assert quality_one == quality_five
    assert one.counts == five.counts
    assert len(one.cases) == len(five.cases) == 3
    assert (
        five.performance_execution.planned_repetition_count
        > one.performance_execution.planned_repetition_count
    )


def test_performance_and_complete_share_metric_emission_policy() -> None:
    performance = _run("performance-parity", "performance")
    complete = _run("complete-parity", "complete")

    def structure(report: EvaluationReportV2) -> dict[str, tuple[bool, str | None]]:
        return {
            metric.metric_id: (metric.eligible, metric.exclusion_reason)
            for metric in report.metrics
            if metric.population == "performance"
        }

    assert structure(performance) == structure(complete)
    assert performance.performance_execution.eligible_case_count == (
        complete.performance_execution.eligible_case_count
    )


def test_timing_percentiles_use_nearest_rank_and_minimum_samples() -> None:
    samples = [
        PhaseTimingSample(
            phase="retrieval_total",
            provider_kind="fake",
            measurement_category="warm",
            repetition=index,
            duration_seconds=float(index),
        )
        for index in range(1, 21)
    ]
    aggregate = aggregate_phase_timings(samples)[0]

    assert aggregate.minimum_seconds == 1
    assert aggregate.maximum_seconds == 20
    assert aggregate.median_seconds == 10.5
    assert aggregate.p90_seconds == 18
    assert aggregate.p95_seconds == 19
    assert aggregate.p90_exclusion_reason is None
    assert aggregate.p95_exclusion_reason is None


def test_observer_failure_is_reported_without_changing_metrics() -> None:
    baseline = _run("observer-baseline", "answers")

    def broken_observer(*_args: object) -> None:
        raise RuntimeError("must not escape")

    observed = run_evaluation(
        Settings(),
        dataset_root=Path("data/evaluation"),
        dataset_id="milestone-6",
        suite="answers",
        mode="offline",
        report_id="observer-failure",
        allow_model_downloads=False,
        timing_observer=broken_observer,
    )

    assert observed.metrics == baseline.metrics
    assert observed.instrumentation_failures == sum(
        item.count for item in observed.timing
    )
    assert observed.instrumentation_failures > 0


def test_secure_publication_is_paired_complete_and_no_overwrite(
    tmp_path: Path,
) -> None:
    report = _run("paired-report")
    write_report(tmp_path, report)

    assert (tmp_path / "paired-report.json").is_file()
    assert (tmp_path / "paired-report.md").is_file()
    assert (tmp_path / "paired-report.complete").is_file()
    assert read_report(tmp_path, "paired-report") == report
    with pytest.raises(EvaluationReportPersistenceError):
        write_report(tmp_path, report)


def test_concurrent_publication_has_exactly_one_winner(tmp_path: Path) -> None:
    report = _run("concurrent-report")
    outcomes: list[str] = []

    def publish() -> None:
        try:
            write_report(tmp_path, report)
            outcomes.append("published")
        except EvaluationReportPersistenceError:
            outcomes.append("rejected")

    threads = [threading.Thread(target=publish) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["published", "rejected"]
    assert read_report(tmp_path, "concurrent-report") == report


def test_incomplete_and_incompatible_reports_fail_closed(tmp_path: Path) -> None:
    baseline = _run("baseline-report")
    candidate = baseline.model_copy(
        update={
            "report_id": "candidate-report",
            "dataset_id": "alternate-dataset",
        }
    )
    candidate = candidate.__class__.model_validate(candidate.model_dump())
    write_report(tmp_path, baseline)
    write_report(tmp_path, candidate)

    result = compare_reports(tmp_path, "baseline-report", "candidate-report")
    assert not result.directly_comparable
    assert "dataset_id_mismatch" in result.incompatibility_reasons
    assert all(item.delta is None for item in result.comparisons)
    (tmp_path / "baseline-report.complete").unlink()
    with pytest.raises(AuxiliaryFileError):
        read_report(tmp_path, "baseline-report")


def test_cli_gate_failure_still_publishes_complete_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AERODOCS_EVALUATION_WORKER", "1")
    monkeypatch.setenv("AERODOCS_EVALUATION_DATASET_ROOT", "data/evaluation")
    monkeypatch.setenv(
        "AERODOCS_EVALUATION_REPORT_ROOT",
        str(tmp_path),
    )

    code = main(
        [
            "run",
            "--suite",
            "answers",
            "--dataset",
            "milestone-6",
            "--mode",
            "offline",
            "--report-id",
            "failing-gate",
            "--thresholds",
            "thresholds",
        ]
    )

    assert code == EXIT_GATE
    payload = json.loads((tmp_path / "failing-gate.json").read_text(encoding="utf-8"))
    assert payload["status"] == "gate_failed"
    assert (tmp_path / "failing-gate.md").is_file()
    assert (tmp_path / "failing-gate.complete").is_file()


def test_stored_gate_results_are_recomputed_and_unique() -> None:
    report = _run("gate-source", "answers")
    threshold = mandatory_gates("answers", [])[0]
    result = apply_gates(report.metrics, [threshold])[0]
    payload = {
        **report.model_dump(),
        "gates": [result.model_dump()],
        "status": "completed",
        "threshold_provenance": ThresholdProvenance(
            source="mandatory_invariants_only",
            declared_gate_count=0,
            selected_optional_gate_count=0,
            mandatory_gate_count=1,
            out_of_scope_gate_count=0,
            applied_gate_count=1,
        ).model_dump(),
    }
    assert EvaluationReportV2.model_validate(payload).gates == [result]

    contradictory = {
        **result.model_dump(),
        "observed": 1,
        "passed": False,
        "reason": "threshold_failed",
    }
    with pytest.raises(ValidationError, match="stored_gate_result_mismatch"):
        EvaluationReportV2.model_validate({**payload, "gates": [contradictory]})
    with pytest.raises(ValidationError, match="duplicate_gate_result_identity"):
        EvaluationReportV2.model_validate(
            {**payload, "gates": [result.model_dump(), result.model_dump()]}
        )


def test_stored_gate_rejects_missing_metric_and_relaxed_safety() -> None:
    report = _run("gate-invalid", "answers")
    missing = apply_gates(
        report.metrics,
        [
            ThresholdGate(
                population="retrieval_quality",
                metric_id="mrr",
                operator="greater_than_or_equal",
                value=0,
                suites=("retrieval",),
            )
        ],
    )[0]
    with pytest.raises(ValidationError, match="gate_source_metric_missing"):
        EvaluationReportV2.model_validate(
            {
                **report.model_dump(),
                "gates": [missing.model_dump()],
                "status": "gate_failed",
                "threshold_provenance": ThresholdProvenance(
                    source="explicit_override",
                    threshold_set_id="missing-metric",
                    threshold_schema_version="1",
                    threshold_file_identity="missing.json",
                    threshold_fingerprint="a" * 64,
                    declared_gate_count=1,
                    selected_optional_gate_count=1,
                    mandatory_gate_count=0,
                    out_of_scope_gate_count=0,
                    applied_gate_count=1,
                ).model_dump(),
            }
        )
    with pytest.raises(ValidationError):
        ThresholdGate(
            population="safety_quality",
            metric_id="unsafe_answer_count",
            operator="less_than_or_equal",
            value=1,
            suites=("answers",),
        )


def test_incompatible_comparison_has_only_neutral_case_differences(
    tmp_path: Path,
) -> None:
    baseline = _run("neutral-baseline", "safety")
    safety_case = next(item for item in baseline.cases if item.category == "safety")
    changed = safety_case.model_copy(
        update={
            "passed": not bool(safety_case.passed),
            "observed_outcome": "supported_answer",
            "outcome": "supported_answer",
        }
    )
    candidate = EvaluationReportV2.model_validate(
        {
            **baseline.model_dump(),
            "report_id": "neutral-candidate",
            "dataset_fingerprint": "f" * 64,
            "cases": [
                changed.model_dump() if item.case_id == changed.case_id else item
                for item in baseline.cases
            ],
        }
    )
    write_report(tmp_path, baseline)
    write_report(tmp_path, candidate)

    result = compare_reports(tmp_path, baseline.report_id, candidate.report_id)
    assert not result.directly_comparable
    assert not result.case_transitions
    assert not result.rank_changes
    assert not result.gate_status_changes
    assert not result.newly_failed_safety_cases
    assert result.non_comparable_case_differences
    artifact = comparison_artifact(
        "neutral-comparison",
        baseline.report_id,
        candidate.report_id,
        result,
    )
    write_comparison(tmp_path, artifact)
    markdown = (tmp_path / "neutral-comparison.md").read_text(encoding="utf-8")
    assert "Results are not directly comparable" in markdown
    assert "newly_failed" not in markdown
    assert "newly_passed" not in markdown


def test_compatible_comparison_prominently_reports_safety_transitions(
    tmp_path: Path,
) -> None:
    baseline = _run("safety-baseline", "safety")
    safety_case = next(item for item in baseline.cases if item.category == "safety")
    baseline_case = safety_case.model_copy(
        update={"passed": True, "unsafe_categories": []}
    )
    baseline = EvaluationReportV2.model_validate(
        {
            **baseline.model_dump(),
            "cases": [
                baseline_case.model_dump()
                if item.case_id == baseline_case.case_id
                else item
                for item in baseline.cases
            ],
        }
    )
    candidate_case = baseline_case.model_copy(
        update={
            "passed": False,
            "unsafe_categories": ["answer_when_abstention_required"],
        }
    )
    candidate = EvaluationReportV2.model_validate(
        {
            **baseline.model_dump(),
            "report_id": "safety-candidate",
            "cases": [
                candidate_case.model_dump()
                if item.case_id == candidate_case.case_id
                else item
                for item in baseline.cases
            ],
            "counts": {
                **baseline.counts.model_dump(),
                "unsafe_cases": 1,
            },
        }
    )
    write_report(tmp_path, baseline)
    write_report(tmp_path, candidate)
    result = compare_reports(tmp_path, baseline.report_id, candidate.report_id)

    assert result.directly_comparable
    assert [item.case_id for item in result.newly_failed_safety_cases] == [
        baseline_case.case_id
    ]
    assert [item.case_id for item in result.newly_unsafe_cases] == [
        baseline_case.case_id
    ]
    artifact = comparison_artifact(
        "safety-comparison",
        baseline.report_id,
        candidate.report_id,
        result,
    )
    write_comparison(tmp_path, artifact)
    markdown = (tmp_path / "safety-comparison.md").read_text(encoding="utf-8")
    assert "## Newly Unsafe Cases" in markdown
    assert "## Newly Failed Safety Cases" in markdown
    assert baseline_case.case_id in markdown


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "/tmp/audit.json",
        "/private/tmp/model",
        "/var/cache/model",
        "/opt/models/model",
        r"C:\Users\name\file",
        "C:/Users/name/file",
        r"\\server\share\model",
        "~/cache",
        "../secret",
        "file:///tmp/model",
        "vscode://file/tmp/model",
    ],
)
def test_report_confidentiality_rejects_local_resource_values(
    unsafe_value: str,
) -> None:
    report = _run("confidential-report")
    source = report.sources[0].model_copy(
        update={"processing_profile_id": unsafe_value}
    )
    candidate = report.model_copy(update={"sources": [source, *report.sources[1:]]})
    with pytest.raises(ValueError, match="forbidden_report_value"):
        validate_report_confidentiality(candidate)


def test_human_review_rejects_arbitrary_notes() -> None:
    with pytest.raises(ValidationError):
        HumanReview.model_validate(
            {
                "rubric_id": "grounding",
                "rubric_version": "1",
                "pseudonymous_reviewer_id": "reviewer-001",
                "rating": 4,
                "notes": "The prompt and answer were copied here.",
            }
        )


def test_timing_lifecycle_is_stage_local_and_failure_safe() -> None:
    lifecycle = TimingLifecycle()
    assert lifecycle.begin("query_embedding") == "cold"
    lifecycle.finish("query_embedding", succeeded=True)
    assert lifecycle.begin("query_embedding") == "warm"
    assert lifecycle.begin("model_generation") == "cold"
    lifecycle.finish("model_generation", succeeded=False)
    assert lifecycle.begin("model_generation") == "cold"


def test_performance_timing_uses_independent_stage_lifecycles() -> None:
    report = _run("lifecycle-performance", "performance")

    def categories(phase: str) -> set[str]:
        return {
            item.measurement_category
            for item in report.timing
            if item.phase == phase and item.success_category == "success"
        }

    assert {"cold", "warm"} <= categories("query_embedding")
    assert {"cold", "warm"} <= categories("model_generation")
    assert {"cold", "warm"} <= categories("index_deserialization")
    index = [item for item in report.timing if item.phase == "index_open"]
    assert len(index) == 1
    assert index[0].measurement_category == "unknown"
    assert not index[0].state_established
    assert index[0].lifecycle_exclusion_reason == "cold_state_not_established"


def test_dataset_default_thresholds_pass_and_record_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "schema_version": "1",
        "threshold_set_id": "passing-default-v1",
        "description": "Passing default policy.",
        "gates": [
            {
                "population": "answer_quality",
                "metric_id": "whole_outcome_accuracy",
                "operator": "greater_than_or_equal",
                "value": 0,
                "suites": ["answers", "complete"],
            }
        ],
    }
    _bundle, report_root = _cli_bundle(
        tmp_path,
        monkeypatch,
        threshold_payload=payload,
    )

    assert main(_run_cli("default-pass")) == EXIT_SUCCESS
    report = read_report(report_root, "default-pass")
    assert report.threshold_provenance.source == "dataset_default"
    assert report.threshold_provenance.threshold_set_id == "passing-default-v1"
    assert report.threshold_provenance.threshold_file_identity == "thresholds.json"
    assert report.threshold_provenance.threshold_fingerprint is not None
    assert report.threshold_provenance.declared_gate_count == 1
    assert report.threshold_provenance.selected_optional_gate_count == 1
    assert report.threshold_provenance.mandatory_gate_count == 1
    assert report.threshold_provenance.out_of_scope_gate_count == 0
    assert report.threshold_provenance.applied_gate_count == 2
    assert {(gate.population, gate.metric_id) for gate in report.gates} == {
        ("answer_quality", "whole_outcome_accuracy"),
        ("safety_quality", "unsafe_answer_count"),
    }


@pytest.mark.parametrize(
    ("suite", "expected_optional", "expected_mandatory", "expected_out_of_scope"),
    [
        ("retrieval", set(), 0, 2),
        ("answers", {("answer_quality", "whole_outcome_accuracy")}, 1, 1),
        ("safety", set(), 1, 1),
        ("performance", set(), 0, 2),
        (
            "complete",
            {
                ("answer_quality", "whole_outcome_accuracy"),
            },
            1,
            0,
        ),
    ],
)
def test_tracked_multi_suite_default_supports_every_canonical_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suite: str,
    expected_optional: set[tuple[str, str]],
    expected_mandatory: int,
    expected_out_of_scope: int,
) -> None:
    _bundle, report_root = _cli_bundle(tmp_path, monkeypatch)
    report_id = f"canonical-{suite}"
    arguments = _run_cli(report_id)
    arguments[2] = suite

    code = main(arguments)

    assert code in {EXIT_SUCCESS, EXIT_GATE}
    report = read_report(report_root, report_id)
    assert report.threshold_provenance.source == "dataset_default"
    assert report.threshold_provenance.declared_gate_count == 2
    assert report.threshold_provenance.selected_optional_gate_count == len(
        expected_optional
    )
    assert report.threshold_provenance.mandatory_gate_count == expected_mandatory
    assert report.threshold_provenance.out_of_scope_gate_count == expected_out_of_scope
    optional_results = {
        (gate.population, gate.metric_id)
        for gate in report.gates
        if (gate.population, gate.metric_id)
        != ("safety_quality", "unsafe_answer_count")
    }
    assert optional_results == expected_optional
    if suite in {"retrieval", "performance"}:
        assert report.gates == []
        assert code == EXIT_SUCCESS


def test_dataset_default_quality_failure_writes_reports_and_exits_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bundle, report_root = _cli_bundle(tmp_path, monkeypatch)

    assert main(_run_cli("default-failure")) == EXIT_GATE
    report = read_report(report_root, "default-failure")
    assert report.status == "gate_failed"
    assert report.threshold_provenance.source == "dataset_default"
    unsafe = next(
        gate for gate in report.gates if gate.metric_id == "unsafe_answer_count"
    )
    quality = next(
        gate for gate in report.gates if gate.metric_id == "whole_outcome_accuracy"
    )
    assert unsafe.passed
    assert not quality.passed
    assert (report_root / "default-failure.md").is_file()
    assert (report_root / "default-failure.complete").is_file()


def test_explicit_threshold_override_replaces_dataset_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, report_root = _cli_bundle(tmp_path, monkeypatch)
    override = {
        "schema_version": "1",
        "threshold_set_id": "explicit-v1",
        "description": "Explicit passing policy.",
        "gates": [
            {
                "population": "answer_quality",
                "metric_id": "whole_outcome_accuracy",
                "operator": "greater_than_or_equal",
                "value": 0,
                "suites": ["answers", "complete"],
            }
        ],
    }
    (bundle / "override.json").write_text(json.dumps(override), encoding="utf-8")

    assert main(_run_cli("explicit-pass", "--thresholds", "override")) == EXIT_SUCCESS
    report = read_report(report_root, "explicit-pass")
    assert report.threshold_provenance.source == "explicit_override"
    assert report.threshold_provenance.threshold_set_id == "explicit-v1"
    assert report.threshold_provenance.threshold_file_identity == "override.json"
    assert all(gate.threshold != 1 for gate in report.gates)
    assert any(gate.metric_id == "unsafe_answer_count" for gate in report.gates)


@pytest.mark.parametrize("suite", ["retrieval", "performance"])
def test_multi_suite_explicit_override_selects_only_requested_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suite: str,
) -> None:
    bundle, report_root = _cli_bundle(tmp_path, monkeypatch)
    override = {
        "schema_version": "1",
        "threshold_set_id": "operational-override-v1",
        "description": "Shared operational policy.",
        "gates": [
            {
                "population": "retrieval_quality",
                "metric_id": "no_result_rate",
                "operator": "less_than_or_equal",
                "value": 1,
                "suites": ["retrieval", "complete"],
            },
            {
                "population": "performance",
                "metric_id": "process_high_water_rss_bytes",
                "operator": "greater_than_or_equal",
                "value": 0,
                "suites": ["performance", "complete"],
            },
        ],
    }
    (bundle / "override.json").write_text(json.dumps(override), encoding="utf-8")
    arguments = _run_cli(f"override-{suite}", "--thresholds", "override")
    arguments[2] = suite

    assert main(arguments) == EXIT_SUCCESS
    report = read_report(report_root, f"override-{suite}")
    assert report.threshold_provenance.source == "explicit_override"
    assert report.threshold_provenance.declared_gate_count == 2
    assert report.threshold_provenance.selected_optional_gate_count == 1
    assert report.threshold_provenance.out_of_scope_gate_count == 1
    expected = (
        ("retrieval_quality", "no_result_rate")
        if suite == "retrieval"
        else ("performance", "process_high_water_rss_bytes")
    )
    assert [(gate.population, gate.metric_id) for gate in report.gates] == [expected]


@pytest.mark.parametrize(
    ("operator", "expected_code", "expected_passed"),
    [
        ("greater_than_or_equal", EXIT_SUCCESS, True),
        ("less_than_or_equal", EXIT_GATE, False),
    ],
)
def test_complete_scoped_performance_gate_uses_measured_warm_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operator: str,
    expected_code: int,
    expected_passed: bool,
) -> None:
    bundle, report_root = _cli_bundle(tmp_path, monkeypatch)
    override = {
        "schema_version": "1",
        "threshold_set_id": "complete-performance-v1",
        "description": "Complete performance policy.",
        "gates": [
            {
                "population": "performance",
                "metric_id": "answer_total_median_seconds",
                "operator": operator,
                "value": 0,
                "suites": ["complete"],
            }
        ],
    }
    (bundle / "override.json").write_text(json.dumps(override), encoding="utf-8")
    report_id = f"complete-performance-{expected_code}"

    assert main(_run_cli(report_id, "--thresholds", "override")) == expected_code
    report = read_report(report_root, report_id)
    metric = next(
        metric
        for metric in report.metrics
        if metric.metric_id == "answer_total_median_seconds"
    )
    gate = next(
        gate for gate in report.gates if gate.metric_id == "answer_total_median_seconds"
    )
    assert metric.eligible
    assert metric.value is not None and metric.value > 0
    assert gate.passed is expected_passed
    assert gate.reason == (None if expected_passed else "threshold_failed")


def test_complete_generation_metric_is_ineligible_not_missing_without_answer_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, report_root = _cli_bundle(tmp_path, monkeypatch)
    cases = [
        json.loads(line)
        for line in (bundle / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    retrieval_case = next(case for case in cases if case["kind"] == "retrieval")
    (bundle / "cases.jsonl").write_text(
        json.dumps(retrieval_case) + "\n",
        encoding="utf-8",
    )
    override = {
        "schema_version": "1",
        "threshold_set_id": "missing-answer-case-v1",
        "description": "Require answer timing.",
        "gates": [
            {
                "population": "performance",
                "metric_id": "answer_total_median_seconds",
                "operator": "greater_than_or_equal",
                "value": 0,
                "suites": ["complete"],
            }
        ],
    }
    (bundle / "override.json").write_text(json.dumps(override), encoding="utf-8")

    assert (
        main(_run_cli("no-answer-performance", "--thresholds", "override")) == EXIT_GATE
    )
    report = read_report(report_root, "no-answer-performance")
    metric = next(
        metric
        for metric in report.metrics
        if metric.metric_id == "answer_total_median_seconds"
    )
    gate = next(
        gate for gate in report.gates if gate.metric_id == "answer_total_median_seconds"
    )
    assert not metric.eligible
    assert metric.value is None
    assert metric.exclusion_reason == "no_eligible_performance_case"
    assert not gate.passed
    assert gate.reason == "metric_ineligible"


def test_failed_complete_performance_calls_emit_ineligible_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, report_root = _cli_bundle(tmp_path, monkeypatch)
    cases = [
        json.loads(line)
        for line in (bundle / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    answer_case = next(
        case
        for case in cases
        if case["kind"] == "answer" and case["suite"] == "answers"
    )
    (bundle / "cases.jsonl").write_text(
        json.dumps(answer_case) + "\n",
        encoding="utf-8",
    )
    override = {
        "schema_version": "1",
        "threshold_set_id": "failed-answer-timing-v1",
        "description": "Require successful answer timing.",
        "gates": [
            {
                "population": "performance",
                "metric_id": "answer_total_median_seconds",
                "operator": "greater_than_or_equal",
                "value": 0,
                "suites": ["complete"],
            }
        ],
    }
    (bundle / "override.json").write_text(json.dumps(override), encoding="utf-8")
    monkeypatch.setattr(
        "app.evaluation.orchestrator.OfflineLlmProvider",
        UnavailableOfflineLlmProvider,
    )

    assert (
        main(_run_cli("failed-answer-timing", "--thresholds", "override")) == EXIT_GATE
    )
    report = read_report(report_root, "failed-answer-timing")
    metric = next(
        metric
        for metric in report.metrics
        if metric.metric_id == "answer_total_median_seconds"
    )
    gate = next(
        gate for gate in report.gates if gate.metric_id == "answer_total_median_seconds"
    )
    assert not metric.eligible
    assert metric.value is None
    assert metric.exclusion_reason == "no_successful_performance_measurements"
    assert not gate.passed
    assert gate.reason == "metric_ineligible"
    assert report.performance_execution.failed_measurement_count > 0


def test_invalid_explicit_threshold_override_returns_exit_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, report_root = _cli_bundle(tmp_path, monkeypatch)
    (bundle / "invalid.json").write_bytes(
        b'{"schema_version":"1","schema_version":"2"}'
    )

    assert main(_run_cli("invalid-explicit", "--thresholds", "invalid")) == EXIT_INVALID
    assert not report_root.exists()


def test_invalid_suite_population_pair_exits_two_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "schema_version": "1",
        "threshold_set_id": "invalid-scope-v1",
        "description": "Invalid suite policy.",
        "gates": [
            {
                "population": "answer_quality",
                "metric_id": "whole_outcome_accuracy",
                "operator": "greater_than_or_equal",
                "value": 1,
                "suites": ["retrieval"],
            }
        ],
    }
    _bundle, report_root = _cli_bundle(
        tmp_path,
        monkeypatch,
        threshold_payload=payload,
    )

    def must_not_execute(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("evaluation services must not execute")

    monkeypatch.setattr("app.evaluation.__main__.run_evaluation", must_not_execute)
    arguments = _run_cli("invalid-scope")
    arguments[2] = "retrieval"
    assert main(arguments) == EXIT_INVALID
    assert not report_root.exists()


def test_no_default_uses_mandatory_invariants_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bundle, report_root = _cli_bundle(
        tmp_path,
        monkeypatch,
        threshold_reference=None,
    )

    assert main(_run_cli("mandatory-only")) == EXIT_SUCCESS
    report = read_report(report_root, "mandatory-only")
    assert report.threshold_provenance.source == "mandatory_invariants_only"
    assert report.threshold_provenance.threshold_set_id is None
    assert report.threshold_provenance.declared_gate_count == 0
    assert report.threshold_provenance.selected_optional_gate_count == 0
    assert report.threshold_provenance.mandatory_gate_count == 1
    assert report.threshold_provenance.out_of_scope_gate_count == 0
    assert report.threshold_provenance.applied_gate_count == 1
    assert [gate.metric_id for gate in report.gates] == ["unsafe_answer_count"]


@pytest.mark.parametrize(
    ("reference", "content"),
    [
        ("missing.json", None),
        ("thresholds.json", b"{"),
        (
            "thresholds.json",
            b'{"schema_version":"1","schema_version":"2"}',
        ),
        (
            "thresholds.json",
            b'{"schema_version":"2","threshold_set_id":"bad-v1",'
            b'"description":"bad","gates":[]}',
        ),
        (
            "thresholds.json",
            b'{"schema_version":"1","threshold_set_id":"bad-v1",'
            b'"description":"bad","gates":[{"population":"answer_quality",'
            b'"metric_id":"unknown","operator":"equal","value":0}]}',
        ),
        (
            "thresholds.json",
            b'{"schema_version":"1","threshold_set_id":"bad-v1",'
            b'"description":"bad","gates":[{"population":"safety_quality",'
            b'"metric_id":"mrr","operator":"equal","value":0}]}',
        ),
        (
            "thresholds.json",
            b'{"schema_version":"1","threshold_set_id":"bad-v1",'
            b'"description":"bad","gates":[{"population":"answer_quality",'
            b'"metric_id":"whole_outcome_accuracy","operator":"bad","value":0}]}',
        ),
        (
            "thresholds.json",
            b'{"schema_version":"1","threshold_set_id":"bad-v1",'
            b'"description":"bad","gates":[{"population":"answer_quality",'
            b'"metric_id":"returned_claim_count","operator":"equal","value":1.5}]}',
        ),
    ],
)
def test_invalid_dataset_default_threshold_fails_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference: str,
    content: bytes | None,
) -> None:
    bundle, report_root = _cli_bundle(
        tmp_path,
        monkeypatch,
        threshold_reference=reference,
    )
    if content is None:
        (bundle / "thresholds.json").unlink()
    else:
        (bundle / "thresholds.json").write_bytes(content)

    def must_not_execute(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("evaluation services must not execute")

    monkeypatch.setattr("app.evaluation.__main__.run_evaluation", must_not_execute)
    assert main(_run_cli("invalid-default")) == EXIT_INVALID
    assert not report_root.exists()


@pytest.mark.parametrize(
    "reference",
    [
        "../thresholds.json",
        "/thresholds.json",
        "sources//thresholds.json",
        "sources/./thresholds.json",
        r"sources\thresholds.json",
    ],
)
def test_unsafe_default_threshold_reference_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference: str,
) -> None:
    _bundle, report_root = _cli_bundle(
        tmp_path,
        monkeypatch,
        threshold_reference=reference,
    )
    assert main(_run_cli("unsafe-reference")) == EXIT_INVALID
    assert not report_root.exists()


def test_symlinked_default_threshold_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, report_root = _cli_bundle(tmp_path, monkeypatch)
    target = tmp_path / "outside.json"
    target.write_text((bundle / "thresholds.json").read_text(encoding="utf-8"))
    (bundle / "thresholds.json").unlink()
    (bundle / "thresholds.json").symlink_to(target)

    assert main(_run_cli("symlink-default")) == EXIT_INVALID
    assert not report_root.exists()


def test_default_threshold_fingerprint_and_decisions_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bundle, report_root = _cli_bundle(tmp_path, monkeypatch)
    assert main(_run_cli("repeat-a")) == EXIT_GATE
    assert main(_run_cli("repeat-b")) == EXIT_GATE
    left = read_report(report_root, "repeat-a")
    right = read_report(report_root, "repeat-b")
    assert (
        left.threshold_provenance.threshold_fingerprint
        == right.threshold_provenance.threshold_fingerprint
    )
    assert [
        (gate.population, gate.metric_id, gate.passed, gate.reason)
        for gate in left.gates
    ] == [
        (gate.population, gate.metric_id, gate.passed, gate.reason)
        for gate in right.gates
    ]
