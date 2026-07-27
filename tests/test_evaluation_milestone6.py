"""Milestone 6 reproducibility, security, matching, and metric tests."""

from __future__ import annotations

import hashlib
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.evaluation.aggregation import (
    AnswerObservation,
    RetrievalObservation,
    answer_metrics,
    operational_metrics,
    retrieval_metrics,
)
from app.evaluation.benchmark import (
    GoldClaim,
    NumericFact,
    PageTarget,
    load_dataset,
)
from app.evaluation.matching import (
    ObservedSupport,
    ReturnedClaim,
    match_claims,
    target_id,
)
from app.evaluation.metrics import metric_record
from app.evaluation.offline import NetworkDeniedError, network_denial_guard
from app.evaluation.reporting import (
    CaseReport,
    CorpusIdentity,
    EnvironmentSummary,
    EvaluationReportV2,
    GateThreshold,
    GitMetadata,
    ProviderReport,
    RunCounts,
    SettingsSnapshot,
    SourceIdentity,
    apply_gates,
    compare_reports,
    write_report,
)
from app.evaluation.security import AuxiliaryFileError, read_bytes
from app.evaluation.synthetic import (
    SyntheticPage,
    SyntheticPdfSpec,
    identity,
    render_pdf,
)
from app.evaluation.timing import MemoryObservation, TimingCollector, notify_observer


def synthetic_spec() -> SyntheticPdfSpec:
    return SyntheticPdfSpec(
        source_id="src_test_manual",
        title="Synthetic Manual",
        pages=[
            SyntheticPage(lines=["SECTION 1", "A qualified human reviews findings."]),
            SyntheticPage(lines=["SECTION 2", "No airworthiness approval is given."]),
        ],
    )


def test_synthetic_pdf_and_identity_are_byte_identical() -> None:
    first = render_pdf(synthetic_spec())
    second = render_pdf(synthetic_spec())

    assert first == second
    assert hashlib.sha256(first).hexdigest() == identity(synthetic_spec()).sha256
    assert identity(synthetic_spec()).document_id.startswith("doc_")
    assert b"/CreationDate" not in first
    assert b"/ID" not in first


def test_tracked_bundle_is_strict_and_resolves_source_ids() -> None:
    root = Path("data/evaluation/milestone-6")
    dataset = load_dataset(root)

    assert dataset.manifest.dataset_id == "milestone-6"
    assert len(dataset.cases) == 3
    assert len(dataset.fingerprint) == 64


def test_safe_reader_rejects_escape_and_symlink(tmp_path: Path) -> None:
    (tmp_path / "inside.json").write_text("{}", encoding="utf-8")
    outside = tmp_path.parent / "outside-evaluation.json"
    outside.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(outside)

    assert read_bytes(tmp_path, "inside.json") == b"{}"
    with pytest.raises(AuxiliaryFileError):
        read_bytes(tmp_path, "../outside-evaluation.json")
    with pytest.raises(AuxiliaryFileError):
        read_bytes(tmp_path, "link.json")


def test_claim_matching_is_one_to_one_and_reorder_stable() -> None:
    fact = NumericFact(fact_id="f1", value="10")
    gold = [
        GoldClaim(
            claim_id="g2",
            support_targets=[
                PageTarget(
                    source_id="src_test_manual",
                    page_start=2,
                    page_end=2,
                )
            ],
            facts=[fact],
            exact_quotes=[],
        ),
        GoldClaim(
            claim_id="g1",
            support_targets=[
                PageTarget(
                    source_id="src_test_manual",
                    page_start=1,
                    page_end=1,
                )
            ],
            facts=[fact],
            exact_quotes=[],
        ),
    ]
    first_target = target_id(gold[1].support_targets[0])
    second_target = target_id(gold[0].support_targets[0])

    def support(
        evidence_id: str, target: str, *, valid: bool = True
    ) -> ObservedSupport:
        return ObservedSupport(
            evidence_id=evidence_id,
            target_ids=[target],
            evidence_id_valid=valid,
            quotation_present=True,
            quotation_valid=valid,
            provenance_valid=valid,
            citation_consistent=valid,
        )

    returned = [
        ReturnedClaim(
            claim_id="r2",
            text="10",
            supports=[support("e2", second_target)],
            preserved_fact_ids=["f1"],
        ),
        ReturnedClaim(
            claim_id="r1",
            text="10",
            supports=[support("e1", first_target)],
            preserved_fact_ids=["f1"],
        ),
        ReturnedClaim(
            claim_id="invalid",
            text="10",
            supports=[support("bad", second_target, valid=False)],
            preserved_fact_ids=["f1"],
        ),
    ]

    result = match_claims(gold, returned)
    reordered = match_claims(list(reversed(gold)), list(reversed(returned)))

    assert result == reordered
    assert [(pair.gold_claim_id, pair.returned_claim_id) for pair in result.pairs] == [
        ("g1", "r1"),
        ("g2", "r2"),
    ]
    assert result.unmatched_returned_claim_ids == ["invalid"]


def test_metric_null_and_gate_insufficiency_are_explicit() -> None:
    metric = metric_record("whole_outcome_accuracy", 0, 0, "case_accuracy")
    gate = GateThreshold(
        metric_id=metric.metric_id,
        operator="greater_than_or_equal",
        value=0.9,
        population="answer_quality",
        suites=("answers",),
    )

    assert metric.value is None
    assert metric.exclusion_reason == "no_eligible_cases"
    result = apply_gates([metric], [gate])[0]
    assert not result.passed
    assert result.reason == "metric_ineligible"
    with pytest.raises(ValueError):
        GateThreshold(
            metric_id="unsafe_answer_count",
            operator="less_than_or_equal",
            value=1,
            population="safety_quality",
            suites=("answers",),
        )


def test_quality_and_fault_populations_have_separate_denominators() -> None:
    retrieval = [
        RetrievalObservation(
            case_id="quality",
            population="model_quality",
            execution_succeeded=False,
            target_grades={"target": 2},
        ),
        RetrievalObservation(
            case_id="fault",
            population="framework_validation",
            execution_succeeded=False,
            target_grades={"target": 2},
            failure_category="embedding_model_unavailable",
        ),
    ]
    answers = [
        AnswerObservation(
            case_id="supported",
            population="model_quality",
            execution_succeeded=True,
            expected_outcome="supported_answer",
            observed_outcome="supported_answer",
            returned_claim_count=1,
            returned_support_count=1,
            matched_claim_count=1,
            required_gold_claims=1,
            matched_required_gold_claims=1,
            complete_support_set=True,
        ),
        AnswerObservation(
            case_id="answer-fault",
            population="framework_validation",
            execution_succeeded=False,
            expected_outcome="model_unavailable",
            observed_outcome="model_unavailable",
            failure_category="llm_model_unavailable",
        ),
    ]

    retrieval_records = {
        item.metric_id: item for item in retrieval_metrics(retrieval, cutoff=5)
    }
    answer_records = {item.metric_id: item for item in answer_metrics(answers)}
    operations = operational_metrics(retrieval, answers)[0]

    assert retrieval_records["recall_at_5"].denominator == 1
    assert retrieval_records["recall_at_5"].value == 0
    assert answer_records["whole_outcome_accuracy"].denominator == 1
    assert answer_records["unsafe_answer_count"].value == 0
    assert operations.denominator == 2
    assert operations.value == 1


def test_timing_observer_failure_is_isolated_and_thread_safe() -> None:
    collector = TimingCollector()

    def broken(*_args: object) -> None:
        raise RuntimeError("observer")

    threads = [
        threading.Thread(
            target=notify_observer,
            kwargs={
                "observer": broken,
                "collector": collector,
                "phase": "retrieval",
                "safe_id": f"case_{index}",
                "duration_seconds": 0.1,
                "category": "success",
            },
        )
        for index in range(10)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(collector.events) == 10
    assert collector.instrumentation_failures == 10


def test_network_guard_denies_socket_connections() -> None:
    import socket

    with network_denial_guard(), pytest.raises(NetworkDeniedError):
        socket.create_connection(("example.invalid", 443))


def _report(report_id: str, value: float) -> EvaluationReportV2:
    metric = metric_record(
        "recall_at_5", value, 1, "macro", population="retrieval_quality"
    )
    return EvaluationReportV2(
        run_id="f" * 32,
        report_id=report_id,
        mode="offline",
        suite="complete",
        status="completed",
        started_at=datetime(2020, 1, 1, tzinfo=UTC),
        finished_at=datetime(2020, 1, 1, tzinfo=UTC),
        observational_duration_seconds=0,
        dataset_id="dataset",
        dataset_version="1",
        dataset_fingerprint="a" * 64,
        evaluation_profile_id="profile",
        git=GitMetadata(commit="e" * 40, dirty=False),
        sources=[
            SourceIdentity(
                source_id="source",
                source_fingerprint="c" * 64,
                redistribution_mode="benchmark_authored",
                resolved_document_id="doc_" + "d" * 64,
                processing_profile_id="process",
                chunking_profile_id="chunk",
            )
        ],
        corpus=CorpusIdentity(
            corpus_fingerprint="cor_" + "b" * 64,
            index_profile_id="index",
            vector_profile_id="vector",
            vector_count=1,
            document_count=1,
        ),
        providers=ProviderReport(
            provider_kind="fake",
            embedding_model_name="fake-embedding",
            embedding_revision="offline-v1",
            generation_model_name="fake-generation",
            generation_revision="offline-v1",
            local_files_only=True,
            downloads_allowed=False,
        ),
        environment=EnvironmentSummary(
            operating_system_family="TestOS",
            architecture="test-arch",
            python_version="3.11.0",
            hardware_class="test-class",
        ),
        settings=SettingsSnapshot(
            retrieval_default_top_k=5,
            retrieval_maximum_top_k=20,
            minimum_similarity_score=0,
            answer_minimum_similarity_score=0,
            embedding_normalization=True,
            index_format_version="1",
            maximum_generation_output_tokens=100,
        ),
        metrics=[metric],
        cases=[
            CaseReport(
                case_id="case",
                case_type="retrieval",
                category="complete",
                population="model_quality",
                selected=True,
                eligible=True,
                passed=True,
                exclusion_reason=None,
                outcome="answer",
                expected_outcome="results",
                observed_outcome="answer",
                metrics=[],
            )
        ],
        counts=RunCounts(
            total_cases=1,
            selected_cases=1,
            not_selected_cases=0,
            operational_failures=0,
            unsafe_cases=0,
        ),
        memory=MemoryObservation(exclusion_reason="memory_observation_unavailable"),
    )


def test_secure_report_comparison_preserves_denominators(tmp_path: Path) -> None:
    write_report(tmp_path, _report("baseline", 0.5))
    write_report(tmp_path, _report("candidate", 1.0))

    comparison = compare_reports(tmp_path, "baseline", "candidate").comparisons[0]

    assert comparison.baseline_value == 0.5
    assert comparison.candidate_value == 1.0
    assert comparison.baseline_denominator == comparison.candidate_denominator == 1
    with pytest.raises(AuxiliaryFileError):
        compare_reports(tmp_path, "../baseline", "candidate")
