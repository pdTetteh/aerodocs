"""Canonical Milestone 6 orchestration through the production service graph."""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal
from uuid import uuid4

from app.core.config import Settings
from app.evaluation.aggregation import (
    AnswerObservation,
    RetrievalObservation,
    answer_metrics,
    operational_metrics,
    retrieval_metrics,
)
from app.evaluation.benchmark import (
    AnswerBenchmarkCase,
    BenchmarkDataset,
    ResolvedDocument,
    RetrievalBenchmarkCase,
    SectionOccurrence,
    load_dataset_by_id,
    normalize_section_heading,
    preflight_dataset_by_id,
    validate_resolved_targets,
)
from app.evaluation.metric_definitions import performance_metric_definitions
from app.evaluation.metrics import ineligible_scalar_record, scalar_record
from app.evaluation.observations import (
    ServiceTrace,
    build_answer_observation,
    build_retrieval_observation,
)
from app.evaluation.offline import network_denial_guard, real_mode_settings
from app.evaluation.providers import (
    FailingOfflineEmbeddingProvider,
    MalformedOfflineLlmProvider,
    OfflineEmbeddingProvider,
    OfflineLlmProvider,
    TimeoutOfflineLlmProvider,
    UnavailableOfflineLlmProvider,
)
from app.evaluation.reporting import (
    CaseReport,
    CorpusIdentity,
    EvaluationReportV2,
    PerformanceExecutionSummary,
    ProviderReport,
    RunCounts,
    SettingsSnapshot,
    SourceIdentity,
    environment_summary,
    git_metadata,
)
from app.evaluation.reproducibility import verify_synthetic_reproducibility
from app.evaluation.security import AuxiliaryFileError, SecureDirectory
from app.evaluation.timing import (
    PhaseTimingSample,
    TimingCollector,
    TimingLifecycle,
    TimingLifecycleLabel,
    TimingObserver,
    aggregate_phase_timings,
    notify_observer,
    process_memory_observation,
)
from app.generation.errors import AuditPersistenceError, GenerationError
from app.generation.models import (
    AnswerRequest,
    PreparedGenerationInput,
    ProviderIdentity,
)
from app.generation.provider import LlmProvider, TransformersLlmProvider
from app.generation.service import AnswerService
from app.ingestion.headings import find_headings
from app.ingestion.service import DocumentService
from app.retrieval.embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)
from app.retrieval.errors import RetrievalError
from app.retrieval.models import ModelIdentity, SearchRequest
from app.retrieval.service import RetrievalService
from app.services.runtime import AeroDocsServices, create_services

EvaluationMode = Literal["offline", "real"]
EvaluationSuite = Literal["retrieval", "answers", "safety", "performance", "complete"]
EVALUATION_INGESTED_AT = datetime(2000, 1, 1, tzinfo=UTC)


def validate_bundle(
    settings: Settings,
    dataset_root: Path,
    dataset_id: str,
    *,
    threshold_input_bytes: int = 0,
) -> BenchmarkDataset:
    """Validate content identities and production ingestion reproducibility."""
    preflight = preflight_dataset_by_id(dataset_root, dataset_id)
    guard = (
        network_denial_guard()
        if os.environ.get("AERODOCS_EVALUATION_WORKER") == "1"
        else nullcontext()
    )
    with guard:
        if preflight.source_mode == "manifest_only":
            services = _open_existing_services(
                settings,
                embedding_provider=OfflineEmbeddingProvider(),
                llm_provider=OfflineLlmProvider(),
            )
            dataset = load_dataset_by_id(
                dataset_root,
                dataset_id,
                documents=services.documents,
                reserved_input_bytes=threshold_input_bytes,
            )
            try:
                _resolve_corpus(
                    services,
                    dataset,
                    dataset_root=dataset_root,
                    rebuild=False,
                )
            except RetrievalError as error:
                raise AuxiliaryFileError("manifest_corpus_validation_failed") from error
        else:
            with TemporaryDirectory(prefix="aerodocs-validate-") as temporary:
                root = Path(temporary)
                validation_settings = settings.model_copy(
                    update={
                        "document_storage_root": root / "documents",
                        "index_runtime_root": root / "indexes",
                        "generation_audit_root": root / "audits",
                        "embedding_model_name": "aerodocs-evaluation-fake",
                        "embedding_model_revision": "offline-v1",
                    }
                )
                services = create_services(
                    validation_settings,
                    embedding_provider=OfflineEmbeddingProvider(),
                    llm_provider=OfflineLlmProvider(),
                )
                dataset = load_dataset_by_id(
                    dataset_root,
                    dataset_id,
                    documents=services.documents,
                    reserved_input_bytes=threshold_input_bytes,
                )
                for source in dataset.sources:
                    if source.kind == "synthetic":
                        verified = verify_synthetic_reproducibility(source, settings)
                        if (
                            verified.pdf_sha256 != source.source_fingerprint
                            or verified.document_id != source.document_id
                        ):
                            raise AuxiliaryFileError(
                                "synthetic_reproducibility_mismatch"
                            )
                _resolve_corpus(
                    services, dataset, dataset_root=dataset_root, rebuild=True
                )
        return dataset


def run_evaluation(
    settings: Settings,
    *,
    dataset_root: Path,
    dataset_id: str,
    suite: EvaluationSuite,
    mode: EvaluationMode,
    report_id: str,
    allow_model_downloads: bool,
    performance_warm_repetitions: int = 3,
    timing_observer: TimingObserver | None = None,
    threshold_input_bytes: int = 0,
) -> EvaluationReportV2:
    """Execute cases directly through Document, Retrieval, and Answer services."""
    preflight = preflight_dataset_by_id(dataset_root, dataset_id)
    guard = (
        network_denial_guard()
        if mode == "offline" and os.environ.get("AERODOCS_EVALUATION_WORKER") == "1"
        else nullcontext()
    )
    with guard:
        with TemporaryDirectory(prefix="aerodocs-evaluation-run-") as temporary:
            root = Path(temporary)
            if preflight.source_mode == "manifest_only":
                run_settings = settings.model_copy(
                    update={"generation_audit_root": root / "audits"}
                )
                if mode == "offline":
                    embedding_provider: EmbeddingProvider = OfflineEmbeddingProvider()
                    llm_provider: LlmProvider = OfflineLlmProvider()
                else:
                    run_settings = real_mode_settings(
                        run_settings,
                        allow_model_downloads=allow_model_downloads,
                    )
                    embedding_provider = SentenceTransformerEmbeddingProvider(
                        run_settings,
                        local_files_only=not allow_model_downloads,
                    )
                    llm_provider = TransformersLlmProvider(
                        run_settings,
                        local_files_only=not allow_model_downloads,
                    )
                services = _open_existing_services(
                    run_settings,
                    embedding_provider=embedding_provider,
                    llm_provider=llm_provider,
                )
            else:
                root = Path(temporary)
                updates: dict[str, object] = {
                    "document_storage_root": root / "documents",
                    "index_runtime_root": root / "indexes",
                    "generation_audit_root": root / "audits",
                }
                if mode == "offline":
                    updates.update(
                        {
                            "embedding_model_name": "aerodocs-evaluation-fake",
                            "embedding_model_revision": "offline-v1",
                        }
                    )
                run_settings = settings.model_copy(update=updates)
                if mode == "offline":
                    embedding_provider = OfflineEmbeddingProvider()
                    llm_provider = OfflineLlmProvider()
                else:
                    run_settings = real_mode_settings(
                        run_settings,
                        allow_model_downloads=allow_model_downloads,
                    )
                    embedding_provider = SentenceTransformerEmbeddingProvider(
                        run_settings,
                        local_files_only=not allow_model_downloads,
                    )
                    llm_provider = TransformersLlmProvider(
                        run_settings,
                        local_files_only=not allow_model_downloads,
                    )
                services = create_services(
                    run_settings,
                    embedding_provider=embedding_provider,
                    llm_provider=llm_provider,
                )
            dataset = load_dataset_by_id(
                dataset_root,
                dataset_id,
                documents=services.documents,
                reserved_input_bytes=threshold_input_bytes,
            )
            if mode == "real" and any(case.fault is not None for case in dataset.cases):
                raise AuxiliaryFileError("operational_fault_not_allowed_in_real_mode")
            return _execute(
                services,
                dataset,
                dataset_root=dataset_root,
                suite=suite,
                mode=mode,
                report_id=report_id,
                allow_model_downloads=allow_model_downloads,
                performance_warm_repetitions=performance_warm_repetitions,
                timing_observer=timing_observer,
            )


def _open_existing_services(
    settings: Settings,
    *,
    embedding_provider: EmbeddingProvider,
    llm_provider: LlmProvider,
) -> AeroDocsServices:
    """Compose services over existing stores without initializing any root."""
    documents = DocumentService.open_existing(settings)
    retrieval = RetrievalService.open_existing(settings, embedding_provider)
    answers = AnswerService(settings, retrieval, llm_provider)
    return AeroDocsServices(
        settings=settings,
        documents=documents,
        retrieval=retrieval,
        answers=answers,
    )


def _execute(
    services: AeroDocsServices,
    dataset: BenchmarkDataset,
    *,
    dataset_root: Path,
    suite: EvaluationSuite,
    mode: EvaluationMode,
    report_id: str,
    allow_model_downloads: bool,
    performance_warm_repetitions: int,
    timing_observer: TimingObserver | None,
) -> EvaluationReportV2:
    if not 1 <= performance_warm_repetitions <= 100:
        raise AuxiliaryFileError("invalid_performance_repetitions")
    started_at = datetime.now(UTC)
    run_started = time.perf_counter()
    provider_kind: Literal["fake", "real"] = "fake" if mode == "offline" else "real"
    timing_samples: list[PhaseTimingSample] = []
    timing_lifecycle = TimingLifecycle()
    source_map = {source.source_id: source for source in dataset.sources}
    index_started = time.perf_counter()
    index = _resolve_corpus(
        services,
        dataset,
        dataset_root=dataset_root,
        rebuild=not all(source.kind == "manifest_only" for source in dataset.sources),
    )
    timing_samples.append(
        PhaseTimingSample.unknown_state(
            phase="index_open",
            provider_kind=provider_kind,
            repetition=1,
            duration_seconds=time.perf_counter() - index_started,
        )
    )
    if mode == "real" and any(case.fault is not None for case in dataset.cases):
        raise AuxiliaryFileError("operational_fault_not_allowed_in_real_mode")
    selected = [
        case
        for case in dataset.cases
        if suite == "complete"
        or case.suite == suite
        or (
            suite == "performance"
            and case.population == "model_quality"
            and case.fault is None
        )
    ]
    performance_enabled = suite in {"performance", "complete"}
    performance_cases = [
        case for case in selected if _is_performance_eligible_case(case)
    ]
    planned_performance_repetitions = 0
    successful_performance_measurements = 0
    failed_performance_measurements = 0
    case_reports: list[CaseReport] = []
    retrieval_observations: list[RetrievalObservation] = []
    answer_observations: list[AnswerObservation] = []
    for case in dataset.cases:
        is_selected = case in selected
        if not is_selected:
            case_reports.append(
                CaseReport(
                    case_id=case.case_id,
                    case_type=(
                        "retrieval"
                        if isinstance(case, RetrievalBenchmarkCase)
                        else "answer"
                    ),
                    category=case.suite,
                    population=case.population,
                    selected=False,
                    eligible=False,
                    passed=None,
                    exclusion_reason="suite_not_selected",
                    outcome="not_selected",
                    expected_outcome=(
                        "results"
                        if isinstance(case, RetrievalBenchmarkCase)
                        else case.expected_outcome
                    ),
                    observed_outcome="not_selected",
                    metrics=[],
                )
            )
            if isinstance(case, RetrievalBenchmarkCase):
                retrieval_observations.append(
                    RetrievalObservation(
                        case_id=case.case_id,
                        population=case.population,
                        selected=False,
                        execution_succeeded=False,
                    )
                )
            elif isinstance(case, AnswerBenchmarkCase):
                answer_observations.append(
                    AnswerObservation(
                        case_id=case.case_id,
                        population=case.population,
                        selected=False,
                        execution_succeeded=False,
                        expected_outcome=case.expected_outcome,
                        observed_outcome="other_operational_failure",
                        complete_support_set=case.complete_support_set,
                        gold_claim_count=len(case.gold_claims),
                        required_gold_claims=sum(
                            item.required for item in case.gold_claims
                        ),
                    )
                )
            continue
        if case.fault is not None:
            fault_report = _execute_fault(services, case)
            case_reports.append(fault_report)
            if isinstance(case, RetrievalBenchmarkCase):
                retrieval_observations.append(
                    RetrievalObservation(
                        case_id=case.case_id,
                        population=case.population,
                        execution_succeeded=False,
                        failure_category=fault_report.failure_category,
                    )
                )
            else:
                answer_observations.append(
                    AnswerObservation(
                        case_id=case.case_id,
                        population=case.population,
                        execution_succeeded=False,
                        expected_outcome=case.fault,
                        observed_outcome=case.fault
                        if fault_report.fault_exercised
                        else "other_operational_failure",
                        failure_category=fault_report.failure_category,
                    )
                )
            continue
        if isinstance(case, RetrievalBenchmarkCase):
            repetitions = (
                performance_warm_repetitions
                + int(
                    timing_lifecycle.state("retrieval_total") != "successfully_invoked"
                )
                if performance_enabled and _is_performance_eligible_case(case)
                else 1
            )
            retrieval_observation = build_retrieval_observation(
                case, None, source_map, failure_category="retrieval_not_executed"
            )
            outcome = "operational_failure"
            failure_category: str | None = "retrieval_not_executed"
            for repetition in range(1, repetitions + 1):
                if performance_enabled and _is_performance_eligible_case(case):
                    planned_performance_repetitions += 1
                category = timing_lifecycle.begin("retrieval_total")
                provider = _TracingEmbeddingProvider(
                    services.retrieval.provider,
                    ServiceTrace(),
                    timing_samples=timing_samples,
                    timing_lifecycle=timing_lifecycle,
                    provider_kind=provider_kind,
                    repetition=repetition,
                    safe_id=case.case_id,
                )
                timed_retrieval = RetrievalService.open_existing(
                    services.settings,
                    provider,
                )
                timed_retrieval.indexes = _TimingIndexStore(  # type: ignore[assignment]
                    timed_retrieval.indexes,
                    timing_samples,
                    timing_lifecycle,
                    provider_kind,
                    repetition,
                )
                call_started = time.perf_counter()
                call_succeeded = False
                try:
                    search_response = timed_retrieval.search(
                        SearchRequest(
                            query=case.query,
                            document_ids=[
                                source_map[source_id].document_id
                                for source_id in (case.document_source_ids or [])
                            ]
                            or None,
                        )
                    )
                    measured_retrieval_observation = build_retrieval_observation(
                        case, search_response, source_map
                    )
                    if repetition == 1:
                        retrieval_observation = measured_retrieval_observation
                        outcome = "results" if search_response.results else "no_results"
                        failure_category = None
                    call_succeeded = True
                except RetrievalError as error:
                    if repetition == 1:
                        retrieval_observation = build_retrieval_observation(
                            case, None, source_map, failure_category=error.code
                        )
                        outcome = "operational_failure"
                        failure_category = error.code
                finally:
                    timing_lifecycle.finish(
                        "retrieval_total",
                        succeeded=call_succeeded,
                    )
                    timing_samples.append(
                        PhaseTimingSample(
                            phase="retrieval_total",
                            provider_kind=provider_kind,
                            measurement_category=category,
                            success_category=(
                                "success" if call_succeeded else "failure"
                            ),
                            repetition=repetition,
                            duration_seconds=time.perf_counter() - call_started,
                        )
                    )
                    if performance_enabled and _is_performance_eligible_case(case):
                        if call_succeeded:
                            successful_performance_measurements += 1
                        else:
                            failed_performance_measurements += 1
            retrieval_observations.append(retrieval_observation)
            audit_id = None
        elif isinstance(case, AnswerBenchmarkCase):
            document_ids = sorted(
                {
                    source_map[source_id].document_id
                    for source_id in (case.document_source_ids or [])
                }
                or {
                    source_map[target.source_id].document_id
                    for claim in case.gold_claims
                    for target in claim.support_targets
                }
                or {
                    source_map[target.source_id].document_id
                    for target in case.conflict_targets
                }
            )
            repetitions = (
                performance_warm_repetitions
                + int(timing_lifecycle.state("answer_total") != "successfully_invoked")
                if performance_enabled and _is_performance_eligible_case(case)
                else 1
            )
            answer_observation = AnswerObservation(
                case_id=case.case_id,
                population=case.population,
                execution_succeeded=False,
                expected_outcome=case.expected_outcome,
                observed_outcome="other_operational_failure",
            )
            outcome = "other_operational_failure"
            audit_id = None
            failure_category = "answer_not_executed"
            for repetition in range(1, repetitions + 1):
                if performance_enabled and _is_performance_eligible_case(case):
                    planned_performance_repetitions += 1
                category = timing_lifecycle.begin("answer_total")
                trace = ServiceTrace()
                traced_retrieval = RetrievalService.open_existing(
                    services.settings,
                    _TracingEmbeddingProvider(
                        services.retrieval.provider,
                        trace,
                        timing_samples=timing_samples,
                        timing_lifecycle=timing_lifecycle,
                        provider_kind=provider_kind,
                        repetition=repetition,
                        safe_id=case.case_id,
                    ),
                )
                traced_retrieval.indexes = _TimingIndexStore(  # type: ignore[assignment]
                    traced_retrieval.indexes,
                    timing_samples,
                    timing_lifecycle,
                    provider_kind,
                    repetition,
                )
                traced_answer = AnswerService(
                    services.settings,
                    _TracingRetrievalService(  # type: ignore[arg-type]
                        traced_retrieval,
                        trace,
                    ),
                    _TracingLlmProvider(
                        services.answers.provider,
                        trace,
                        timing_samples=timing_samples,
                        timing_lifecycle=timing_lifecycle,
                        provider_kind=provider_kind,
                        repetition=repetition,
                        safe_id=case.case_id,
                    ),
                )
                traced_answer.audits = _TimingAuditStore(  # type: ignore[assignment]
                    traced_answer.audits,
                    timing_samples,
                    timing_lifecycle,
                    provider_kind,
                    repetition,
                )
                call_started = time.perf_counter()
                call_succeeded = False
                try:
                    answer_response = traced_answer.answer(
                        AnswerRequest(
                            question=case.question,
                            document_ids=document_ids or None,
                        )
                    )
                    validation_started = time.perf_counter()
                    measured_answer_observation = build_answer_observation(
                        case, answer_response, source_map, trace
                    )
                    timing_samples.append(
                        PhaseTimingSample(
                            phase="evaluation_observation",
                            provider_kind=provider_kind,
                            measurement_category=timing_lifecycle.begin(
                                "evaluation_observation"
                            ),
                            repetition=repetition,
                            duration_seconds=(time.perf_counter() - validation_started),
                        )
                    )
                    timing_lifecycle.finish(
                        "evaluation_observation",
                        succeeded=True,
                    )
                    if repetition == 1:
                        answer_observation = measured_answer_observation
                        outcome = answer_observation.observed_outcome
                        audit_id = answer_response.audit_id
                        failure_category = None
                    call_succeeded = True
                except (RetrievalError, GenerationError) as error:
                    failure_outcome = _failure_outcome(error.code)
                    if repetition == 1:
                        answer_observation = build_answer_observation(
                            case,
                            None,
                            source_map,
                            trace,
                            failure_outcome=failure_outcome,
                            failure_category=error.code,
                        )
                        outcome = failure_outcome
                        audit_id = None
                        failure_category = error.code
                finally:
                    timing_lifecycle.finish("answer_total", succeeded=call_succeeded)
                    timing_samples.append(
                        PhaseTimingSample(
                            phase="answer_total",
                            provider_kind=provider_kind,
                            measurement_category=category,
                            success_category=(
                                "success" if call_succeeded else "failure"
                            ),
                            repetition=repetition,
                            duration_seconds=time.perf_counter() - call_started,
                        )
                    )
                    if performance_enabled and _is_performance_eligible_case(case):
                        if call_succeeded:
                            successful_performance_measurements += 1
                        else:
                            failed_performance_measurements += 1
            answer_observations.append(answer_observation)
        else:
            raise AuxiliaryFileError("unsupported_case")
        case_reports.append(
            CaseReport(
                case_id=case.case_id,
                case_type=(
                    "retrieval"
                    if isinstance(case, RetrievalBenchmarkCase)
                    else "answer"
                ),
                category=case.suite,
                population=case.population,
                selected=True,
                eligible=True,
                passed=(
                    (
                        retrieval_observation.execution_succeeded
                        and set(retrieval_observation.target_grades)
                        <= {
                            target
                            for item in retrieval_observation.ranked_items
                            for target in item.matching_target_ids
                        }
                    )
                    if isinstance(case, RetrievalBenchmarkCase)
                    else (
                        answer_observation.execution_succeeded
                        and answer_observation.expected_outcome
                        == answer_observation.observed_outcome
                        and not answer_observation.unsafe_categories
                    )
                ),
                exclusion_reason=None,
                outcome=outcome,
                expected_outcome=(
                    "results"
                    if isinstance(case, RetrievalBenchmarkCase)
                    else case.expected_outcome
                ),
                observed_outcome=outcome,
                metrics=[],
                safe_audit_id=audit_id,
                failure_category=failure_category,
                unsafe_categories=(
                    answer_observation.unsafe_categories
                    if isinstance(case, AnswerBenchmarkCase)
                    else []
                ),
                retrieval_ranks=(
                    [
                        rank
                        for rank, item in enumerate(
                            retrieval_observation.ranked_items,
                            start=1,
                        )
                        if item.matching_target_ids
                    ]
                    if isinstance(case, RetrievalBenchmarkCase)
                    else []
                ),
            )
        )
    operations = operational_metrics(retrieval_observations, answer_observations)
    timing = aggregate_phase_timings(timing_samples)
    timing_collector = TimingCollector()
    for sample in timing_samples:
        notify_observer(
            timing_observer,
            timing_collector,
            phase=sample.phase,
            safe_id=report_id,
            duration_seconds=sample.duration_seconds,
            category=sample.success_category,
        )
    memory = process_memory_observation()
    performance_metrics = []
    if performance_enabled:
        for definition in performance_metric_definitions():
            if definition.performance_case_kind == "run":
                continue
            phase = definition.required_stage
            if phase is None:
                raise RuntimeError("performance_metric_stage_missing")
            case_type = (
                RetrievalBenchmarkCase
                if definition.performance_case_kind == "retrieval"
                else AnswerBenchmarkCase
            )
            eligible_case_count = sum(
                isinstance(case, case_type) for case in performance_cases
            )
            selected_timings = [
                item
                for item in timing
                if item.phase == phase
                and item.measurement_category == "warm"
                and item.success_category == "success"
            ]
            if selected_timings:
                aggregate = selected_timings[0]
                performance_metrics.append(
                    scalar_record(
                        definition.metric_id,
                        aggregate.median_seconds,
                        eligible_case_count=aggregate.count,
                        total_population_case_count=aggregate.count,
                    )
                )
            else:
                performance_metrics.append(
                    ineligible_scalar_record(
                        definition.metric_id,
                        (
                            "no_eligible_performance_case"
                            if eligible_case_count == 0
                            else "no_successful_performance_measurements"
                        ),
                        total_population_case_count=eligible_case_count,
                    )
                )
        if memory.process_high_water_rss_bytes is not None:
            performance_metrics.append(
                scalar_record(
                    "process_high_water_rss_bytes",
                    memory.process_high_water_rss_bytes,
                    eligible_case_count=len(selected),
                    total_population_case_count=len(selected),
                )
            )
        else:
            performance_metrics.append(
                ineligible_scalar_record(
                    "process_high_water_rss_bytes",
                    "memory_observation_unavailable",
                    total_population_case_count=len(performance_cases),
                )
            )
    if suite == "retrieval":
        metrics = [*retrieval_metrics(retrieval_observations), *operations]
    elif suite in {"answers", "safety"}:
        metrics = [*answer_metrics(answer_observations), *operations]
    elif suite == "complete":
        metrics = [
            *retrieval_metrics(retrieval_observations),
            *answer_metrics(answer_observations),
            *operations,
            *performance_metrics,
        ]
    else:
        metrics = [*operations, *performance_metrics]
    finished_at = datetime.now(UTC)
    processing_profile = services.documents.processing_profile
    provider_identity = services.answers.provider.identity()
    return EvaluationReportV2(
        run_id=uuid4().hex,
        report_id=report_id,
        mode=mode,
        suite=suite,
        dataset_id=dataset.manifest.dataset_id,
        dataset_version=dataset.manifest.dataset_version,
        dataset_fingerprint=dataset.fingerprint,
        evaluation_profile_id=dataset.manifest.evaluation_profile_id,
        status="completed",
        started_at=started_at,
        finished_at=finished_at,
        observational_duration_seconds=time.perf_counter() - run_started,
        git=git_metadata(Path.cwd()),
        sources=[
            SourceIdentity(
                source_id=source.source_id,
                source_fingerprint=source.source_fingerprint,
                redistribution_mode=(
                    "manifest_only"
                    if source.kind == "manifest_only"
                    else (
                        "benchmark_authored"
                        if source.kind == "synthetic"
                        else "redistributable"
                    )
                ),
                resolved_document_id=source.document_id,
                manifest_fingerprint=source.manifest_fingerprint,
                processing_profile_id=(
                    source.processing_profile_id
                    or processing_profile.processing_profile_id
                ),
                chunking_profile_id=(
                    source.chunking_profile_id
                    or processing_profile.chunking_format_version
                ),
            )
            for source in sorted(dataset.sources, key=lambda item: item.source_id)
        ],
        corpus=CorpusIdentity(
            corpus_fingerprint=index.corpus_fingerprint,
            index_profile_id=index.index_profile_id,
            vector_profile_id=index.vector_profile_id,
            vector_count=index.vector_count,
            document_count=index.source_document_count,
        ),
        providers=ProviderReport(
            provider_kind=provider_kind,
            embedding_model_name=index.model.name,
            embedding_revision=index.model.revision,
            generation_model_name=provider_identity.name,
            generation_revision=provider_identity.revision,
            local_files_only=(mode == "offline" or not allow_model_downloads),
            downloads_allowed=(mode == "real" and allow_model_downloads),
        ),
        environment=environment_summary(),
        settings=SettingsSnapshot(
            retrieval_default_top_k=services.settings.retrieval_default_top_k,
            retrieval_maximum_top_k=services.settings.retrieval_maximum_top_k,
            minimum_similarity_score=services.settings.minimum_similarity_score,
            answer_minimum_similarity_score=(
                services.settings.answer_minimum_similarity_score
            ),
            embedding_normalization=services.settings.embedding_normalization,
            index_format_version=services.settings.index_format_version,
            maximum_generation_output_tokens=(
                services.settings.maximum_generation_output_tokens
            ),
        ),
        metrics=metrics,
        cases=case_reports,
        counts=RunCounts(
            total_cases=len(case_reports),
            selected_cases=sum(case.selected for case in case_reports),
            not_selected_cases=sum(not case.selected for case in case_reports),
            operational_failures=sum(
                case.failure_category is not None for case in case_reports
            ),
            unsafe_cases=sum(bool(case.unsafe_categories) for case in case_reports),
        ),
        timing=timing,
        performance_execution=PerformanceExecutionSummary(
            eligible_case_count=len(performance_cases),
            planned_repetition_count=planned_performance_repetitions,
            successful_measurement_count=successful_performance_measurements,
            failed_measurement_count=failed_performance_measurements,
        ),
        memory=memory,
        instrumentation_failures=timing_collector.instrumentation_failures,
    )


def _is_performance_eligible_case(
    case: RetrievalBenchmarkCase | AnswerBenchmarkCase,
) -> bool:
    """Use one eligibility rule for performance and complete suite execution."""
    return case.population == "model_quality" and case.fault is None


class _TracingEmbeddingProvider:
    def __init__(
        self,
        provider: EmbeddingProvider,
        trace: ServiceTrace,
        *,
        timing_samples: list[PhaseTimingSample] | None = None,
        timing_lifecycle: TimingLifecycle | None = None,
        provider_kind: Literal["fake", "real"] = "fake",
        repetition: int = 1,
        safe_id: str = "evaluation",
    ) -> None:
        self._provider = provider
        self._trace = trace
        self._timing_samples = timing_samples
        self._timing_lifecycle = timing_lifecycle
        self._provider_kind = provider_kind
        self._repetition = repetition
        self._safe_id = safe_id

    def model_identity(self) -> ModelIdentity:
        return self._provider.model_identity()

    def embed_documents(self, texts: Any) -> Any:
        self._trace.embedding_calls += 1
        return self._timed("document_embedding", self._provider.embed_documents, texts)

    def embed_query(self, text: str) -> Any:
        self._trace.embedding_calls += 1
        return self._timed("query_embedding", self._provider.embed_query, text)

    def _timed(self, phase: str, function: Any, value: Any) -> Any:
        category = (
            self._timing_lifecycle.begin(phase)
            if self._timing_lifecycle is not None
            else "unknown"
        )
        started = time.perf_counter()
        succeeded = False
        try:
            result = function(value)
            succeeded = True
            return result
        finally:
            if self._timing_lifecycle is not None:
                self._timing_lifecycle.finish(phase, succeeded=succeeded)
            if self._timing_samples is not None:
                sample = (
                    PhaseTimingSample(
                        phase=phase,
                        provider_kind=self._provider_kind,
                        measurement_category=category,
                        success_category="success" if succeeded else "failure",
                        repetition=self._repetition,
                        duration_seconds=time.perf_counter() - started,
                    )
                    if self._timing_lifecycle is not None
                    else PhaseTimingSample.unknown_state(
                        phase=phase,
                        provider_kind=self._provider_kind,
                        success_category="success" if succeeded else "failure",
                        repetition=self._repetition,
                        duration_seconds=time.perf_counter() - started,
                    )
                )
                self._timing_samples.append(sample)


class _TracingLlmProvider:
    def __init__(
        self,
        provider: LlmProvider,
        trace: ServiceTrace,
        *,
        timing_samples: list[PhaseTimingSample] | None = None,
        timing_lifecycle: TimingLifecycle | None = None,
        provider_kind: Literal["fake", "real"] = "fake",
        repetition: int = 1,
        safe_id: str = "evaluation",
    ) -> None:
        self._provider = provider
        self._trace = trace
        self._timing_samples = timing_samples
        self._timing_lifecycle = timing_lifecycle
        self._provider_kind = provider_kind
        self._repetition = repetition
        self._safe_id = safe_id

    def identity(self) -> ProviderIdentity:
        return self._provider.identity()

    def prepare(self, system_prompt: str, user_prompt: str) -> PreparedGenerationInput:
        self._trace.generation_calls += 1
        category = self._begin("model_input_preparation")
        started = time.perf_counter()
        succeeded = False
        try:
            result = self._provider.prepare(system_prompt, user_prompt)
            succeeded = True
            return result
        finally:
            self._record(
                "model_input_preparation",
                started,
                category=category,
                succeeded=succeeded,
            )

    def count_tokens(self, text: str) -> int:
        return self._provider.count_tokens(text)

    def generate(self, prepared: PreparedGenerationInput) -> str:
        category = self._begin("model_generation")
        started = time.perf_counter()
        succeeded = False
        try:
            result = self._provider.generate(prepared)
            succeeded = True
            return result
        finally:
            self._record(
                "model_generation",
                started,
                category=category,
                succeeded=succeeded,
            )

    def _begin(self, phase: str) -> TimingLifecycleLabel:
        if self._timing_lifecycle is None:
            return "unknown"
        return self._timing_lifecycle.begin(phase)

    def _record(
        self,
        phase: str,
        started: float,
        *,
        category: TimingLifecycleLabel,
        succeeded: bool,
    ) -> None:
        if self._timing_lifecycle is not None:
            self._timing_lifecycle.finish(phase, succeeded=succeeded)
        if self._timing_samples is not None:
            sample = (
                PhaseTimingSample(
                    phase=phase,
                    provider_kind=self._provider_kind,
                    measurement_category=category,
                    success_category="success" if succeeded else "failure",
                    repetition=self._repetition,
                    duration_seconds=time.perf_counter() - started,
                )
                if self._timing_lifecycle is not None
                else PhaseTimingSample.unknown_state(
                    phase=phase,
                    provider_kind=self._provider_kind,
                    success_category="success" if succeeded else "failure",
                    repetition=self._repetition,
                    duration_seconds=time.perf_counter() - started,
                )
            )
            self._timing_samples.append(sample)


class _TracingRetrievalService:
    def __init__(self, retrieval: RetrievalService, trace: ServiceTrace) -> None:
        self._retrieval = retrieval
        self._trace = trace

    def search(self, request: SearchRequest) -> Any:
        self._trace.retrieval_calls += 1
        return self._retrieval.search(request)


class _TimingAuditStore:
    def __init__(
        self,
        store: Any,
        timing_samples: list[PhaseTimingSample],
        timing_lifecycle: TimingLifecycle,
        provider_kind: Literal["fake", "real"],
        repetition: int,
    ) -> None:
        self._store = store
        self._timing_samples = timing_samples
        self._timing_lifecycle = timing_lifecycle
        self._provider_kind = provider_kind
        self._repetition = repetition

    def write(self, record: Any) -> None:
        category = self._timing_lifecycle.begin("audit_persistence")
        started = time.perf_counter()
        succeeded = False
        try:
            self._store.write(record)
            succeeded = True
        finally:
            self._timing_lifecycle.finish("audit_persistence", succeeded=succeeded)
            self._timing_samples.append(
                PhaseTimingSample(
                    phase="audit_persistence",
                    provider_kind=self._provider_kind,
                    measurement_category=category,
                    success_category="success" if succeeded else "failure",
                    repetition=self._repetition,
                    duration_seconds=time.perf_counter() - started,
                )
            )


class _TimingIndexStore:
    """Evaluation-only proxy timing actual FAISS deserialization."""

    def __init__(
        self,
        store: Any,
        timing_samples: list[PhaseTimingSample],
        timing_lifecycle: TimingLifecycle,
        provider_kind: Literal["fake", "real"],
        repetition: int,
    ) -> None:
        self._store = store
        self._timing_samples = timing_samples
        self._timing_lifecycle = timing_lifecycle
        self._provider_kind = provider_kind
        self._repetition = repetition

    def active(self) -> Any:
        category = self._timing_lifecycle.begin("index_deserialization")
        started = time.perf_counter()
        succeeded = False
        try:
            result = self._store.active()
            succeeded = True
            return result
        finally:
            self._timing_lifecycle.finish(
                "index_deserialization",
                succeeded=succeeded,
            )
            self._timing_samples.append(
                PhaseTimingSample(
                    phase="index_deserialization",
                    provider_kind=self._provider_kind,
                    measurement_category=category,
                    success_category="success" if succeeded else "failure",
                    repetition=self._repetition,
                    duration_seconds=time.perf_counter() - started,
                )
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


def _failure_outcome(code: str) -> Any:
    return {
        "generation_output_invalid": "malformed_generation",
        "llm_model_unavailable": "model_unavailable",
        "generation_timeout": "generation_timeout",
        "audit_persistence_failed": "audit_failure",
    }.get(
        code,
        "retrieval_failure"
        if "retrieval" in code or "embedding" in code
        else "other_operational_failure",
    )


def _resolve_corpus(
    services: AeroDocsServices,
    dataset: BenchmarkDataset,
    *,
    dataset_root: Path,
    rebuild: bool,
) -> Any:
    """Resolve source artifacts and targets once for validate and run."""
    source_map = {source.source_id: source for source in dataset.sources}
    for source in dataset.sources:
        if source.kind in {"synthetic", "redistributable"}:
            if source.kind == "synthetic":
                if source.content is None:
                    raise AuxiliaryFileError("source_content_unavailable")
                stream: Any = BytesIO(source.content)
                result = services.documents.ingest(
                    stream,
                    f"{source.source_id}.pdf",
                    services.settings.allowed_pdf_media_type,
                    ingested_at=EVALUATION_INGESTED_AT,
                )
            else:
                if source.relative_file is None:
                    raise AuxiliaryFileError("source_content_unavailable")
                with SecureDirectory(dataset_root) as root:
                    with root.open_directory(dataset.manifest.dataset_id) as bundle:
                        stream = bundle.open_binary(source.relative_file)
                        try:
                            result = services.documents.ingest(
                                stream,
                                f"{source.source_id}.pdf",
                                services.settings.allowed_pdf_media_type,
                                ingested_at=EVALUATION_INGESTED_AT,
                            )
                        finally:
                            stream.close()
            if result.document.document_id != source.document_id:
                raise AuxiliaryFileError("resolved_document_id_mismatch")
    index = (
        services.retrieval.rebuild().index
        if rebuild
        else services.retrieval.active_metadata_without_models()
    )
    document_details: dict[str, ResolvedDocument] = {}
    for source_id, source in source_map.items():
        validated = services.documents.store.validate_document(
            source.document_id,
            page_limit=services.settings.max_pdf_pages,
            chunk_limit=services.settings.maximum_chunks_per_index_build,
        )
        document_details[source_id] = ResolvedDocument(
            manifest_fingerprint=services.documents.store.manifest_sha256(
                source.document_id
            ),
            page_count=validated.manifest.page_count,
            chunk_ids={chunk.chunk_id for chunk in validated.chunks},
            section_occurrences=_section_occurrences(validated.pages),
            processing_profile_id=validated.manifest.processing_profile_id,
            chunking_profile_id=validated.manifest.chunking_format_version,
        )
        if (
            (
                source.processing_profile_id is not None
                and source.processing_profile_id
                != validated.manifest.processing_profile_id
            )
            or (
                source.chunking_profile_id is not None
                and source.chunking_profile_id
                != validated.manifest.chunking_format_version
            )
            or (
                source.manifest_fingerprint is not None
                and source.manifest_fingerprint
                != document_details[source_id].manifest_fingerprint
            )
            or (
                source.kind == "manifest_only"
                and source.allowed_corpus_fingerprint is None
            )
        ):
            raise AuxiliaryFileError("source_profile_identity_mismatch")
        if (
            source.allowed_corpus_fingerprint is not None
            and source.allowed_corpus_fingerprint != index.corpus_fingerprint
        ):
            raise AuxiliaryFileError("corpus_identity_mismatch")
    validate_resolved_targets(
        dataset,
        document_details=document_details,
        corpus_fingerprint=index.corpus_fingerprint,
    )
    return index


def _section_occurrences(pages: list[Any]) -> tuple[SectionOccurrence, ...]:
    occurrences: list[SectionOccurrence] = []
    for page in sorted(pages, key=lambda item: item.page_number):
        for _offset, raw_heading in find_headings(page.text):
            occurrences.append(
                SectionOccurrence(
                    heading=normalize_section_heading(raw_heading),
                    page_start=page.page_number,
                    page_end=page.page_number,
                    occurrence_id=f"section-{len(occurrences) + 1}",
                )
            )
    return tuple(occurrences)


class _FailingAuditStore:
    def write(self, _record: object) -> None:
        raise AuditPersistenceError


def _execute_fault(
    services: AeroDocsServices,
    case: RetrievalBenchmarkCase | AnswerBenchmarkCase,
) -> CaseReport:
    requested = case.fault
    assert requested is not None
    observed: str | None = None
    try:
        if requested == "retrieval_failure":
            retrieval = RetrievalService(
                services.settings,
                FailingOfflineEmbeddingProvider(),
            )
            retrieval.search(SearchRequest(query=_fault_query(case)))
        else:
            providers = {
                "malformed_generation": MalformedOfflineLlmProvider(),
                "model_unavailable": UnavailableOfflineLlmProvider(),
                "generation_timeout": TimeoutOfflineLlmProvider(),
                "audit_failure": OfflineLlmProvider(),
            }
            answer = AnswerService(
                services.settings,
                services.retrieval,
                providers[requested],
            )
            if requested == "audit_failure":
                answer.audits = _FailingAuditStore()  # type: ignore[assignment]
            answer.answer(AnswerRequest(question=_fault_query(case)))
    except (RetrievalError, GenerationError) as error:
        observed = error.code
    expected = {
        "retrieval_failure": "embedding_model_unavailable",
        "malformed_generation": "generation_output_invalid",
        "model_unavailable": "llm_model_unavailable",
        "generation_timeout": "generation_timeout",
        "audit_failure": "audit_persistence_failed",
    }[requested]
    exercised = observed == expected
    return CaseReport(
        case_id=case.case_id,
        case_type=(
            "retrieval" if isinstance(case, RetrievalBenchmarkCase) else "answer"
        ),
        category=case.suite,
        population=case.population,
        selected=True,
        eligible=False,
        passed=exercised,
        exclusion_reason="operational_fault",
        outcome="operational_failure" if exercised else "fault_mismatch",
        expected_outcome=requested,
        observed_outcome=(requested if exercised else "other_operational_failure"),
        metrics=[],
        failure_category=observed or "no_failure_observed",
        requested_fault=requested,
        fault_exercised=exercised,
    )


def _fault_query(case: RetrievalBenchmarkCase | AnswerBenchmarkCase) -> str:
    return case.query if isinstance(case, RetrievalBenchmarkCase) else case.question
