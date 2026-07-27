"""Canonical AeroDocs Milestone 6 evaluation command line."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence

from app.core.config import Settings
from app.evaluation.benchmark import preflight_dataset_by_id
from app.evaluation.config import EvaluationSettings
from app.evaluation.offline import NetworkDeniedError
from app.evaluation.orchestrator import run_evaluation, validate_bundle
from app.evaluation.reporting import (
    EvaluationReportPersistenceError,
    ThresholdProvenance,
    apply_gates,
    compare_reports,
    comparison_artifact,
    load_dataset_threshold_selection,
    mandatory_gates,
    select_threshold_gates,
    validate_report_root_for_publication,
    write_comparison,
    write_report,
)
from app.evaluation.security import (
    MAXIMUM_EVALUATION_DATASET_BYTES,
    AuxiliaryFileError,
    safe_identifier,
)

EXIT_SUCCESS = 0
EXIT_INVALID = 2
EXIT_OPERATIONAL = 3
EXIT_GATE = 4
EXIT_REPORT = 5


def main(arguments: Sequence[str] | None = None) -> int:
    """Parse and execute a canonical evaluation command."""
    parser = _parser()
    try:
        parsed = parser.parse_args(arguments)
        if parsed.command == "validate":
            safe_identifier(parsed.dataset)
        elif parsed.command == "run":
            safe_identifier(parsed.dataset)
            safe_identifier(parsed.report_id)
            if parsed.thresholds is not None:
                safe_identifier(parsed.thresholds)
        else:
            safe_identifier(parsed.baseline)
            safe_identifier(parsed.candidate)
            safe_identifier(parsed.report_id)
        if os.environ.get("AERODOCS_EVALUATION_WORKER") != "1" and (
            parsed.command == "validate"
            or (parsed.command == "run" and parsed.mode == "offline")
        ):
            return _isolated_worker(list(arguments or sys.argv[1:]))
        settings = Settings()
        evaluation_settings = EvaluationSettings()
        if parsed.command == "validate":
            preflight = preflight_dataset_by_id(
                evaluation_settings.dataset_root,
                parsed.dataset,
            )
            default_thresholds = (
                load_dataset_threshold_selection(
                    evaluation_settings.dataset_root,
                    parsed.dataset,
                    preflight.manifest.thresholds_file,
                    remaining_input_bytes=(
                        MAXIMUM_EVALUATION_DATASET_BYTES - preflight.input_bytes
                    ),
                )
                if preflight.manifest.thresholds_file is not None
                else None
            )
            validate_bundle(
                settings,
                evaluation_settings.dataset_root,
                parsed.dataset,
                threshold_input_bytes=(
                    0 if default_thresholds is None else default_thresholds.input_bytes
                ),
            )
            print("evaluation_validation=success")
            return EXIT_SUCCESS
        if parsed.command == "run":
            if parsed.mode == "offline" and parsed.allow_model_downloads:
                raise AuxiliaryFileError("downloads_not_allowed_offline")
            preflight = preflight_dataset_by_id(
                evaluation_settings.dataset_root,
                parsed.dataset,
            )
            threshold_reference = (
                f"{parsed.thresholds}.json"
                if parsed.thresholds is not None
                else preflight.manifest.thresholds_file
            )
            loaded_thresholds = (
                load_dataset_threshold_selection(
                    evaluation_settings.dataset_root,
                    parsed.dataset,
                    threshold_reference,
                    remaining_input_bytes=(
                        MAXIMUM_EVALUATION_DATASET_BYTES - preflight.input_bytes
                    ),
                )
                if threshold_reference is not None
                else None
            )
            selected_declared = (
                select_threshold_gates(
                    loaded_thresholds.thresholds,
                    parsed.suite,
                )
                if loaded_thresholds is not None
                else []
            )
            selected_optional = [
                gate
                for gate in selected_declared
                if (gate.population, gate.metric_id)
                != ("safety_quality", "unsafe_answer_count")
            ]
            applied_thresholds = mandatory_gates(parsed.suite, selected_optional)
            mandatory_gate_count = len(applied_thresholds) - len(selected_optional)
            report = run_evaluation(
                settings,
                dataset_root=evaluation_settings.dataset_root,
                dataset_id=parsed.dataset,
                suite=parsed.suite,
                mode=parsed.mode,
                report_id=parsed.report_id,
                allow_model_downloads=parsed.allow_model_downloads,
                performance_warm_repetitions=(
                    evaluation_settings.performance_warm_repetitions
                ),
                threshold_input_bytes=(
                    0 if loaded_thresholds is None else loaded_thresholds.input_bytes
                ),
            )
            metric_identities = {
                (metric.population, metric.metric_id) for metric in report.metrics
            }
            if any(
                (gate.population, gate.metric_id) not in metric_identities
                for gate in applied_thresholds
            ):
                raise RuntimeError("selected_gate_metric_not_emitted")
            gate_results = apply_gates(
                report.metrics,
                applied_thresholds,
            )
            gate_failed = any(not result.passed for result in gate_results)
            threshold_provenance = (
                ThresholdProvenance(
                    source="mandatory_invariants_only",
                    declared_gate_count=0,
                    selected_optional_gate_count=0,
                    mandatory_gate_count=mandatory_gate_count,
                    out_of_scope_gate_count=0,
                    applied_gate_count=len(applied_thresholds),
                )
                if loaded_thresholds is None
                else ThresholdProvenance(
                    source=(
                        "explicit_override"
                        if parsed.thresholds is not None
                        else "dataset_default"
                    ),
                    threshold_set_id=(loaded_thresholds.thresholds.threshold_set_id),
                    threshold_schema_version=(
                        loaded_thresholds.thresholds.schema_version
                    ),
                    threshold_file_identity=loaded_thresholds.file_identity,
                    threshold_fingerprint=loaded_thresholds.fingerprint,
                    declared_gate_count=len(loaded_thresholds.thresholds.gates),
                    selected_optional_gate_count=len(selected_optional),
                    mandatory_gate_count=mandatory_gate_count,
                    out_of_scope_gate_count=(
                        len(loaded_thresholds.thresholds.gates) - len(selected_declared)
                    ),
                    applied_gate_count=len(applied_thresholds),
                )
            )
            report = report.model_copy(
                update={
                    "gates": gate_results,
                    "status": "gate_failed" if gate_failed else "completed",
                    "threshold_provenance": threshold_provenance,
                }
            )
            try:
                write_report(evaluation_settings.report_root, report)
            except EvaluationReportPersistenceError:
                print("evaluation_run=report_persistence_failure")
                return EXIT_REPORT
            if gate_failed:
                print("evaluation_run=regression_gate_failure")
                return EXIT_GATE
            print("evaluation_run=success")
            return EXIT_SUCCESS
        try:
            validate_report_root_for_publication(evaluation_settings.report_root)
            result = compare_reports(
                evaluation_settings.report_root,
                parsed.baseline,
                parsed.candidate,
            )
            write_comparison(
                evaluation_settings.report_root,
                comparison_artifact(
                    parsed.report_id,
                    parsed.baseline,
                    parsed.candidate,
                    result,
                ),
            )
        except EvaluationReportPersistenceError:
            print("evaluation_compare=report_persistence_failure")
            return EXIT_REPORT
        print("evaluation_compare=success")
        return EXIT_SUCCESS
    except (AuxiliaryFileError, ValueError):
        print("evaluation=invalid_dataset_or_configuration")
        return EXIT_INVALID
    except NetworkDeniedError:
        print("evaluation=operational_failure")
        return EXIT_OPERATIONAL
    except Exception:
        print("evaluation=operational_failure")
        return EXIT_OPERATIONAL


def _isolated_worker(arguments: list[str]) -> int:
    """Run network-denied commands in a dedicated fixed-module subprocess."""
    environment = os.environ.copy()
    environment["AERODOCS_EVALUATION_WORKER"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "app.evaluation", *arguments],
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError):
        print("evaluation=operational_failure")
        return EXIT_OPERATIONAL
    allowed = {0, 2, 3, 4, 5}
    code = result.returncode if result.returncode in allowed else EXIT_OPERATIONAL
    messages = {
        EXIT_SUCCESS: (
            "evaluation_validation=success"
            if arguments[0] == "validate"
            else "evaluation_run=success"
        ),
        EXIT_INVALID: "evaluation=invalid_dataset_or_configuration",
        EXIT_OPERATIONAL: "evaluation=operational_failure",
        EXIT_GATE: "evaluation_run=regression_gate_failure",
        EXIT_REPORT: "evaluation_run=report_persistence_failure",
    }
    print(messages[code])
    return code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--dataset", required=True)
    run = commands.add_parser("run")
    run.add_argument(
        "--suite",
        choices=("retrieval", "answers", "safety", "performance", "complete"),
        required=True,
    )
    run.add_argument("--dataset", required=True)
    run.add_argument("--mode", choices=("offline", "real"), required=True)
    run.add_argument("--report-id", required=True)
    run.add_argument("--thresholds")
    run.add_argument("--allow-model-downloads", action="store_true")
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--report-id", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
